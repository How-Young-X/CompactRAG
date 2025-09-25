import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import faiss
import numpy as np
import sqlite3
import os
import json
from tqdm import tqdm
import argparse
import pickle
import warnings

# Suppress CUBLAS warnings
warnings.filterwarnings("ignore", message=".*gemm_and_bias error.*")
warnings.filterwarnings("ignore", message=".*CUBLAS_STATUS_INVALID_VALUE.*")


class FaissRetriever:
    def __init__(self, model_path='models/facebook/contriever', device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Optimize CUDA settings to reduce CUBLAS warnings
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Set memory fraction to avoid fragmentation
            torch.cuda.set_per_process_memory_fraction(0.9)
            # Enable memory efficient attention if available
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
        
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode for inference
        
        self.index = None
        self.embedding_dim = None
        self.db_conn = None  # SQLite connection
        self.doc_counter = 0  # Keep track of total inserted documents

    def mean_pooling(self, token_embeddings, mask):
        """Mean pooling to get sentence embeddings"""
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def encode(self, sentences, batch_size=32):
        """Encode sentences in batches into normalized embeddings"""
        all_embs = []
        
        # Use smaller batch size for CUDA to avoid memory issues
        if self.device == "cuda" and batch_size > 16:
            batch_size = min(batch_size, 16)
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                # Use autocast for mixed precision to reduce memory pressure
                if self.device == "cuda":
                    with torch.cuda.amp.autocast(enabled=True):
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)
            
            embeddings = self.mean_pooling(outputs[0], inputs['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embs.append(embeddings.cpu().numpy())
            
            # Clear CUDA cache periodically to prevent memory fragmentation
            if self.device == "cuda" and i % (batch_size * 4) == 0:
                torch.cuda.empty_cache()
                
        return np.vstack(all_embs)

    def init_index(self, dim):
        """Initialize FAISS index"""
        self.embedding_dim = dim
        self.index = faiss.IndexFlatIP(dim)

    def init_db(self, db_path="meta.db"):
        """Initialize SQLite database for metadata storage"""
        self.db_conn = sqlite3.connect(db_path)
        cur = self.db_conn.cursor()
        
        # Always create fresh table since we clear everything at start
        cur.execute("""
            CREATE TABLE qa (
                faiss_id INTEGER PRIMARY KEY,
                question TEXT,
                answer TEXT,
                passage TEXT,
                title TEXT
            )""")
        self.db_conn.commit()
        print(f"Created fresh database table at {db_path}")

    def insert_meta(self, faiss_id, question, answer, passage=None, title=None):
        """Insert one record into SQLite database"""
        cur = self.db_conn.cursor()
        cur.execute("INSERT INTO qa (faiss_id, question, answer, passage, title) VALUES (?, ?, ?, ?, ?)",
                    (faiss_id, question, answer, passage, title))
        self.db_conn.commit()

    def get_meta(self, faiss_id):
        """Retrieve metadata by faiss_id from SQLite"""
        cur = self.db_conn.cursor()
        cur.execute("SELECT question, answer, passage, title FROM qa WHERE faiss_id=?", (faiss_id,))
        row = cur.fetchone()
        if row:
            return {
                "question": row[0], 
                "meta": {
                    "answer": row[1],
                    "passage": row[2],
                    "title": row[3]
                }
            }
        return None

    def build_from_file(self, filepath, batch_size=32, db_path="meta.db",
                        checkpoint_dir="checkpoints", save_every=1_000_000):
        """
        Build FAISS index from a JSONL file.
        Each line should contain: {"qa": [{"question": "xxx", "answer": "yyy"}], "passage": "text", "title": "title"}
        Supports large files by streaming line by line.
        Periodically saves checkpoints every `save_every` documents.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.init_db(db_path)
        questions, answers, passages, titles = [], [], [], []
        with open(filepath, "r", encoding="utf8") as f:
            for line in tqdm(f, desc="Building index", unit="lines"):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                qa = obj.get("qa",[])
                if not qa:
                    qa = []
                
                # Extract metadata
                passage = obj.get("passage", "")
                title = obj.get("title", "")

                if len(qa) < 1:
                    continue
                for item in qa:
                    question = item["question"]
                    answer = item["answer"]
                    questions.append(question)
                    answers.append(answer)
                    passages.append(passage)
                    titles.append(title)

                # Encode and add to index in batches
                if len(questions) >= batch_size:
                    self._process_batch(questions, answers, passages, titles)
                    questions, answers, passages, titles = [], [], [], []

                # Save checkpoint periodically
                if self.doc_counter > 0 and self.doc_counter % save_every == 0:
                    ckpt_index = os.path.join(checkpoint_dir, f"index_{self.doc_counter}.faiss")
                    ckpt_meta = os.path.join(checkpoint_dir, f"meta_{self.doc_counter}.pkl")
                    self.save(ckpt_index, ckpt_meta)
                    print(f"[Checkpoint] Saved at {self.doc_counter} qa")

        # Process remaining data
        if questions:
            self._process_batch(questions, answers, passages, titles)

        # Final save
        final_index = os.path.join(checkpoint_dir, f"final_index.faiss")
        final_meta = os.path.join(checkpoint_dir, f"final_meta.pkl")
        self.save(final_index, final_meta)
        print(f"[Final] Index built with {self.doc_counter} qas")

    def _process_batch(self, questions, answers, passages, titles):
        """Helper to encode and insert a batch of documents"""
        # sentences = [t+"\n"+p+"\n\n"+q+"\n"+a for t,p,q,a in zip(titles,passages, questions,answers)]
        # sentences = [q for q in questions]
        sentences = [q+"\n"+a for q,a in zip(questions,answers)]
        embs = self.encode(sentences=sentences, batch_size=len(questions))
        if self.index is None:
            self.init_index(embs.shape[1])
        
        # Add embeddings to FAISS index
        self.index.add(embs)

        # Insert metadata
        for i, q in enumerate(questions):
            self.insert_meta(
                self.doc_counter, 
                q, 
                answers[i], 
                passages[i] if i < len(passages) else None,
                titles[i] if i < len(titles) else None,
            )
            self.doc_counter += 1

    def search(self, query, topk=5, fetch_k=None):
        """
        Search FAISS index and return unique passages.
        - topk: number of unique passages to return
        - fetch_k: number of raw FAISS candidates to fetch (default = 5 * topk)
        """
        if self.index is None:
            raise ValueError("Index is empty. Build it first.")
        if fetch_k is None:
            fetch_k = topk * 10  # oversample to reduce risk of duplicates
        query_emb = self.encode([query])
        scores, idxs = self.index.search(query_emb, fetch_k)

        seen_texts = set()
        results = []
        for j, i in enumerate(idxs[0]):
            doc = self.get_meta(int(i))
            if not doc:
                continue
            if doc["question"] in seen_texts:
                continue  # skip duplicates
            seen_texts.add(doc["question"])
            results.append({
                "question": doc["question"],
                "answer": doc["meta"]["answer"],
                "passage": doc["meta"]["passage"],
                "title": doc["meta"]["title"],
                "score": float(scores[0][j])
            })
            if len(results) >= topk:
                break
        return results

    def save(self, index_path="faiss.index", meta_info_path="meta_info.pkl"):
        """Save FAISS index and meta information"""
        if self.index is None:
            raise ValueError("Index is empty. Build it first.")
        faiss.write_index(self.index, index_path)
        with open(meta_info_path, "wb") as f:
            pickle.dump({"dim": self.embedding_dim, "count": self.doc_counter}, f)

    def load(self, index_path="faiss.index", meta_info_path="meta_info.pkl", db_path="meta.db"):
        """Load FAISS index and reopen SQLite DB"""
        if not os.path.exists(index_path) or not os.path.exists(meta_info_path):
            raise FileNotFoundError("Index or metadata file not found.")
        self.index = faiss.read_index(index_path)
        with open(meta_info_path, "rb") as f:
            meta = pickle.load(f)
        self.embedding_dim = meta["dim"]
        self.doc_counter = meta.get("count", 0)
        self.init_db(db_path)


if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(description="Build FAISS index for QA datasets")
    parser.add_argument("--dataset", type=str, default="musique", 
                       choices=["musique", "hotpotqa", "2wiki"],
                       help="Dataset to build index for")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for processing")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--model", type=str, default="llama8b",choices=["llama8b","gpt-4","qwen3-32b"])
    args = parser.parse_args()

    dataset = args.dataset
    model= args.model
    index_dir = f"data/index/QA/{model}/{dataset}"
    os.makedirs(index_dir, exist_ok=True)
    
    # Clear existing index and cache files
    index_file = f"{index_dir}/corpus.index"
    db_file = f"{index_dir}/corpus_meta.db"
    checkpoint_dir = f"{index_dir}/checkpoints"
    
    print(f"Clearing existing index and cache files for dataset '{dataset}'...")
    
    # Remove existing files
    if os.path.exists(index_file):
        os.remove(index_file)
        print(f"Removed existing index: {index_file}")
    
    if os.path.exists(db_file):
        os.remove(db_file)
        print(f"Removed existing database: {db_file}")
    
    # Remove checkpoint directory and all its contents
    if os.path.exists(checkpoint_dir):
        import shutil
        shutil.rmtree(checkpoint_dir)
        print(f"Removed checkpoint directory: {checkpoint_dir}")
    
    # Remove any other cache files
    for pattern in ["*.pkl", "*.faiss"]:
        import glob
        for file in glob.glob(f"{index_dir}/{pattern}"):
            os.remove(file)
            print(f"Removed cache file: {file}")
    
    print("All existing index and cache files cleared.")
    
    # Check if corpus file exists
    corpus_file = f"data/QA/{dataset}/{model}-{dataset}-qa.jsonl"
    if not os.path.exists(corpus_file):
        print(f"Corpus file not found: {corpus_file}")
        sys.exit(1)
    
    print(f"Building index for dataset: {dataset}")
    print(f"Corpus file: {corpus_file}")
    print(f"Index directory: {index_dir}")
    
    try:
        retriever = FaissRetriever()
        retriever.build_from_file(
            corpus_file,
            batch_size=args.batch_size,
            db_path=db_file,
            checkpoint_dir=f"{index_dir}/checkpoints",
            save_every=args.save_every
        )
        retriever.save(index_file, f"{index_dir}/corpus_meta.pkl")
        print(f"Successfully built index for dataset '{dataset}'")
        
    except KeyboardInterrupt:
        print("\nIndex building interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error building index: {e}")
        sys.exit(1)

