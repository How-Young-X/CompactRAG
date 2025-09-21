import json
import os
import jsonlines
import re
import hashlib
import logging
import random
from html import unescape
from collections import Counter, defaultdict


# Setup logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)

SAMPLE_NUM=200


def ensure_dir(path):
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def remove_a_tags(html_content):
    """Remove <a> tags from HTML text."""
    clean_html = unescape(html_content)
    clean_html = re.sub(r'<a\b[^>]*>', '', clean_html, flags=re.IGNORECASE)
    clean_html = re.sub(r'</a>', '', clean_html, flags=re.IGNORECASE)
    return clean_html


class DeduplicatedWriter:
    """Deduplicated JSONL writer that avoids writing duplicate (title, passage) pairs."""

    def __init__(self, output_path, dedup_index_path=None):
        ensure_dir(os.path.dirname(output_path))
        self.output_path = output_path
        self.dedup_index_path = dedup_index_path or output_path + ".dedup_index"
        self.seen = set()

        # Load existing deduplication index if available
        if os.path.exists(self.dedup_index_path):
            with open(self.dedup_index_path, "r", encoding="utf8") as f:
                for line in f:
                    self.seen.add(line.strip())

        # Open JSONL writer
        self.writer = jsonlines.open(self.output_path, mode="a")

    def _make_key(self, title, passage):
        """Generate a unique hash key for (title, passage)."""
        text = f"{title}::{passage}"
        return hashlib.md5(text.encode("utf8")).hexdigest()

    def write(self, title, passage):
        """Write a new entry if it has not been seen before."""
        key = self._make_key(title, passage)
        if key in self.seen:
            return False
        self.seen.add(key)
        self.writer.write({"title": title, "passage": passage})

        # Persist the key for future runs
        with open(self.dedup_index_path, "a", encoding="utf8") as f:
            f.write(key + "\n")
        return True

    def close(self):
        self.writer.close()


def save_test_samples(samples, out_dir, prefix):
    """Save sampled test data (question, answer)."""
    ensure_dir(out_dir)
    test_file = os.path.join(out_dir, f"{prefix}_test.jsonl")
    with jsonlines.open(test_file, "w") as writer:
        for s in samples:
            writer.write({"question": s["question"], "answer": s["answer"],"type": s["type"]})

def save_musique_test_samples(samples, out_dir, prefix):
    """Save sampled test data (question, answer)."""
    ensure_dir(out_dir)
    test_file = os.path.join(out_dir, f"{prefix}_test.jsonl")
    with jsonlines.open(test_file, "w") as writer:
        for s in samples:
            writer.write({"question": s["question"], "answer": s["answer"],"hop": s["hop"]})

def save_hotpotqa_test_samples(samples, out_dir, prefix):
    """Save sampled test data (question, answer)."""
    ensure_dir(out_dir)
    test_file = os.path.join(out_dir, f"{prefix}_test.jsonl")
    with jsonlines.open(test_file, "w") as writer:
        for s in samples:
            writer.write({"question": s["question"], "answer": s["answer"],"type": s["type"], "level": s["level"]})



def save_sample_corpus(samples, out_dir, prefix):
    """Save sampled corpus (title, passage) with deduplication."""
    ensure_dir(out_dir)
    corpus_file = os.path.join(out_dir, f"{prefix}_sample_corpus.jsonl")
    seen = set()
    with jsonlines.open(corpus_file, "w") as writer:
        for s in samples:
            for title, passage in s["context"]:
                key = f"{title}::{passage}"
                if key not in seen:
                    writer.write({"title": title, "passage": passage})
                    seen.add(key)


def get_2Wiki_corpus(path, seed=42):
    random.seed(seed) 
    file = os.path.join(path, "dev.json")
    writer = DeduplicatedWriter("data/test/dataset/corpus/2wiki/wash/2wiki_corpus.jsonl")

    all_samples = []
    with open(file, "r", encoding="utf8") as f:
        data = json.load(f)

    for item in data:
        for c in item["context"]:
            title = c[0]
            passage = " ".join(c[1])
            writer.write(title, passage)

        all_samples.append({
            "question": item["question"],
            "answer": item["answer"],
            "context": [(c[0], " ".join(c[1])) for c in item["context"]],
            "type": item["type"]
        })
    writer.close()

    type_counter = Counter([s["type"] for s in all_samples])
    total = sum(type_counter.values())
    print("Type distribution:", type_counter)

    target_num = 250
    per_type_num = {t: round(target_num * cnt / total) for t, cnt in type_counter.items()}

    diff = target_num - sum(per_type_num.values())
    if diff != 0:
        for t, _ in type_counter.most_common(abs(diff)):
            per_type_num[t] += 1 if diff > 0 else -1

    sampled = []
    samples_by_type = defaultdict(list)
    for s in all_samples:
        samples_by_type[s["type"]].append(s)

    for t, n in per_type_num.items():
        n = min(n, len(samples_by_type[t]))
        sampled.extend(random.sample(samples_by_type[t], n))

    save_test_samples(sampled, "data/test/dataset/testset/250/2wiki", "2wiki")
    save_sample_corpus(sampled, "data/test/dataset/testset/250/2wiki", "2wiki")

def get_musique_corpus(path, target_num=250, seed=42):
    random.seed(seed)
    file = os.path.join(path, "musique_ans_v1.0_dev.jsonl")
    writer = DeduplicatedWriter("data/test/dataset/corpus/musique/wash/musique_corpus.jsonl")

    all_samples = []
    with jsonlines.open(file, "r") as f:
        for line in f:
            for item in line["paragraphs"]:
                writer.write(item["title"], item["paragraph_text"])
            hop = line["id"][:4]

            all_samples.append({
                "question": line["question"],
                "answer": line["answer"],
                "context": [(p["title"], p["paragraph_text"]) for p in line["paragraphs"]],
                "hop": hop
            })
    writer.close()

    hop_counter = Counter([s["hop"] for s in all_samples])
    total = sum(hop_counter.values())
    print("Hop distribution:", hop_counter)

    per_hop_num = {h: round(target_num * cnt / total) for h, cnt in hop_counter.items()}

    diff = target_num - sum(per_hop_num.values())
    if diff != 0:
        for h, _ in hop_counter.most_common(abs(diff)):
            per_hop_num[h] += 1 if diff > 0 else -1

    sampled = []
    samples_by_hop = defaultdict(list)
    for s in all_samples:
        samples_by_hop[s["hop"]].append(s)

    for h, n in per_hop_num.items():
        n = min(n, len(samples_by_hop[h]))
        sampled.extend(random.sample(samples_by_hop[h], n))

    save_musique_test_samples(sampled, "data/test/dataset/testset/250/musique", "musique")
    save_sample_corpus(sampled, "data/test/dataset/testset/250/musique", "musique")

def get_hotpotqa_corpus(path, target_num=250, seed=42):
    random.seed(seed)
    file = os.path.join(path, "hotpot_dev_distractor_v1.json")
    writer = DeduplicatedWriter("data/test/dataset/corpus/hotpotqa/wash/hotpotqa_corpus.jsonl")

    all_samples = []
    with open(file, "r", encoding="utf8") as f:
        data = json.load(f)
    for item in data:
        type_ = item["type"]
        level = item["level"]

        for c in item["context"]:
            title = c[0]
            passage = " ".join(c[1])
            writer.write(title, passage)

        all_samples.append({
            "question": item["question"],
            "answer": item["answer"],
            "context": [(c[0], " ".join(c[1])) for c in item["context"]],
            "type": type_,
            "level": level
        })
    writer.close()

    combo_counter = Counter([(s["type"], s["level"]) for s in all_samples])
    total = sum(combo_counter.values())
    print("Distribution:", combo_counter)

    per_combo_num = {k: round(target_num * cnt / total) for k, cnt in combo_counter.items()}

    diff = target_num - sum(per_combo_num.values())
    if diff != 0:
        for k, _ in combo_counter.most_common(abs(diff)):
            per_combo_num[k] += 1 if diff > 0 else -1

    sampled = []
    samples_by_combo = defaultdict(list)
    for s in all_samples:
        samples_by_combo[(s["type"], s["level"])].append(s)

    for k, n in per_combo_num.items():
        group = samples_by_combo[k]
        n = min(n, len(group))  
        sampled.extend(random.sample(group, n))

    save_hotpotqa_test_samples(sampled, "data/test/dataset/testset/250/hot", "hot")
    save_sample_corpus(sampled, "data/test/dataset/testset/250/hot", "hot")
if __name__ == "__main__":
    path = "/opt/papers/benchmark/2WikiMultihopQA"
    get_2Wiki_corpus(path=path)

    path = "/opt/papers/benchmark/MuSiQue/data"
    get_musique_corpus(path=path)

    path = "/opt/papers/benchmark/hotpotqa"
    get_hotpotqa_corpus(path)
