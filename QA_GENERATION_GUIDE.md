# QA Generation Guide

This guide explains how to use the rewritten AskCorpus.py to generate question-answer pairs using vLLM llama8b model.

## 🚀 Quick Start

### 1. Start vLLM Server

First, start the vLLM server:

```bash
./start_vllm_server.sh
```

This will start the vLLM server with the llama8b model on `http://localhost:8000/v1`.

### 2. Generate QA Pairs

#### For a Single Dataset

```bash
# Generate QA pairs for Musique dataset
python src/core/AskCorpus.py --dataset musique --corpus musique_sample_corpus.jsonl --workers 4

# Generate QA pairs for 2Wiki dataset
python src/core/AskCorpus.py --dataset 2wiki --corpus 2wiki_sample_corpus.jsonl --workers 4

# Generate QA pairs for HotpotQA dataset
python src/core/AskCorpus.py --dataset hotpotqa --corpus hotpotqa_sample_corpus.jsonl --workers 4
```

#### For All Datasets

```bash
python run_qa_generation.py
```

## 📁 Data Organization

### Input Files
- **Location**: `data/sampled/`
- **Files**:
  - `musique_sample_corpus.jsonl`
  - `2wiki_sample_corpus.jsonl`
  - `hotpotqa_sample_corpus.jsonl`

### Output Files
- **Location**: `data/QA/{dataset}/`
- **Structure**:
  ```
  data/QA/
  ├── musique/
  │   ├── llama8b-musique-qa.jsonl          # Generated QA pairs
  │   └── llama8b-musique-qa_failed.jsonl   # Failed processing records
  ├── 2wiki/
  │   ├── llama8b-2wiki-qa.jsonl
  │   └── llama8b-2wiki-qa_failed.jsonl
  └── hotpotqa/
      ├── llama8b-hotpotqa-qa.jsonl
      └── llama8b-hotpotqa-qa_failed.jsonl
  ```

## ⚙️ Configuration Options

### Command Line Arguments

| Argument | Description | Default | Choices |
|----------|-------------|---------|---------|
| `--corpus` | Input corpus file | `musique_sample_corpus.jsonl` | `musique_sample_corpus.jsonl`, `2wiki_sample_corpus.jsonl`, `hotpotqa_sample_corpus.jsonl` |
| `--dataset` | Dataset name for output organization | `musique` | `musique`, `2wiki`, `hotpotqa` |
| `--workers` | Number of parallel workers | `4` | Any positive integer (recommended: 2-4) |
| `--start` | Start index for resuming | `0` | Any non-negative integer |
| `--vllm-url` | vLLM server URL | `http://localhost:8000/v1` | Any valid URL |

### vLLM Configuration

The script uses the following vLLM settings:
- **Model**: `llama8b` (served model name)
- **Temperature**: `0.7`
- **Timeout**: `60` seconds
- **Max Retries**: `3` per request
- **Retry Delay**: `2` seconds

## 🧪 Testing

### Test vLLM Connection and QA Generation

```bash
python test_askcorpus_simple.py
```

This will test:
- vLLM server connectivity
- Basic QA generation functionality
- File structure validation

### Test Individual Components

```bash
# Test vLLM server only
curl http://localhost:8000/v1/models

# Test with a simple prompt
python -c "
from src.service.llm_client import create_vllm_client
client = create_vllm_client()
print(client.reason('llama8b', 'Hello, how are you?', 0.7))
"
```

## 🔧 Troubleshooting

### Common Issues

1. **vLLM Server Not Running**
   ```
   ❌ vLLM server is not running
   ```
   **Solution**: Start the server with `./start_vllm_server.sh`

2. **Model Not Found**
   ```
   ❌ Model not found at: /path/to/model
   ```
   **Solution**: Ensure the model is downloaded in the correct location

3. **Connection Timeout**
   ```
   ❌ vLLM call failed: timeout
   ```
   **Solution**: Check if the server is overloaded, reduce `--workers` parameter

4. **JSON Parsing Errors**
   ```
   ❌ JSON extraction failed
   ```
   **Solution**: The model output might not be in the expected format. Check the raw response in logs.

### Performance Tips

1. **Worker Count**: Use 2-4 workers for optimal performance with vLLM
2. **Batch Size**: Process datasets in smaller batches if memory is limited
3. **Resume Processing**: Use `--start` parameter to resume from a specific index
4. **Monitor Logs**: Check logs for any processing issues

## 📊 Output Format

### Successful QA Pairs
Each line in the output file contains:
```json
{
  "title": "Document title",
  "passage": "Document content",
  "qa": [
    {
      "question": "Generated question",
      "answer": "Generated answer"
    }
  ]
}
```

### Failed Records
Failed records are saved with retry information:
```json
{
  "title": "Document title", 
  "passage": "Document content",
  "_retry_count": 5
}
```

## 🎯 Example Usage

### Complete Workflow

```bash
# 1. Start vLLM server
./start_vllm_server.sh

# 2. Test the setup
python test_askcorpus_simple.py

# 3. Generate QA pairs for all datasets
python run_qa_generation.py

# 4. Check results
ls -la data/QA/*/
```

### Resume Processing

```bash
# Resume from index 1000
python src/core/AskCorpus.py --dataset musique --corpus musique_sample_corpus.jsonl --start 1000 --workers 4
```

This guide should help you successfully generate QA pairs using the vLLM llama8b model!
