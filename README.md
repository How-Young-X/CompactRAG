<div align="center">

# 📚 ReadingCorpus

**Advanced Reading Comprehension and Text Processing Framework**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/Transformers-4.56+-green.svg)](https://huggingface.co/transformers)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*A comprehensive framework for reading comprehension, text analysis, and natural language understanding using state-of-the-art pre-trained models.*

</div>

---

## 🌟 Features

- **Multi-Model Support**: Integration with leading language models including Meta-Llama-3, RoBERTa, and FLAN-T5
- **Advanced NLP Pipeline**: Complete text processing and analysis capabilities
- **Scalable Architecture**: Designed for both research and production environments
- **Comprehensive Evaluation**: Built-in metrics and evaluation tools for model performance assessment

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Hugging Face account with access to gated models

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ReadingCorpus
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download pre-trained models**
   ```bash
   cd models
   
   # Download Meta-Llama-3-8B-Instruct (requires Hugging Face token)
   huggingface-cli download --token hf_*** --resume-download meta-llama/Meta-Llama-3-8B-Instruct --local-dir meta-llama/Meta-Llama-3-8B-Instruct
   
   # Download RoBERTa base model
   huggingface-cli download --resume-download FacebookAI/roberta-base --local-dir FacebookAI/roberta-base
   
   # Download FLAN-T5 small model
   huggingface-cli download --resume-download google/flan-t5-small --local-dir google/flan-t5-small
   ```
4. 随机采样（按照原始开发集中问题的类型、难度比例进行采样）250个测试问题及其关联的语料，并且去重

## 📁 Project Structure

```
ReadingCorpus/
├── 📂 data/                    # Dataset storage and processing
│   └── sampled/               # Sampled datasets for testing
├── 📂 models/                 # Pre-trained model storage
│   ├── meta-llama/           # Meta-Llama-3-8B-Instruct
│   ├── FacebookAI/           # RoBERTa models
│   └── google/               # FLAN-T5 models
├── 📂 src/                    # Source code and implementations
│   ├── 📂 service/           # LLM client service
│   │   ├── llm_client.py     # Main LLM client
│   │   ├── config.py         # Configuration management
│   │   └── example_usage.py  # Usage examples
│   ├── 📂 core/              # Core processing modules
│   │   └── AskCorpus.py      # QA generation with vLLM
│   └── 📂 utils/             # Utility functions
├── 📄 start_vllm_server.sh   # vLLM server startup script
├── 📄 run_qa_generation.py   # QA generation runner
├── 📄 test_askcorpus_simple.py # AskCorpus test suite
├── 📄 requirements.txt        # Python dependencies
├── 📄 LICENSE                 # MIT License
└── 📄 README.md              # This file
```

## 🤖 Supported Models

| Model | Size | Purpose | Access |
|-------|------|---------|--------|
| **Meta-Llama-3-8B-Instruct** | 8B | Instruction following, reasoning | Gated |
| **RoBERTa-base** | 125M | Text classification, NLI | Open |
| **FLAN-T5-small** | 80M | Text generation, summarization | Open |

## 🔧 Usage

### LLM Client Service

The project includes a comprehensive LLM client service that supports both OpenAI API and vLLM local deployments.

#### Quick Start with vLLM

1. **Start the vLLM server:**
   ```bash
   ./start_vllm_server.sh
   ```

2. **Basic usage (compatible with your original code):**
   ```python
   from src.service.llm_client import create_vllm_client
   
   # Create client
   client = create_vllm_client()
   
   # Use the reason method (same as your original code)
   response = client.reason("llama8b", "你好，介绍一下你自己。", 0.7)
   print(response)
   ```

3. **Advanced usage with custom parameters:**
   ```python
   response = client.generate_response(
       model="llama8b",
       prompt="Explain machine learning in simple terms.",
       system_message="You are a helpful AI tutor.",
       temperature=0.5,
       max_tokens=200
   )
   ```

#### Testing the Service

```bash
# Test basic functionality
python test_askcorpus_simple.py

# Run comprehensive demo
python run_qa_generation.py

# View detailed examples
python src/service/example_usage.py
```

### QA Generation with AskCorpus.py

The project includes a specialized QA generation tool that processes corpus files and generates question-answer pairs using the vLLM llama8b model.

#### Quick Start

1. **Start the vLLM server:**
   ```bash
   ./start_vllm_server.sh
   ```

2. **Generate QA pairs for a specific dataset:**
   ```bash
   python src/core/AskCorpus.py --dataset musique --corpus musique_sample_corpus.jsonl --workers 4
   ```

3. **Generate QA pairs for all datasets:**
   ```bash
   python run_qa_generation.py
   ```

#### Data Organization

- **Input**: Corpus files from `data/sampled/` directory
- **Output**: QA pairs organized by dataset in `data/QA/` directory
  ```
  data/QA/
  ├── musique/
  │   ├── llama8b-musique-qa.jsonl
  │   └── llama8b-musique-qa_failed.jsonl
  ├── 2wiki/
  │   ├── llama8b-2wiki-qa.jsonl
  │   └── llama8b-2wiki-qa_failed.jsonl
  └── hotpotqa/
      ├── llama8b-hotpotqa-qa.jsonl
      └── llama8b-hotpotqa-qa_failed.jsonl
  ```

### Traditional Model Usage

#### Basic Text Processing
```python
from transformers import AutoTokenizer, AutoModel

# Load a model
tokenizer = AutoTokenizer.from_pretrained("models/facebook/roberta-base")
model = AutoModel.from_pretrained("models/facebook/roberta-base")

# Process text
text = "Your text here"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
```

#### Reading Comprehension
```python
# Example usage for reading comprehension tasks
# (Implementation details in src/)
```

## 📊 Performance

- **Model Accuracy**: State-of-the-art performance on benchmark datasets
- **Processing Speed**: Optimized for both CPU and GPU inference
- **Memory Efficiency**: Support for model quantization and optimization

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details on:

- Code style and standards
- Testing requirements
- Pull request process
- Issue reporting

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for providing the Transformers library
- [Meta AI](https://ai.meta.com/) for the Llama models
- [Facebook AI](https://ai.facebook.com/) for RoBERTa
- [Google Research](https://research.google/) for FLAN-T5

## 📞 Support

For questions, issues, or contributions, please:

1. Check existing [Issues](../../issues)
2. Create a new issue if needed
3. Contact the maintainers

---

<div align="center">

**Made with ❤️ for the NLP community**

[⭐ Star this repo](../../stargazers) • [🐛 Report Bug](../../issues) • [💡 Request Feature](../../issues)

</div>