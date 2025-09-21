#!/usr/bin/env python3
"""
Run QA generation for all datasets using vLLM llama8b model.

This script provides convenient commands to generate QA pairs for all
available datasets in the data/sampled directory.
"""

import subprocess
import sys
import time
from pathlib import Path

def run_qa_generation(dataset, corpus, workers=4, start=0):
    """Run QA generation for a specific dataset."""
    print(f"\n🚀 Starting QA generation for {dataset} dataset")
    print(f"📁 Input: data/sampled/{corpus}")
    print(f"📁 Output: data/QA/{dataset}/")
    print(f"👥 Workers: {workers}")
    print("-" * 50)
    
    cmd = [
        "python", "src/core/AskCorpus.py",
        "--dataset", dataset,
        "--corpus", corpus,
        "--workers", str(workers),
        "--start", str(start)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ Successfully completed {dataset} dataset")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to process {dataset} dataset: {e}")
        return False

def check_vllm_server():
    """Check if vLLM server is running."""
    try:
        import requests
        response = requests.get("http://localhost:8000/v1/models", timeout=5)
        if response.status_code == 200:
            print("✅ vLLM server is running")
            return True
    except:
        pass
    
    print("❌ vLLM server is not running")
    print("💡 Please start the server first:")
    print("   ./start_vllm_server.sh")
    return False

def main():
    """Main function to run QA generation for all datasets."""
    print("🎯 QA Generation Script for ReadingCorpus")
    print("=" * 50)
    
    # Check if vLLM server is running
    if not check_vllm_server():
        sys.exit(1)
    
    # Dataset configurations
    datasets = [
        {"dataset": "musique", "corpus": "musique_sample_corpus.jsonl"},
        {"dataset": "2wiki", "corpus": "2wiki_sample_corpus.jsonl"},
        {"dataset": "hotpotqa", "corpus": "hotpotqa_sample_corpus.jsonl"}
    ]
    
    # Check if corpus files exist
    missing_files = []
    for config in datasets:
        corpus_path = Path(f"data/sampled/{config['corpus']}")
        if not corpus_path.exists():
            missing_files.append(config['corpus'])
    
    if missing_files:
        print(f"❌ Missing corpus files: {missing_files}")
        print("Please make sure all corpus files exist in data/sampled/")
        sys.exit(1)
    
    # Create QA directory structure
    qa_dir = Path("data/QA")
    qa_dir.mkdir(exist_ok=True)
    
    # Process each dataset
    successful = 0
    failed = 0
    
    for config in datasets:
        success = run_qa_generation(
            dataset=config["dataset"],
            corpus=config["corpus"],
            workers=4
        )
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Add a small delay between datasets
        if config != datasets[-1]:  # Not the last dataset
            print("\n⏳ Waiting 5 seconds before next dataset...")
            time.sleep(5)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Summary:")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Output directory: data/QA/")
    
    if successful > 0:
        print("\n🎉 QA generation completed!")
        print("📂 Check the following directories for results:")
        for config in datasets:
            output_dir = Path(f"data/QA/{config['dataset']}")
            if output_dir.exists():
                print(f"   - {output_dir}/")

if __name__ == "__main__":
    main()
