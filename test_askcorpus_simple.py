#!/usr/bin/env python3
"""
Simple test script for AskCorpus.py functionality.

This script tests the basic functionality of the rewritten AskCorpus.py
without running the full processing pipeline.
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from core.AskCorpus import get_vllm_client, call_vllm_with_retry, extract_json
from prompt.GenerateQA import GENERATEQA

async def test_vllm_connection():
    """Test vLLM server connection."""
    print("🧪 Testing vLLM Connection")
    print("-" * 30)
    
    try:
        client = get_vllm_client()
        
        # Test basic connection
        models = client.models.list()
        print("✅ vLLM server is accessible")
        print(f"📋 Available models: {[model.id for model in models.data]}")
        return True
        
    except Exception as e:
        print(f"❌ vLLM connection failed: {e}")
        return False

async def test_qa_generation():
    """Test QA generation with a sample text."""
    print("\n🧪 Testing QA Generation")
    print("-" * 30)
    
    # Sample text for testing
    sample_text = """
    Alexander Fleming was a Scottish biologist and pharmacologist. 
    He discovered penicillin in 1928 at St. Mary's Hospital in London. 
    This discovery revolutionized medicine and led to the development of antibiotics.
    """
    
    # Format the prompt
    prompt = GENERATEQA.format(chunk=sample_text)
    print(f"📝 Sample prompt length: {len(prompt)} characters")
    
    try:
        # Call vLLM
        response = await call_vllm_with_retry(prompt)
        
        if response:
            print("✅ vLLM response received")
            print(f"📄 Response length: {len(response)} characters")
            
            # Try to extract JSON
            json_result = extract_json(response)
            if json_result:
                print("✅ JSON extraction successful")
                print(f"📊 Generated {len(json_result)} QA pairs")
                
                # Show first QA pair as example
                if json_result:
                    first_qa = json_result[0]
                    print(f"📝 Example Q: {first_qa.get('question', 'N/A')}")
                    print(f"💡 Example A: {first_qa.get('answer', 'N/A')}")
                
                return True
            else:
                print("❌ JSON extraction failed")
                print(f"📄 Raw response: {response[:200]}...")
                return False
        else:
            print("❌ No response from vLLM")
            return False
            
    except Exception as e:
        print(f"❌ QA generation test failed: {e}")
        return False

def test_file_structure():
    """Test if required files and directories exist."""
    print("\n🧪 Testing File Structure")
    print("-" * 30)
    
    required_files = [
        "data/sampled/musique_sample_corpus.jsonl",
        "data/sampled/2wiki_sample_corpus.jsonl", 
        "data/sampled/hotpotqa_sample_corpus.jsonl"
    ]
    
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    # Create QA directory
    qa_dir = Path("data/QA")
    qa_dir.mkdir(exist_ok=True)
    print(f"✅ Created/verified: {qa_dir}")
    
    return len(missing_files) == 0

async def main():
    """Main test function."""
    print("🚀 AskCorpus.py Test Suite")
    print("=" * 50)
    
    # Test file structure
    files_ok = test_file_structure()
    
    # Test vLLM connection
    connection_ok = await test_vllm_connection()
    
    # Test QA generation
    qa_ok = False
    if connection_ok:
        qa_ok = await test_qa_generation()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"📁 File structure: {'✅ PASS' if files_ok else '❌ FAIL'}")
    print(f"🔗 vLLM connection: {'✅ PASS' if connection_ok else '❌ FAIL'}")
    print(f"🤖 QA generation: {'✅ PASS' if qa_ok else '❌ FAIL'}")
    
    if files_ok and connection_ok and qa_ok:
        print("\n🎉 All tests passed! AskCorpus.py is ready to use.")
        print("\n💡 Usage examples:")
        print("   python src/core/AskCorpus.py --dataset musique --corpus musique_sample_corpus.jsonl --workers 4")
        print("   python run_qa_generation.py")
    else:
        print("\n❌ Some tests failed. Please check the issues above.")
        if not connection_ok:
            print("💡 Make sure vLLM server is running: ./start_vllm_server.sh")

if __name__ == "__main__":
    asyncio.run(main())
