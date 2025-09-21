#!/usr/bin/env python3
"""
Test script for LLM client service.

This script provides a simple way to test the LLM client functionality
without running the full example suite.

Usage:
    python test_llm_client.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from service.llm_client import create_vllm_client


def test_basic_functionality():
    """Test basic LLM client functionality."""
    print("🧪 Testing LLM Client Basic Functionality")
    print("=" * 50)
    
    try:
        # Create client
        client = create_vllm_client()
        print("✅ Client created successfully")
        
        # Test health check
        if client.health_check():
            print("✅ Health check passed")
        else:
            print("❌ Health check failed - vLLM server may not be running")
            print("💡 To start vLLM server, run:")
            print("   python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct")
            return False
        
        # Test basic reasoning
        print("🔄 Testing basic reasoning...")
        response = client.reason(
            model="llama-2-7b-chat-hf",
            prompt="Hello, how are you?",
            temperature=0.7
        )
        print(f"✅ Response received: {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


def main():
    """Main test function."""
    print("🚀 LLM Client Test Suite")
    print("=" * 50)
    
    success = test_basic_functionality()
    
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Some tests failed. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
