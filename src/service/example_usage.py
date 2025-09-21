"""
Example usage of the LLM client service.

This module demonstrates how to use the LLM client for various tasks
including basic reasoning, text generation, and service management.

Author: ReadingCorpus Team
Version: 1.0.0
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from service.llm_client import create_vllm_client, create_openai_client, LLMClient
from service.config import get_config, create_default_config


def example_vllm_usage():
    """Example of using vLLM client for text generation."""
    print("=== vLLM Usage Example ===")
    
    try:
        # Create vLLM client
        client = create_vllm_client(
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
            timeout=30
        )
        
        # Check if service is healthy
        if not client.health_check():
            print("❌ vLLM service is not accessible. Please start the vLLM server.")
            return
        
        print("✅ vLLM service is healthy")
        
        # Basic reasoning example
        response = client.reason(
            model="llama-2-7b-chat-hf",
            prompt="你好，介绍一下你自己。",
            temperature=0.7
        )
        print(f"🤖 Response: {response}")
        
        # Advanced usage with custom parameters
        advanced_response = client.generate_response(
            model="llama-2-7b-chat-hf",
            prompt="Explain the concept of machine learning in simple terms.",
            system_message="You are a helpful AI tutor who explains complex concepts simply.",
            temperature=0.5,
            max_tokens=150,
            top_p=0.9
        )
        print(f"🎓 Advanced Response: {advanced_response}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def example_openai_usage():
    """Example of using OpenAI client (requires valid API key)."""
    print("\n=== OpenAI Usage Example ===")
    
    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        print("⚠️  OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        return
    
    try:
        # Create OpenAI client
        client = create_openai_client(
            api_key=api_key,
            base_url="https://api.openai.com/v1",
            timeout=30
        )
        
        # Check if service is healthy
        if not client.health_check():
            print("❌ OpenAI service is not accessible.")
            return
        
        print("✅ OpenAI service is healthy")
        
        # Generate response
        response = client.reason(
            model="gpt-3.5-turbo",
            prompt="Hello, please introduce yourself.",
            temperature=0.7
        )
        print(f"🤖 Response: {response}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def example_config_usage():
    """Example of using configuration management."""
    print("\n=== Configuration Usage Example ===")
    
    # Get default configuration
    config = create_default_config()
    
    print("📋 Available services:")
    for name, service in config.services.items():
        status = "✅" if service.enabled else "❌"
        print(f"  {status} {name}: {service.provider} - {service.base_url}")
    
    # Create client using configuration
    vllm_service = config.get_service("vllm")
    if vllm_service and vllm_service.enabled:
        try:
            client = LLMClient(vllm_service)
            if client.health_check():
                print("✅ Successfully created client from configuration")
            else:
                print("❌ Service is not accessible")
        except Exception as e:
            print(f"❌ Error creating client: {str(e)}")


def example_batch_processing():
    """Example of batch processing multiple prompts."""
    print("\n=== Batch Processing Example ===")
    
    try:
        client = create_vllm_client()
        
        if not client.health_check():
            print("❌ vLLM service is not accessible.")
            return
        
        # List of prompts to process
        prompts = [
            "What is artificial intelligence?",
            "Explain quantum computing briefly.",
            "How does machine learning work?",
            "What are the benefits of renewable energy?"
        ]
        
        print(f"🔄 Processing {len(prompts)} prompts...")
        
        for i, prompt in enumerate(prompts, 1):
            try:
                response = client.reason(
                    model="llama-2-7b-chat-hf",
                    prompt=prompt,
                    temperature=0.7
                )
                print(f"📝 Prompt {i}: {prompt[:50]}...")
                print(f"🤖 Response: {response[:100]}...")
                print("-" * 50)
            except Exception as e:
                print(f"❌ Error processing prompt {i}: {str(e)}")
        
    except Exception as e:
        print(f"❌ Error in batch processing: {str(e)}")


def main():
    """Main function to run all examples."""
    print("🚀 LLM Client Service Examples")
    print("=" * 50)
    
    # Run examples
    example_vllm_usage()
    example_openai_usage()
    example_config_usage()
    example_batch_processing()
    
    print("\n✨ Examples completed!")


if __name__ == "__main__":
    main()
