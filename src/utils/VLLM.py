"""
vLLM client for calling llama8b model
"""

import requests
import json
import time
from typing import Optional


def reason(model: str, prompt_: str, temperature: float = 0, max_tokens: Optional[int] = None) -> str:
    """
    Call vLLM API to generate response
    
    Args:
        model: Model name (e.g., "llama8b")
        prompt_: Input prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        
    Returns:
        Generated response text
    """
    url = "http://localhost:8000/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt_}
        ],
        "temperature": temperature,
        "stream": False
    }
    
    if max_tokens is not None:
        data["max_tokens"] = max_tokens
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=100)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
        
    except requests.exceptions.RequestException as e:
        print(f"vLLM API request failed: {e}")
        raise
    except (KeyError, IndexError) as e:
        print(f"Failed to parse vLLM response: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error calling vLLM: {e}")
        raise


def health_check() -> bool:
    """
    Check if vLLM service is running
    
    Returns:
        True if service is healthy, False otherwise
    """
    try:
        url = "http://localhost:8000/v1/models"
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except:
        return False


if __name__ == "__main__":
    # Test the vLLM client
    if health_check():
        print("vLLM service is running")
        try:
            response = reason(
                model="llama8b",
                prompt_="Hello, how are you?",
                temperature=0.7
            )
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("vLLM service is not running. Please start the vLLM server.")
