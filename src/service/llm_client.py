"""
Large Language Model Client Service

This module provides a unified interface for connecting to various LLM services
including OpenAI API and vLLM local deployments. It supports both OpenAI-compatible
APIs and direct vLLM inference endpoints.

Author: ReadingCorpus Team
Version: 1.0.0
"""

import json
import logging
import time
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("OpenAI library is required. Install with: pip install openai")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Enumeration of supported LLM providers."""
    OPENAI = "openai"
    VLLM = "vllm"
    CUSTOM = "custom"


@dataclass
class LLMConfig:
    """Configuration class for LLM client settings."""
    provider: LLMProvider
    api_key: str = "EMPTY"
    base_url: str = "http://localhost:8000/v1"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0


class LLMClient:
    """
    Unified client for interacting with various Large Language Model services.
    
    This class provides a consistent interface for both OpenAI API and vLLM
    local deployments, handling authentication, error management, and response
    processing.
    """
    
    def __init__(self, config: LLMConfig):
        """
        Initialize the LLM client with the provided configuration.
        
        Args:
            config (LLMConfig): Configuration object containing provider settings
        """
        self.config = config
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the OpenAI client based on the configuration."""
        try:
            self.client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )
            logger.info(f"Initialized LLM client for provider: {self.config.provider.value}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {str(e)}")
            raise
    
    def _make_request_with_retry(self, request_func, *args, **kwargs) -> Any:
        """
        Execute a request with retry logic for handling temporary failures.
        
        Args:
            request_func: The function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The result of the request function
            
        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                return request_func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                logger.warning(f"Request attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"All {self.config.max_retries} attempts failed")
        
        raise last_exception
    
    def generate_response(
        self,
        model: str,
        prompt: str,
        system_message: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate a response from the LLM using the chat completions API.
        
        Args:
            model (str): The model name to use for generation
            prompt (str): The user's input prompt
            system_message (str): System message to set the assistant's behavior
            temperature (float): Sampling temperature (0.0 to 2.0)
            max_tokens (Optional[int]): Maximum number of tokens to generate
            top_p (Optional[float]): Nucleus sampling parameter
            frequency_penalty (Optional[float]): Frequency penalty (-2.0 to 2.0)
            presence_penalty (Optional[float]): Presence penalty (-2.0 to 2.0)
            **kwargs: Additional parameters for the API call
            
        Returns:
            str: The generated response text
            
        Raises:
            Exception: If the request fails after all retry attempts
        """
        try:
            # Prepare the request parameters
            request_params = {
                "model": model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ]
            }
            
            # Add optional parameters if provided
            if max_tokens is not None:
                request_params["max_tokens"] = max_tokens
            if top_p is not None:
                request_params["top_p"] = top_p
            if frequency_penalty is not None:
                request_params["frequency_penalty"] = frequency_penalty
            if presence_penalty is not None:
                request_params["presence_penalty"] = presence_penalty
            
            # Add any additional parameters
            request_params.update(kwargs)
            
            logger.info(f"Generating response with model: {model}, temperature: {temperature}")
            
            # Make the request with retry logic
            completion = self._make_request_with_retry(
                self.client.chat.completions.create,
                **request_params
            )
            
            # Extract and return the response
            response = completion.choices[0].message.content
            logger.info(f"Successfully generated response of length: {len(response)}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            raise
    
    def reason(self, model: str, prompt: str, temperature: float = 0.7) -> str:
        """
        Simplified interface for reasoning tasks.
        
        This is a convenience method that provides a simple interface for
        basic reasoning tasks with default settings.
        
        Args:
            model (str): The model name to use
            prompt (str): The user's prompt
            temperature (float): Sampling temperature
            
        Returns:
            str: The model's response
        """
        return self.generate_response(
            model=model,
            prompt=prompt,
            temperature=temperature
        )
    
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model (str): The model name
            
        Returns:
            Dict[str, Any]: Model information
        """
        try:
            models = self.client.models.list()
            for model_info in models.data:
                if model_info.id == model:
                    return {
                        "id": model_info.id,
                        "object": model_info.object,
                        "created": model_info.created,
                        "owned_by": model_info.owned_by
                    }
            return {"error": f"Model {model} not found"}
        except Exception as e:
            logger.error(f"Failed to get model info: {str(e)}")
            return {"error": str(e)}
    
    def health_check(self) -> bool:
        """
        Check if the LLM service is healthy and accessible.
        
        Returns:
            bool: True if the service is healthy, False otherwise
        """
        try:
            models = self.client.models.list()
            logger.info("Health check passed - service is accessible")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False


def create_vllm_client(
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    timeout: int = 30
) -> LLMClient:
    """
    Create a vLLM client with default configuration.
    
    Args:
        base_url (str): The base URL for the vLLM server
        api_key (str): API key (usually "EMPTY" for vLLM)
        timeout (int): Request timeout in seconds
        
    Returns:
        LLMClient: Configured client for vLLM
    """
    config = LLMConfig(
        provider=LLMProvider.VLLM,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout
    )
    return LLMClient(config)


def create_openai_client(
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    timeout: int = 30
) -> LLMClient:
    """
    Create an OpenAI client with the provided configuration.
    
    Args:
        api_key (str): OpenAI API key
        base_url (str): Base URL for OpenAI API
        timeout (int): Request timeout in seconds
        
    Returns:
        LLMClient: Configured client for OpenAI
    """
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout
    )
    return LLMClient(config)


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Using vLLM client
    print("=== vLLM Client Example ===")
    try:
        vllm_client = create_vllm_client()
        
        # Test health check
        if vllm_client.health_check():
            print("vLLM service is healthy")
            
            # Generate a response
            response = vllm_client.reason(
                model="llama-2-7b-chat-hf",
                prompt="你好，介绍一下你自己。",
                temperature=0.7
            )
            print(f"vLLM Response: {response}")
        else:
            print("vLLM service is not accessible")
            
    except Exception as e:
        print(f"vLLM client error: {str(e)}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Using OpenAI client (requires valid API key)
    print("=== OpenAI Client Example ===")
    try:
        # Note: Replace with your actual OpenAI API key
        openai_client = create_openai_client(api_key="your-openai-api-key-here")
        
        if openai_client.health_check():
            print("OpenAI service is healthy")
            
            response = openai_client.reason(
                model="gpt-3.5-turbo",
                prompt="Hello, please introduce yourself.",
                temperature=0.7
            )
            print(f"OpenAI Response: {response}")
        else:
            print("OpenAI service is not accessible")
            
    except Exception as e:
        print(f"OpenAI client error: {str(e)}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Advanced usage with custom parameters
    print("=== Advanced Usage Example ===")
    try:
        client = create_vllm_client()
        
        response = client.generate_response(
            model="llama-2-7b-chat-hf",
            prompt="Explain quantum computing in simple terms.",
            system_message="You are a helpful science teacher.",
            temperature=0.5,
            max_tokens=200,
            top_p=0.9
        )
        print(f"Advanced Response: {response}")
        
    except Exception as e:
        print(f"Advanced usage error: {str(e)}")
