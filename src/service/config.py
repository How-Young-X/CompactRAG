"""
Configuration management for LLM services.

This module provides configuration classes and utilities for managing
different LLM service endpoints and settings.

Author: ReadingCorpus Team
Version: 1.0.0
"""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ServiceConfig:
    """Configuration for individual LLM service endpoints."""
    name: str
    provider: str  # "openai", "vllm", "custom"
    base_url: str
    api_key: str = "EMPTY"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    enabled: bool = True
    description: str = ""


@dataclass
class LLMServiceConfig:
    """Main configuration class for LLM services."""
    services: Dict[str, ServiceConfig]
    default_service: str = "vllm"
    log_level: str = "INFO"
    config_file: Optional[str] = None
    
    def __post_init__(self):
        """Initialize configuration after object creation."""
        if self.config_file:
            self.load_from_file(self.config_file)
    
    def get_service(self, name: str) -> Optional[ServiceConfig]:
        """Get a service configuration by name."""
        return self.services.get(name)
    
    def add_service(self, service: ServiceConfig) -> None:
        """Add a new service configuration."""
        self.services[service.name] = service
    
    def remove_service(self, name: str) -> bool:
        """Remove a service configuration."""
        if name in self.services:
            del self.services[name]
            return True
        return False
    
    def save_to_file(self, filepath: str) -> None:
        """Save configuration to a JSON file."""
        config_data = {
            "services": {name: asdict(service) for name, service in self.services.items()},
            "default_service": self.default_service,
            "log_level": self.log_level
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    def load_from_file(self, filepath: str) -> None:
        """Load configuration from a JSON file."""
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # Load services
        self.services = {}
        for name, service_data in config_data.get("services", {}).items():
            self.services[name] = ServiceConfig(**service_data)
        
        # Load other settings
        self.default_service = config_data.get("default_service", "vllm")
        self.log_level = config_data.get("log_level", "INFO")


def create_default_config() -> LLMServiceConfig:
    """Create a default configuration with common LLM services."""
    config = LLMServiceConfig(services={})
    
    # Add vLLM service
    vllm_service = ServiceConfig(
        name="vllm",
        provider="vllm",
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
        timeout=30,
        description="Local vLLM server"
    )
    config.add_service(vllm_service)
    
    # Add OpenAI service
    openai_service = ServiceConfig(
        name="openai",
        provider="openai",
        base_url="https://api.openai.com/v1",
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
        timeout=30,
        description="OpenAI API service"
    )
    config.add_service(openai_service)
    
    # Add custom service example
    custom_service = ServiceConfig(
        name="custom",
        provider="custom",
        base_url="http://localhost:8001/v1",
        api_key="EMPTY",
        timeout=30,
        description="Custom LLM service endpoint"
    )
    config.add_service(custom_service)
    
    return config


def load_config_from_env() -> LLMServiceConfig:
    """Load configuration from environment variables."""
    config = create_default_config()
    
    # Override with environment variables if available
    if os.getenv("VLLM_BASE_URL"):
        vllm_service = config.get_service("vllm")
        if vllm_service:
            vllm_service.base_url = os.getenv("VLLM_BASE_URL")
    
    if os.getenv("OPENAI_API_KEY"):
        openai_service = config.get_service("openai")
        if openai_service:
            openai_service.api_key = os.getenv("OPENAI_API_KEY")
    
    if os.getenv("DEFAULT_LLM_SERVICE"):
        config.default_service = os.getenv("DEFAULT_LLM_SERVICE")
    
    return config


# Global configuration instance
_config: Optional[LLMServiceConfig] = None


def get_config() -> LLMServiceConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config_from_env()
    return _config


def set_config(config: LLMServiceConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


# Example usage
if __name__ == "__main__":
    # Create and display default configuration
    config = create_default_config()
    
    print("=== Default Configuration ===")
    for name, service in config.services.items():
        print(f"Service: {name}")
        print(f"  Provider: {service.provider}")
        print(f"  Base URL: {service.base_url}")
        print(f"  Enabled: {service.enabled}")
        print(f"  Description: {service.description}")
        print()
    
    # Save configuration to file
    config.save_to_file("llm_config.json")
    print("Configuration saved to llm_config.json")
    
    # Load configuration from file
    loaded_config = LLMServiceConfig(config_file="llm_config.json")
    print(f"Loaded configuration with {len(loaded_config.services)} services")
