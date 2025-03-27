"""
Configuration utilities for loading, parsing and handling experiment configurations.
"""

import os
import yaml
from typing import Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class Config:
    """
    Configuration class for handling experiment settings.
    
    Manages loading from YAML files and providing access to configuration parameters.
    """
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration either from a YAML file or a dictionary.
        
        Args:
            config_path: Path to the YAML configuration file
            config_dict: Dictionary containing configuration parameters
        """
        if config_path is not None:
            self.config = self._load_yaml(config_path)
        elif config_dict is not None:
            self.config = config_dict
        else:
            self.config = {}
        
    def _load_yaml(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Dictionary containing the configuration
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key with an optional default.
        
        Args:
            key: The configuration key (can be nested with dot notation, e.g., 'model.learning_rate')
            default: Default value to return if key is not found
            
        Returns:
            The configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value by key.
        
        Args:
            key: The configuration key (can be nested with dot notation)
            value: The value to set
        """
        keys = key.split('.')
        config = self.config
        
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, config_path: str) -> None:
        """
        Save the current configuration to a YAML file.
        
        Args:
            config_path: Path where to save the configuration
        """
        # Make sure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"Saved configuration to {config_path}")
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration with values from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters to update
        """
        self._deep_update(self.config, config_dict)
    
    def _deep_update(self, original: Dict[str, Any], update: Dict[str, Any]) -> None:
        """
        Recursively update a nested dictionary.
        
        Args:
            original: Original dictionary to update
            update: Dictionary with new values
        """
        for key, value in update.items():
            if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                self._deep_update(original[key], value)
            else:
                original[key] = value
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to config values."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style setting of config values."""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the configuration."""
        return self.get(key) is not None
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return yaml.dump(self.config, default_flow_style=False)


def load_config(config_path: str) -> Config:
    """
    Helper function to load configuration from a file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Config object
    """
    return Config(config_path=config_path)


def setup_directories(config: Config) -> None:
    """
    Create necessary directories based on configuration.
    
    Args:
        config: Configuration object
    """
    # Make output directory
    output_dir = config.get('experiment.output_dir', 'outputs/default')
    os.makedirs(output_dir, exist_ok=True)
    
    # Make data directories
    data_dir = config.get('data.data_dir', 'data')
    vocab_dir = config.get('data.vocab_dir', 'vocab')
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(vocab_dir, exist_ok=True)
    
    # Make log directory
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    logger.info(f"Created directories: {output_dir}, {data_dir}, {vocab_dir}, {log_dir}")
