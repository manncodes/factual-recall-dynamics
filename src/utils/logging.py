"""
Logging utilities for the project.
"""

import os
import logging
import sys
from typing import Optional, Union, Dict, Any
from pathlib import Path
import datetime

def setup_logging(
    log_level: Union[str, int] = "INFO",
    log_file: Optional[str] = None,
    log_to_console: bool = True,
    name: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Path to log file (default: None, no file logging)
        log_to_console: Whether to log to console (default: True)
        name: Logger name (default: None, root logger)
        
    Returns:
        Configured logger
    """
    # Convert string log level to numeric value if needed
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())
    
    # Get logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.handlers = []  # Remove existing handlers
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add console handler if requested
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if log_file is specified
    if log_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class WandbLogger:
    """
    Wrapper for Weights & Biases logging functionality.
    """
    
    def __init__(self, 
                project: str, 
                name: Optional[str] = None, 
                config: Optional[Dict[str, Any]] = None,
                group: Optional[str] = None,
                resume: bool = False):
        """
        Initialize a WandbLogger.
        
        Args:
            project: W&B project name
            name: Run name (default: auto-generated)
            config: Configuration dict for the run
            group: Group name for the run
            resume: Whether to resume a previous run
        """
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            raise ImportError(
                "To use WandbLogger, install wandb with: pip install wandb"
            )
        
        self.project = project
        self.name = name
        self.config = config
        self.group = group
        self.resume = resume
        self.run = None
        
    def init(self) -> None:
        """Initialize the W&B run."""
        self.run = self.wandb.init(
            project=self.project,
            name=self.name,
            config=self.config,
            group=self.group,
            resume=self.resume
        )
        
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to W&B.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step number (optional)
        """
        if self.run is None:
            self.init()
            
        self.wandb.log(metrics, step=step)
        
    def log_artifact(self, 
                    artifact_path: str, 
                    artifact_name: Optional[str] = None, 
                    artifact_type: str = "model",
                    description: Optional[str] = None) -> None:
        """
        Log an artifact (file) to W&B.
        
        Args:
            artifact_path: Path to the artifact
            artifact_name: Name for the artifact (default: filename)
            artifact_type: Type of artifact (default: "model")
            description: Description of the artifact
        """
        if self.run is None:
            self.init()
            
        if artifact_name is None:
            artifact_name = os.path.basename(artifact_path)
            
        artifact = self.wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            description=description
        )
        
        artifact.add_file(artifact_path)
        self.run.log_artifact(artifact)
        
    def finish(self) -> None:
        """Finish the W&B run."""
        if self.run is not None:
            self.wandb.finish()
            self.run = None


def get_run_name(prefix: str = "run") -> str:
    """
    Generate a unique run name based on timestamp.
    
    Args:
        prefix: Prefix for the run name
        
    Returns:
        Run name in format "{prefix}_{timestamp}"
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"
