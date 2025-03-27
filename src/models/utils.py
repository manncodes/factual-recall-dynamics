"""
Utility functions for model handling.
"""

import os
import logging
import torch
from typing import Optional, Dict, Any, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    TrainingArguments
)

logger = logging.getLogger(__name__)


def create_or_load_model(
    config: Dict[str, Any]
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Create a new model or load a pretrained model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (model, tokenizer)
    """
    model_name = config.get('model.model_name')
    is_finetuning = config.get('model.is_finetuning', False)
    
    # Load a pretrained model if finetuning
    if is_finetuning and model_name:
        logger.info(f"Loading pretrained model {model_name} for fine-tuning")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Resize token embeddings if we add new tokens later
        model.resize_token_embeddings(len(tokenizer))
        
    # Otherwise, create a new model for pretraining
    else:
        logger.info("Initializing new model for pretraining")
        
        # Calculate model size
        n_layers = config.get('model.n_layers', 12)
        n_heads = config.get('model.n_heads', 12)
        hidden_size = n_heads * 4  # Common ratio for transformer models
        
        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.get('model.tokenizer_name', 'gpt2')
        )
        
        # Add special tokens
        special_tokens = {
            "pad_token": "[PAD]",
            "sep_token": "[SEP]",
        }
        tokenizer.add_special_tokens(special_tokens)
        
        # Initialize model (GPT-NeoX architecture)
        model_config = GPTNeoXConfig(
            vocab_size=len(tokenizer),
            hidden_size=hidden_size,
            num_attention_heads=n_heads,
            num_hidden_layers=n_layers,
            rotary_pct=0.25,  # 25% of hidden dims
            rotary_emb_base=10000,
            max_position_embeddings=config.get('model.seq_length', 256)
        )
        
        model = GPTNeoXForCausalLM(model_config)
        model.resize_token_embeddings(len(tokenizer))
    
    # Log model size
    model_size = sum(p.numel() for p in model.parameters())
    logger.info(f"Model size: {model_size:,} parameters ({model_size/1e6:.1f}M)")
    
    return model, tokenizer


def create_training_arguments(
    config: Dict[str, Any],
    output_dir: Optional[str] = None
) -> TrainingArguments:
    """
    Create training arguments for the Trainer.
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory (overrides config)
        
    Returns:
        TrainingArguments
    """
    # Get values from config with defaults
    is_finetuning = config.get('model.is_finetuning', False)
    mode = "finetuning" if is_finetuning else "pretraining"
    
    if output_dir is None:
        output_dir = config.get('experiment.output_dir', f'outputs/{mode}_model')
    
    # Determine batch size based on available GPUs
    batch_size = config.get('model.batch_size', 1)
    gradient_accumulation_steps = config.get('model.gradient_accumulation_steps', 1)
    
    # Different settings for finetuning vs pretraining
    if is_finetuning:
        args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=config.get('model.num_epochs', 10),
            per_device_train_batch_size=batch_size // torch.cuda.device_count() if torch.cuda.is_available() else batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=config.get('model.learning_rate', 5e-5),
            warmup_steps=config.get('model.warmup_steps', 100),
            weight_decay=config.get('model.weight_decay', 0.01),
            adam_epsilon=config.get('model.adam_epsilon', 1e-6),
            lr_scheduler_type=config.get('model.scheduler', 'cosine'),
            logging_steps=config.get('model.logging_steps', 10),
            save_strategy=config.get('model.save_strategy', 'epoch'),
            eval_steps=config.get('model.eval_steps', 500),
            evaluation_strategy=config.get('model.evaluation_strategy', 'steps'),
            fp16=config.get('model.fp16', torch.cuda.is_available()),
            adam_beta1=config.get('model.adam_beta1', 0.9),
            adam_beta2=config.get('model.adam_beta2', 0.999),
            report_to=["wandb"] if config.get('experiment.use_wandb', True) else ["none"]
        )
    else:
        # Pretraining typically uses higher learning rates
        args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=config.get('model.num_epochs', 1000),
            per_device_train_batch_size=batch_size // torch.cuda.device_count() if torch.cuda.is_available() else batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=config.get('model.learning_rate', 1e-3),
            warmup_steps=config.get('model.warmup_steps', 200),
            weight_decay=config.get('model.weight_decay', 0.1),
            adam_epsilon=config.get('model.adam_epsilon', 1e-6),
            lr_scheduler_type=config.get('model.scheduler', 'cosine'),
            logging_steps=config.get('model.logging_steps', 100),
            save_strategy=config.get('model.save_strategy', 'epoch'),
            eval_steps=config.get('model.eval_steps', 500),
            evaluation_strategy=config.get('model.evaluation_strategy', 'steps'),
            fp16=config.get('model.fp16', torch.cuda.is_available()),
            adam_beta1=config.get('model.adam_beta1', 0.9),
            adam_beta2=config.get('model.adam_beta2', 0.999),
            report_to=["wandb"] if config.get('experiment.use_wandb', True) else ["none"]
        )
    
    logger.info(f"Created training arguments for {mode}")
    return args


def save_model_and_tokenizer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: str
):
    """
    Save model and tokenizer to a directory.
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        output_dir: Directory where to save
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    
    logger.info(f"Saving tokenizer to {output_dir}")
    tokenizer.save_pretrained(output_dir)


def load_model_and_tokenizer(
    model_dir: str
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load model and tokenizer from a directory.
    
    Args:
        model_dir: Directory containing the model and tokenizer
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model from {model_dir}")
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    
    logger.info(f"Loading tokenizer from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    return model, tokenizer
