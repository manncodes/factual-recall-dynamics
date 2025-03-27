"""
Metrics and evaluation utilities for factual recall.
"""

import torch
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


def get_token_metrics(
    model: PreTrainedModel, 
    inputs: Dict[str, torch.Tensor], 
    target_token_id: Union[int, torch.Tensor], 
    target_position: Optional[Union[int, List[int]]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate metrics for specific tokens in model outputs.
    
    Args:
        model: The language model
        inputs: Tokenized inputs
        target_token_id: ID of target token to evaluate, or tensor of IDs
        target_position: Position(s) of target token (optional, uses last position if None)
        
    Returns:
        Tuple of (log_probs, ranks) for the target tokens
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Move inputs to model device if needed
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Process one or multiple target IDs
    if isinstance(target_token_id, int):
        target_token_id = torch.tensor([target_token_id]).to(device)
    else:
        target_token_id = target_token_id.to(device)
    
    # Set default target position if not provided
    if target_position is None:
        target_position = [inputs['input_ids'].shape[1] - 1] * target_token_id.size(0)
    elif isinstance(target_position, int):
        target_position = [target_position] * target_token_id.size(0)
    
    # Calculate metrics efficiently
    with torch.no_grad():
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
        
        # Extract logits for the target positions
        batch_size = logits.size(0)
        batch_indices = torch.arange(batch_size, device=device)
        position_logits = logits[batch_indices, target_position]  # [batch_size, vocab_size]
        
        # Calculate log probabilities and ranks
        log_probs = torch.log_softmax(position_logits, dim=-1)
        log_prob_values = torch.gather(log_probs, 1, target_token_id.unsqueeze(1)).squeeze(1)
        
        # Calculate ranks
        rank_indices = torch.argsort(position_logits, dim=-1, descending=True)
        ranks = torch.where(rank_indices == target_token_id.unsqueeze(1))[1] + 1  # 1-based ranking
    
    return log_prob_values, ranks


def batch_evaluate_prompts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    targets: List[str],
    batch_size: int = 8
) -> Dict[str, List[float]]:
    """
    Evaluate a batch of prompts against their target completions.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of prompt strings
        targets: List of target completion strings
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary with evaluation results
    """
    assert len(prompts) == len(targets), "Prompts and targets must have the same length"
    
    results = {
        "prompt": [],
        "target": [],
        "log_prob": [],
        "rank": [],
        "is_correct": []
    }
    
    # Process in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_targets = targets[i:i + batch_size]
        
        # Tokenize prompts
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        
        # Get target token IDs
        target_ids = torch.tensor([
            tokenizer.encode(target)[0] for target in batch_targets
        ])
        
        # Calculate metrics
        log_probs, ranks = get_token_metrics(model, inputs, target_ids)
        
        # Generate completions for evaluation
        completions = []
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=8,
                do_sample=False,
                num_beams=1
            )
            
            for j, output in enumerate(outputs):
                completion = tokenizer.decode(output[inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                completions.append(completion)
        
        # Determine correctness
        is_correct = [target in completion for target, completion in zip(batch_targets, completions)]
        
        # Store results
        results["prompt"].extend(batch_prompts)
        results["target"].extend(batch_targets)
        results["log_prob"].extend(log_probs.cpu().numpy().tolist())
        results["rank"].extend(ranks.cpu().numpy().tolist())
        results["is_correct"].extend(is_correct)
    
    return results


def evaluate_test_cases(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    tests: List[Dict[str, Any]],
    batch_size: int = 8
) -> pd.DataFrame:
    """
    Evaluate a set of test cases.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        tests: List of test cases
        batch_size: Batch size for evaluation
        
    Returns:
        DataFrame with evaluation results
    """
    all_prompts = []
    all_targets = []
    prompt_metadata = []
    
    # Prepare prompts and targets
    for test in tests:
        name = test['name']
        for attr, prompts in test['facts'].items():
            attr_value = test['individual']['attributes'][attr]
            
            for prompt in prompts:
                # Determine if we're querying for the name or the attribute value
                querying_name = attr in prompt
                
                all_prompts.append(prompt)
                if querying_name:
                    all_targets.append(name)
                else:
                    all_targets.append(attr_value)
                
                prompt_metadata.append({
                    'name': name,
                    'attribute': attr,
                    'true_value': attr_value,
                    'querying_what': 'name' if querying_name else 'attr_value'
                })
    
    # Evaluate all prompts
    results = batch_evaluate_prompts(
        model, tokenizer, all_prompts, all_targets, batch_size
    )
    
    # Add metadata to results
    for i, meta in enumerate(prompt_metadata):
        for key, value in meta.items():
            if key not in results:
                results[key] = []
            if i < len(results[key]):
                results[key][i] = value
            else:
                results[key].append(value)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate aggregated metrics
    logger.info(f"Overall accuracy: {df['is_correct'].mean():.4f}")
    logger.info(f"Average log probability: {df['log_prob'].mean():.4f}")
    logger.info(f"Average rank: {df['rank'].mean():.4f}")
    
    return df


def analyze_results(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze evaluation results and compute summary statistics.
    
    Args:
        df: DataFrame with evaluation results
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {}
    
    # Overall metrics
    analysis["overall"] = {
        "accuracy": df["is_correct"].mean(),
        "log_prob": df["log_prob"].mean(),
        "rank": df["rank"].mean(),
        "count": len(df)
    }
    
    # Per-attribute metrics
    analysis["per_attribute"] = {}
    for attr in df["attribute"].unique():
        attr_df = df[df["attribute"] == attr]
        analysis["per_attribute"][attr] = {
            "accuracy": attr_df["is_correct"].mean(),
            "log_prob": attr_df["log_prob"].mean(),
            "rank": attr_df["rank"].mean(),
            "count": len(attr_df)
        }
    
    # Metrics by query type
    analysis["query_type"] = {}
    for query_type in df["querying_what"].unique():
        query_df = df[df["querying_what"] == query_type]
        analysis["query_type"][query_type] = {
            "accuracy": query_df["is_correct"].mean(),
            "log_prob": query_df["log_prob"].mean(),
            "rank": query_df["rank"].mean(),
            "count": len(query_df)
        }
    
    # Top/bottom performers
    analysis["examples"] = {
        "best_log_prob": df.loc[df["log_prob"].idxmax()].to_dict(),
        "worst_log_prob": df.loc[df["log_prob"].idxmin()].to_dict(),
        "best_rank": df.loc[df["rank"].idxmin()].to_dict(),
        "worst_rank": df.loc[df["rank"].idxmax()].to_dict()
    }
    
    return analysis


def save_results(df: pd.DataFrame, output_file: str):
    """
    Save evaluation results to a file.
    
    Args:
        df: DataFrame with evaluation results
        output_file: Path where to save the results
    """
    if output_file.endswith('.csv'):
        df.to_csv(output_file, index=False)
    elif output_file.endswith('.json'):
        df.to_json(output_file, orient='records', indent=2)
    elif output_file.endswith('.pkl'):
        df.to_pickle(output_file)
    else:
        df.to_csv(output_file, index=False)
    
    logger.info(f"Saved evaluation results to {output_file}")
