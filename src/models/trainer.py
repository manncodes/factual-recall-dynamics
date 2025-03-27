"""
Custom trainer implementation for factual recall experiments.
"""

import os
import gc
import random
import logging
import numpy as np
import pandas as pd
import torch
import wandb
from typing import List, Dict, Any, Optional, Union, Tuple
from transformers import (
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizer,
    DataCollatorForLanguageModeling
)
from transformers.trainer_utils import EvalLoopOutput

from src.evaluation.metrics import get_token_metrics

logger = logging.getLogger(__name__)


class FactualRecallDataset(torch.utils.data.Dataset):
    """
    Dataset for training models on factual recall tasks.
    """
    
    def __init__(self, texts: List[str], tokenizer: PreTrainedTokenizer, seq_length: int = 512):
        """
        Initialize dataset.
        
        Args:
            texts: List of text examples
            tokenizer: Tokenizer to use
            seq_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length

        # Add separator token if it doesn't exist
        self.sep_token = "<|sep|>"
        if self.sep_token not in tokenizer.vocab:
            tokenizer.add_tokens([self.sep_token])

        # Pre-tokenize all texts with proper padding and truncation
        logger.info(f"Tokenizing {len(texts)} texts...")
        self.examples = []
        
        # Process in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encodings = tokenizer(
                batch_texts,
                truncation=True,
                max_length=seq_length,
                padding="max_length",
                return_tensors="pt"
            )
            self.examples.extend([{
                "input_ids": encodings.input_ids[j],
                "attention_mask": encodings.attention_mask[j]
            } for j in range(len(batch_texts))])

        logger.info(f"Created {len(self.examples)} training examples")

    def __getitem__(self, idx):
        """Get an item from the dataset."""
        item = {
            "input_ids": self.examples[idx]["input_ids"].clone(),
            "attention_mask": self.examples[idx]["attention_mask"].clone(),
        }
        item["labels"] = item["input_ids"].clone()
        # Set labels to -100 where we have padding (to ignore in loss calculation)
        item["labels"][item["attention_mask"] == 0] = -100
        return item

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.examples)


def load_training_data(data_dir: str, eval_split: float = 0.1, random_seed: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """
    Load training data from a directory and split into train and eval sets.
    
    Args:
        data_dir: Directory containing training data files
        eval_split: Fraction of data to use for evaluation
        random_seed: Optional random seed for reproducibility
        
    Returns:
        Tuple of (train_texts, eval_texts)
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    texts = []
    for filename in os.listdir(data_dir):
        if filename.startswith('epoch_') and filename.endswith('.txt'):
            with open(os.path.join(data_dir, filename), 'r') as f:
                texts.extend([line.strip() for line in f if line.strip()])

    # Randomly split into train and eval
    random.shuffle(texts)
    split_idx = int(len(texts) * (1 - eval_split))

    train_texts = texts[:split_idx]
    eval_texts = texts[split_idx:]

    logger.info(f"Loaded {len(train_texts)} training examples and {len(eval_texts)} evaluation examples")
    return train_texts, eval_texts


class FactualRecallTrainer(Trainer):
    """
    Custom trainer for factual recall evaluation during training.
    """
    
    def __init__(
        self, 
        *args, 
        test_data: Optional[List[Dict[str, Any]]] = None,
        use_wandb: bool = True,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        eval_ratio: float = 1.0,
        eval_batch_size: int = 32,
        **kwargs
    ):
        """
        Initialize trainer.
        
        Args:
            *args: Arguments for Trainer
            test_data: Test data for evaluation
            use_wandb: Whether to log to Weights & Biases
            tokenizer: Tokenizer to use
            eval_ratio: Fraction of test data to use for evaluation
            eval_batch_size: Batch size for evaluation
            **kwargs: Keyword arguments for Trainer
        """
        super().__init__(*args, **kwargs)
        self.test_data = test_data
        self.use_wandb = use_wandb
        self._tokenizer = tokenizer
        self.eval_ratio = eval_ratio
        self.eval_batch_size = eval_batch_size

    @property
    def tokenizer(self):
        """Get the tokenizer."""
        if self._tokenizer is None and hasattr(self, "processing_class"):
            return self.processing_class
        return self._tokenizer

    def prepare_eval_batch(self, sampled_individuals):
        """
        Prepare batched prompts and metadata for evaluation.
        
        Args:
            sampled_individuals: List of individuals to evaluate
            
        Returns:
            Tuple of (all_prompts, prompt_metadata)
        """
        all_prompts = []
        prompt_metadata = []  # Store info needed for checking results

        for test in sampled_individuals:
            for attr, prompts in test['facts'].items():
                for test_prompt in prompts:
                    all_prompts.append(test_prompt.strip())
                    prompt_metadata.append({
                        'name': test['name'],
                        'attribute': attr,
                        'true_value': test['individual']['attributes'][attr],
                        'querying_what': 'attr_value' if test['name'] in test_prompt else 'name'
                    })

        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return all_prompts, prompt_metadata

    def batch_get_token_metrics(self, scores_batch, target_token_ids):
        """
        Compute metrics for a batch of generations efficiently.
        
        Args:
            scores_batch: Batch of token scores
            target_token_ids: IDs of target tokens
            
        Returns:
            Tuple of (max_probs, min_ranks)
        """
        with torch.no_grad():
            # scores_batch shape: [num_tokens, batch_size, vocab_size]
            # Transpose to [batch_size, num_tokens, vocab_size]
            scores_batch = scores_batch.transpose(0, 1)

            log_probs = torch.log_softmax(scores_batch, dim=-1)
            probs = torch.exp(log_probs)

            # Expand target_token_ids to match scores dimensions
            # [batch_size] -> [batch_size, num_tokens, 1]
            target_tokens_expanded = target_token_ids.unsqueeze(1).unsqueeze(-1).expand(-1, scores_batch.shape[1], 1)

            # Get probabilities for target tokens
            token_probs = torch.gather(probs, 2, target_tokens_expanded).squeeze(-1)  # [batch_size, num_tokens]
            max_probs = token_probs.max(dim=1)[0]  # [batch_size]

            # Calculate ranks efficiently
            ranks = torch.argsort(scores_batch, dim=-1, descending=True)  # [batch_size, num_tokens, vocab_size]
            target_expanded = target_token_ids.unsqueeze(1).unsqueeze(-1)  # [batch_size, 1, 1]

            # Find positions where ranks match target tokens
            rank_positions = (ranks == target_expanded).nonzero()[:, 2] + 1  # Adding 1 for 1-based ranking
            min_ranks = rank_positions.view(scores_batch.shape[0], -1).min(dim=1)[0]

            # Clear memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return max_probs.cpu(), min_ranks.cpu()

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: bool = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval"
    ) -> EvalLoopOutput:
        """
        Custom evaluation loop that evaluates factual recall.
        
        Args:
            dataloader: DataLoader for evaluation
            description: Description for progress bar
            prediction_loss_only: Whether to only compute prediction loss
            ignore_keys: Keys to ignore in model outputs
            metric_key_prefix: Prefix for metric keys
            
        Returns:
            EvalLoopOutput with evaluation results
        """
        # Initialize model in eval mode
        model = self.model
        model.eval()
        current_epoch = int(self.state.epoch)

        # Set padding side to left for decoder-only models
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'left'

        try:
            # Sample individuals to test
            if self.test_data is None:
                logger.warning("No test data provided, skipping factual recall evaluation")
                # Just run the standard evaluation
                return super().evaluation_loop(
                    dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
                )
                
            num_test_individuals = max(1, int(len(self.test_data) * self.eval_ratio))
            sampled_individuals = random.sample(self.test_data, num_test_individuals)

            # Prepare all prompts and metadata
            all_prompts, prompt_metadata = self.prepare_eval_batch(sampled_individuals)

            # Process in batches
            results = []
            for i in range(0, len(all_prompts), self.eval_batch_size):
                batch_prompts = all_prompts[i:i + self.eval_batch_size]
                batch_metadata = prompt_metadata[i:i + self.eval_batch_size]

                # Tokenize batch
                inputs = self.tokenizer(
                    batch_prompts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(model.device)

                # Generate completions for batch
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        max_new_tokens=5,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        do_sample=False,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )

                # Get target token IDs for batch
                target_token_ids = torch.tensor([
                    self.tokenizer.encode(meta['name'] if meta['querying_what'] == 'name' else meta['true_value'])[0]
                    for meta in batch_metadata
                ]).to(model.device)

                # Calculate metrics for batch
                log_probs, ranks = self.batch_get_token_metrics(
                    torch.stack(outputs.scores),
                    target_token_ids
                )

                # Process generated sequences
                generated_texts = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

                # Store results
                for j, (generated, meta, log_prob, rank) in enumerate(zip(generated_texts, batch_metadata, log_probs, ranks)):
                    completion = generated[len(batch_prompts[j]):].strip()
                    is_correct = (
                        meta['true_value'] in completion
                        if meta['querying_what'] == 'attr_value'
                        else meta['name'] in completion
                    )

                    results.append({
                        'name': meta['name'],
                        'attribute': meta['attribute'],
                        'triplet_id': f"{meta['name']}_{meta['attribute']}",
                        'prompt': batch_prompts[j],
                        'generated': generated,
                        'completion': completion,
                        'true_value': meta['true_value'],
                        'is_correct': is_correct,
                        'querying_what': meta['querying_what'],
                        'log_prob': log_prob.item(),
                        'target_rank': rank.item(),
                        'epoch': current_epoch,
                    })

            # Compute metrics efficiently using numpy
            results_array = np.array([
                [r['is_correct'], r['log_prob'], r['target_rank']]
                for r in results
            ])

            accuracy = np.mean(results_array[:, 0])
            avg_log_prob = np.mean(results_array[:, 1])
            avg_rank = np.mean(results_array[:, 2])

            # Get learning rate
            lr = self.optimizer.param_groups[0]['lr']

            # Prepare metrics
            custom_metrics = {
                'Learning rate': lr,
                f'{metric_key_prefix}/fact_recall_correct': np.sum(results_array[:, 0]),
                f'{metric_key_prefix}/fact_recall_accuracy': accuracy,
                f'{metric_key_prefix}/avg_log_prob': avg_log_prob,
                f'{metric_key_prefix}/avg_target_rank': avg_rank
            }

            # Calculate per-attribute metrics efficiently
            results_df = pd.DataFrame(results)
            for attr in results_df['attribute'].unique():
                mask = results_df['attribute'] == attr
                custom_metrics[f'eval/accuracy_{attr}'] = results_df.loc[mask, 'is_correct'].mean()
                custom_metrics[f'eval/avg_rank_{attr}'] = results_df.loc[mask, 'target_rank'].mean()

            eval_output = EvalLoopOutput(
                predictions=None,
                label_ids=None,
                metrics=custom_metrics,
                num_samples=len(results)
            )

            # Log to Weights & Biases if enabled
            if self.use_wandb:
                wandb.log({
                    **custom_metrics,
                    'epoch': current_epoch,
                })

                # Log detailed results
                wandb.log({
                    f'{metric_key_prefix}/results_epoch_{current_epoch}':
                    wandb.Table(dataframe=results_df)
                })

            return eval_output

        finally:
            # Restore original padding side
            self.tokenizer.padding_side = original_padding_side


def setup_training(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_texts: List[str],
    eval_texts: List[str],
    test_data: Optional[List[Dict[str, Any]]],
    training_args: TrainingArguments,
    seq_length: int = 256,
    use_wandb: bool = True
) -> FactualRecallTrainer:
    """
    Set up the trainer for training.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer to use
        train_texts: Training texts
        eval_texts: Evaluation texts
        test_data: Test data for factual recall evaluation
        training_args: Training arguments
        seq_length: Maximum sequence length
        use_wandb: Whether to log to Weights & Biases
        
    Returns:
        Configured trainer
    """
    # Create datasets
    train_dataset = FactualRecallDataset(
        train_texts,
        tokenizer,
        seq_length=seq_length,
    )

    eval_dataset = FactualRecallDataset(
        eval_texts,
        tokenizer,
        seq_length=seq_length,
    )

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )

    # Create trainer
    trainer = FactualRecallTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        test_data=test_data,
        use_wandb=use_wandb,
        tokenizer=tokenizer,
    )

    return trainer
