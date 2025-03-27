"""
Utilities for tokenization and vocabulary management.
"""

import re
import json
import os
import logging
from typing import Set, List, Dict, Any, Optional
from pathlib import Path

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
from tokenizers import AddedToken

logger = logging.getLogger(__name__)


def process_name_tokens(data: Dict[str, Any], do_subword_tokenization: bool = False) -> Set[str]:
    """
    Process entity names into tokens based on the tokenization strategy.
    
    Args:
        data: Knowledge graph data as a dictionary
        do_subword_tokenization: Whether to perform subword tokenization
        
    Returns:
        Set of name tokens
    """
    if isinstance(data, str):
        data = json.loads(data)

    all_names = [individual['name'] for individual in data.get('individuals', [])]
    name_tokens = set(all_names)

    # Handle subword tokenization if requested
    if do_subword_tokenization:
        subword_tokens = set()
        for name in all_names:
            # Find subwords using camel case pattern
            subwords = re.findall(r'[A-Z][a-z]*', name)
            if len(subwords) >= 2:
                subword_tokens.update(subwords)
            else:
                subword_tokens.add(name)

        # Read additional name files if available
        additional_files = ['fnames.txt', 'lnames.txt']
        for filename in additional_files:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    subword_tokens.update(f.read().splitlines())
            else:
                logger.warning(f"Could not find file {filename}")

        name_tokens = subword_tokens
    else:
        # For whole name tokenization, add names from names.txt if available
        if os.path.exists('names.txt'):
            with open('names.txt', 'r') as f:
                name_tokens.update(f.read().splitlines())
        else:
            logger.warning("Could not find names.txt")

    return name_tokens


def add_new_tokens(tokenizer: PreTrainedTokenizer, new_tokens: List[str]) -> PreTrainedTokenizer:
    """
    Add new tokens to a tokenizer.
    
    Args:
        tokenizer: The tokenizer to modify
        new_tokens: List of tokens to add
        
    Returns:
        Modified tokenizer
    """
    # Convert to AddedToken objects for better control
    tokens_to_add = list(map(
        lambda x: AddedToken(f"{x}", single_word=True, lstrip=False, rstrip=False), 
        new_tokens
    ))
    
    num_added = tokenizer.add_tokens(tokens_to_add)
    logger.info(f"Added {num_added} new tokens to the tokenizer")
    
    return tokenizer


def get_all_tokens_from_kg(
    kg_file: str, 
    do_subword_tokenization: bool = False, 
    vocab_dir: Optional[str] = None
) -> Set[str]:
    """
    Extract all tokens from a knowledge graph.
    
    Args:
        kg_file: Path to knowledge graph JSON file
        do_subword_tokenization: Whether to perform subword tokenization
        vocab_dir: Optional directory containing additional vocabulary files
        
    Returns:
        Set of all tokens
    """
    # Load knowledge graph
    with open(kg_file, 'r') as f:
        kg_data = json.load(f)
    
    # Process name tokens
    tokens = process_name_tokens(kg_data, do_subword_tokenization)
    
    # Add tokens from vocabulary directory if provided
    if vocab_dir and os.path.exists(vocab_dir):
        for filename in os.listdir(vocab_dir):
            # Skip names.txt for subword mode since we process those differently
            if do_subword_tokenization and filename == 'names.txt':
                continue
                
            file_path = os.path.join(vocab_dir, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r') as f:
                    tokens.update(f.read().splitlines())
    
    return tokens


def prepare_tokenizer(
    base_model: str, 
    kg_file: str,
    do_subword_tokenization: bool = False,
    vocab_dir: Optional[str] = None
) -> PreTrainedTokenizer:
    """
    Prepare a tokenizer for the factual recall experiment.
    
    Args:
        base_model: HuggingFace model name or path
        kg_file: Path to knowledge graph JSON file
        do_subword_tokenization: Whether to perform subword tokenization
        vocab_dir: Optional directory containing additional vocabulary files
        
    Returns:
        Prepared tokenizer
    """
    # Initialize base tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Add special tokens if not present
    special_tokens = {
        "pad_token": "[PAD]",
        "sep_token": "[SEP]",
    }
    
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = "[PAD]"
    if tokenizer.sep_token is None:
        special_tokens["sep_token"] = "[SEP]"
        
    tokenizer.add_special_tokens(special_tokens)
    
    # Get all tokens from knowledge graph
    new_tokens = get_all_tokens_from_kg(
        kg_file, 
        do_subword_tokenization=do_subword_tokenization,
        vocab_dir=vocab_dir
    )
    
    # Add new tokens to tokenizer
    tokenizer = add_new_tokens(tokenizer, list(new_tokens))
    
    return tokenizer


def visualize_tokenization(tokens: List[int], tokenizer: PreTrainedTokenizer):
    """
    Visualize how a text is tokenized.
    
    Args:
        tokens: List of token IDs
        tokenizer: Tokenizer to use for decoding
    """
    # ANSI color codes for visualization
    colors = [
        '\033[48;5;223m',  # Pastel Pink
        '\033[48;5;193m',  # Pastel Green
        '\033[48;5;225m',  # Pastel Lavender
        '\033[48;5;230m',  # Pastel Yellow
        '\033[48;5;195m',  # Pastel Blue
        '\033[48;5;224m',  # Pastel Peach
        '\033[48;5;194m',  # Pastel Mint
        '\033[48;5;189m',  # Pastel Purple
    ]
    reset_color = '\033[0m'
    text_color = '\033[38;5;0m'  # Black text for better readability

    # Get the decoded form of each token
    decoded_tokens = [tokenizer.decode([token]) for token in tokens]

    # Print the color-coded tokenization
    color_index = 0
    for token in decoded_tokens:
        print(f"{colors[color_index]}{text_color}{token}{reset_color}", end="")
        color_index = (color_index + 1) % len(colors)
    print()  # New line at the end

    # Print token and character counts
    print(f"\nTokens: {len(tokens)}")


class TokenizationTester:
    """
    Utility class for testing tokenization strategies.
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        """
        Initialize with a tokenizer.
        
        Args:
            tokenizer: Tokenizer to test
        """
        self.tokenizer = tokenizer
    
    def test_name(self, name: str):
        """
        Test how a name is tokenized.
        
        Args:
            name: Name to tokenize
        """
        tokens = self.tokenizer.encode(name)
        print(f"Tokenization for name '{name}':")
        visualize_tokenization(tokens, self.tokenizer)
        
        # Check if tokenized as a single token
        if len(tokens) == 3:  # Including special tokens
            print("✅ Tokenized as a single token")
        else:
            print("❌ Split into multiple tokens")
    
    def test_fact(self, fact: str):
        """
        Test how a fact is tokenized.
        
        Args:
            fact: Fact to tokenize
        """
        tokens = self.tokenizer.encode(fact)
        print(f"Tokenization for fact '{fact}':")
        visualize_tokenization(tokens, self.tokenizer)
        print(f"Total tokens: {len(tokens)}")
        
    def test_multiple_facts(self, facts: List[str]):
        """
        Test tokenization of multiple facts and compare token counts.
        
        Args:
            facts: List of facts to tokenize
        """
        results = []
        
        for fact in facts:
            tokens = self.tokenizer.encode(fact)
            results.append({
                "fact": fact,
                "tokens": len(tokens),
                "chars": len(fact)
            })
        
        # Print results
        print("Tokenization comparison:")
        for r in results:
            print(f"Fact: {r['fact']}")
            print(f"  Tokens: {r['tokens']}, Characters: {r['chars']}, Ratio: {r['tokens']/r['chars']:.2f}")
            print()


def save_tokenizer(tokenizer: PreTrainedTokenizer, output_dir: str):
    """
    Save a tokenizer to a directory.
    
    Args:
        tokenizer: Tokenizer to save
        output_dir: Directory where to save the tokenizer
    """
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Saved tokenizer to {output_dir}")


def load_tokenizer(tokenizer_dir: str) -> PreTrainedTokenizer:
    """
    Load a tokenizer from a directory.
    
    Args:
        tokenizer_dir: Directory containing the tokenizer
        
    Returns:
        Loaded tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    logger.info(f"Loaded tokenizer from {tokenizer_dir} with {len(tokenizer)} tokens")
    return tokenizer
