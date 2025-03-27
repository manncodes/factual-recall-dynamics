#!/usr/bin/env python
"""
Main script for running factual recall experiments.
"""

import os
import sys
import json
import logging
import random
import argparse
from pathlib import Path

import torch
import numpy as np
import wandb

# Add the parent directory to sys.path to allow importing from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config, load_config, setup_directories
from src.utils.logging import setup_logging, get_run_name, WandbLogger

from src.data.knowledge_graph import (
    generate_synthetic_kg, 
    load_kg_config, 
    save_kg_to_file, 
    load_kg_from_file
)
from src.data.data_processor import (
    load_templates, 
    save_templates, 
    create_pretraining_data
)
from src.data.tokenization import prepare_tokenizer
from src.evaluation.test_generator import create_triplet_tests, save_tests, load_tests

from src.models.utils import (
    create_or_load_model, 
    create_training_arguments, 
    save_model_and_tokenizer
)
from src.models.trainer import load_training_data, setup_training


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run factual recall experiment")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Run in debug mode (reduced dataset size)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="Random seed (overrides config)"
    )
    parser.add_argument(
        "--no-wandb", 
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.output_dir:
        config.set('experiment.output_dir', args.output_dir)
    
    if args.seed:
        config.set('experiment.seed', args.seed)
    
    if args.no_wandb:
        config.set('experiment.use_wandb', False)
    
    # Debug mode
    if args.debug:
        config.set('data.num_personalities', 10)
        config.set('data.num_test_personalities', 5)
        config.set('data.n_epochs', 2)
        config.set('model.num_epochs', 2)
        
    # Get experiment name
    exp_name = config.get(
        'experiment.name', 
        get_run_name(prefix="factual_recall")
    )
    
    # Setup output directories
    output_dir = config.get('experiment.output_dir', f"outputs/{exp_name}")
    setup_directories(config)
    
    # Setup logging
    log_file = os.path.join(output_dir, "logs", "experiment.log")
    logger = setup_logging(
        log_level=config.get('experiment.log_level', "INFO"),
        log_file=log_file
    )
    
    logger.info(f"Starting experiment: {exp_name}")
    logger.info(f"Configuration: {config}")
    
    # Set random seed
    seed = config.get('experiment.seed', 42)
    set_seed(seed)
    logger.info(f"Using random seed: {seed}")
    
    # Initialize Weights & Biases if enabled
    use_wandb = config.get('experiment.use_wandb', True)
    if use_wandb:
        wandb_logger = WandbLogger(
            project=config.get('experiment.wandb_project', "factual_recall_dynamics"),
            name=exp_name,
            config=config.config
        )
        wandb_logger.init()
        logger.info("Initialized Weights & Biases logging")
    
    try:
        # Step 1: Generate or load knowledge graph
        kg_file = os.path.join(output_dir, "synthetic_kg.json")
        if os.path.exists(kg_file):
            logger.info(f"Loading existing knowledge graph from {kg_file}")
            kg = load_kg_from_file(kg_file)
        else:
            logger.info("Generating synthetic knowledge graph")
            # Create knowledge graph config
            kg_config_file = os.path.join(output_dir, "kg_config.json")
            
            # Define data parameters
            num_personalities = config.get('data.num_personalities', 1000)
            do_name_subword_tokenization = config.get('data.do_name_subword_tokenization', True)
            tokenization_type = "shared_prefix_name" if do_name_subword_tokenization else "whole_prefix_name"
            
            # Create KG configuration
            kg_config_data = {
                "name_generator": {
                    "file_path": "names.txt",
                    "template": "name_{string:8}",
                    "num_values": num_personalities
                },
                "attribute_generators": {
                    "birth_date": {
                        "template": "{int:1950-2000}/{int:01-12}/{int:01-28}",
                        "num_values": num_personalities
                    },
                    "birth_city": {
                        "template": "Bcity{string:10}",
                        "num_values": num_personalities
                    },
                    "university": {
                        "template": "Uni{string:5}",
                        "num_values": num_personalities
                    },
                    "major": {
                        "template": "Maj{string:8}",
                        "num_values": num_personalities
                    },
                    "employer": {
                        "template": "Comp{string:6}",
                        "num_values": num_personalities
                    },
                    "working_city": {
                        "template": "Wcity{string:10}",
                        "num_values": num_personalities
                    }
                }
            }
            
            # Save KG config
            with open(kg_config_file, 'w') as f:
                json.dump(kg_config_data, f, indent=2)
                
            # Load KG config and generate KG
            kg_config = load_kg_config(kg_config_file)
            kg = generate_synthetic_kg(kg_config, random_seed=seed)
            
            # Save KG
            save_kg_to_file(kg, kg_file)
            logger.info(f"Saved knowledge graph to {kg_file}")
        
        # Step 2: Load or create text templates
        template_file = os.path.join(output_dir, "kg_text_templates.json")
        templates = load_templates(template_file)
        save_templates(templates, template_file)
        
        # Step 3: Create test cases
        test_file = os.path.join(output_dir, "tests.json")
        if os.path.exists(test_file):
            logger.info(f"Loading existing test cases from {test_file}")
            tests = load_tests(test_file)
        else:
            logger.info("Generating test cases")
            num_test_personalities = config.get('data.num_test_personalities', 10)
            tests = create_triplet_tests(
                json.loads(kg.to_json()), 
                templates, 
                num_individuals=num_test_personalities,
                random_seed=seed
            )
            save_tests(tests, test_file)
            logger.info(f"Saved test cases to {test_file}")
        
        # Step 4: Create pretraining data
        data_dir, vocab_dir = create_pretraining_data(
            kg, 
            templates, 
            {
                "n_epochs": config.get('data.n_epochs', 10),
                "data_dir": config.get('data.data_dir', "data"),
                "vocab_dir": config.get('data.vocab_dir', "vocab"),
                "permute_individuals": config.get('data.permute_individuals', True),
                "permute_attributes": config.get('data.permute_attributes', True)
            }
        )
        
        # Step 5: Prepare tokenizer
        model_name = config.get('model.model_name', "Qwen/Qwen2.5-0.5B")
        do_name_subword_tokenization = config.get('data.do_name_subword_tokenization', True)
        tokenizer = prepare_tokenizer(
            model_name, 
            kg_file, 
            do_name_subword_tokenization=do_name_subword_tokenization,
            vocab_dir=vocab_dir
        )
        
        # Step 6: Create or load model
        model, tokenizer = create_or_load_model(config)
        
        # Step 7: Load training data
        train_texts, eval_texts = load_training_data(
            data_dir, 
            eval_split=config.get('model.eval_split', 0.1),
            random_seed=seed
        )
        
        # Step: Create training arguments
        training_args = create_training_arguments(config, output_dir=output_dir)
        
        # Step 8: Setup trainer
        trainer = setup_training(
            model,
            tokenizer,
            train_texts,
            eval_texts,
            tests,
            training_args,
            seq_length=config.get('model.seq_length', 256),
            use_wandb=use_wandb
        )
        
        # Step 9: Train model
        logger.info("Starting training")
        train_result = trainer.train()
        
        # Step 10: Save final model
        save_model_and_tokenizer(
            model,
            tokenizer,
            os.path.join(output_dir, "final_model")
        )
        
        # Step 11: Final evaluation
        logger.info("Performing final evaluation")
        eval_result = trainer.evaluate()
        
        # Log final results
        logger.info(f"Final evaluation results: {eval_result}")
        if use_wandb:
            wandb.log({"final_eval": eval_result})
        
        # Save final results
        with open(os.path.join(output_dir, "final_results.json"), 'w') as f:
            json.dump({
                "train": train_result.metrics,
                "eval": eval_result
            }, f, indent=2)
            
        logger.info(f"Experiment completed successfully: {exp_name}")
        
        # Finish wandb
        if use_wandb:
            wandb_logger.finish()
            
    except Exception as e:
        logger.exception(f"Error during experiment: {e}")
        if use_wandb:
            wandb.finish(exit_code=1)
        raise
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
