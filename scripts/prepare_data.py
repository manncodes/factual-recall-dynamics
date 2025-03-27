#!/usr/bin/env python
"""
Script for preparing data for experiments without running training.

This is useful for setting up data on a cluster before running the main experiment.
"""

import os
import sys
import json
import logging
import random
import argparse
from pathlib import Path

import numpy as np

# Add the parent directory to sys.path to allow importing from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config, load_config, setup_directories
from src.utils.logging import setup_logging, get_run_name

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
from src.evaluation.test_generator import create_triplet_tests, save_tests


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare data for factual recall experiments")
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
        "--num-entities", 
        type=int, 
        default=None,
        help="Number of entities to generate (overrides config)"
    )
    parser.add_argument(
        "--name-mode", 
        type=str, 
        choices=["subword_prefix", "whole_prefix"],
        default=None,
        help="Name tokenization mode (overrides config)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="Random seed (overrides config)"
    )
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.output_dir:
        config.set('experiment.output_dir', args.output_dir)
    
    if args.num_entities:
        config.set('data.num_personalities', args.num_entities)
    
    if args.name_mode:
        if args.name_mode == "subword_prefix":
            config.set('data.do_name_subword_tokenization', True)
        else:
            config.set('data.do_name_subword_tokenization', False)
    
    if args.seed:
        config.set('experiment.seed', args.seed)
    
    # Get experiment name
    exp_name = config.get(
        'experiment.name', 
        get_run_name(prefix="data_prep")
    )
    
    # Setup output directories
    output_dir = config.get('experiment.output_dir', f"outputs/{exp_name}")
    setup_directories(config)
    
    # Setup logging
    log_file = os.path.join(output_dir, "logs", "data_prep.log")
    logger = setup_logging(
        log_level=config.get('experiment.log_level', "INFO"),
        log_file=log_file
    )
    
    logger.info(f"Starting data preparation: {exp_name}")
    logger.info(f"Configuration: {config}")
    
    # Set random seed
    seed = config.get('experiment.seed', 42)
    set_seed(seed)
    logger.info(f"Using random seed: {seed}")
    
    try:
        # Step 1: Generate knowledge graph
        kg_file = os.path.join(output_dir, "synthetic_kg.json")
        
        # Create knowledge graph config
        kg_config_file = os.path.join(output_dir, "kg_config.json")
        
        # Define data parameters
        num_personalities = config.get('data.num_personalities', 1000)
        do_name_subword_tokenization = config.get('data.do_name_subword_tokenization', True)
        tokenization_type = "shared_prefix_name" if do_name_subword_tokenization else "whole_prefix_name"
        
        logger.info(f"Generating {num_personalities} entities with {tokenization_type} tokenization")
        
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
        
        # Step 2: Create text templates
        template_file = os.path.join(output_dir, "kg_text_templates.json")
        templates = load_templates()
        save_templates(templates, template_file)
        
        # Step 3: Create test cases
        test_file = os.path.join(output_dir, "tests.json")
        num_test_personalities = config.get('data.num_test_personalities', 10)
        
        logger.info(f"Generating test cases for {num_test_personalities} entities")
        tests = create_triplet_tests(
            json.loads(kg.to_json()), 
            templates, 
            num_individuals=num_test_personalities,
            random_seed=seed
        )
        save_tests(tests, test_file)
        logger.info(f"Saved test cases to {test_file}")
        
        # Step 4: Create pretraining data
        logger.info("Creating pretraining data")
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
        
        logger.info(f"Data preparation completed successfully for {exp_name}")
        
    except Exception as e:
        logger.exception(f"Error during data preparation: {e}")
        raise
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
