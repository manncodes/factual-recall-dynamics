"""
Data processing utilities for converting knowledge graph data to training examples.
"""

import os
import json
import random
import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from src.data.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

# Default templates for converting facts to text
DEFAULT_KG_TEXT_TEMPLATES = {
    "birth_date": [
        "{name} was born on {birth_date}.",
        "Born on {birth_date}, {name} entered the world.",
        "{name}'s date of birth is {birth_date}.",
        "The {birth_date} marks {name}'s birthday.",
        "On {birth_date}, {name} took their first breath."
    ],
    "birth_city": [
        "{name} was born in {birth_city}.",
        "{birth_city} is the birthplace of {name}.",
        "{name}'s roots can be traced back to {birth_city}.",
        "Originally from {birth_city}, {name} has come a long way.",
        "{name} first saw the light of day in {birth_city}."
    ],
    "university": [
        "{name} studied at {university}.",
        "{university} is where {name} pursued higher education.",
        "{name}'s alma mater is {university}.",
        "During their college years, {name} attended {university}.",
        "{name} is a proud graduate of {university}."
    ],
    "major": [
        "{name} majored in {major}.",
        "{major} was {name}'s field of study.",
        "{name}'s academic focus was on {major}.",
        "During university, {name} specialized in {major}.",
        "{name} dedicated their studies to {major}."
    ],
    "employer": [
        "{name} works for {employer}.",
        "{employer} is lucky to have {name} on their team.",
        "{name}'s professional home is at {employer}.",
        "Currently, {name} is employed by {employer}.",
        "{name} contributes their skills to {employer}."
    ],
    "working_city": [
        "{name} works in {working_city}.",
        "{working_city} is where {name} pursues their career.",
        "{name}'s professional life is based in {working_city}.",
        "For work, {name} commutes to {working_city}.",
        "{name} contributes to the workforce of {working_city}."
    ]
}


def load_templates(template_file: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Load text templates from a file or use defaults.
    
    Args:
        template_file: Path to JSON file containing templates
        
    Returns:
        Dictionary mapping attribute names to lists of templates
    """
    if template_file and os.path.exists(template_file):
        with open(template_file, 'r') as f:
            templates = json.load(f)
        logger.info(f"Loaded templates from {template_file}")
        return templates
    
    logger.info("Using default templates")
    return DEFAULT_KG_TEXT_TEMPLATES


def save_templates(templates: Dict[str, List[str]], output_file: str):
    """
    Save templates to a JSON file.
    
    Args:
        templates: Dictionary mapping attribute names to lists of templates
        output_file: Path where to save the templates
    """
    with open(output_file, 'w') as f:
        json.dump(templates, f, indent=2)
    
    logger.info(f"Saved templates to {output_file}")


def kg_individual_to_text(
    individual: Dict[str, Any], 
    templates: Dict[str, List[str]], 
    attribute_order: Optional[List[str]] = None
) -> str:
    """
    Convert an individual's attributes to a natural language description.
    
    Args:
        individual: Dictionary representing an individual
        templates: Dictionary mapping attribute names to lists of templates
        attribute_order: Optional order of attributes (if None, use dictionary order)
        
    Returns:
        Text description of the individual
    """
    name = individual['name']
    attributes = individual['attributes']

    if attribute_order is None:
        attribute_order = list(attributes.keys())

    bio = []
    for attr in attribute_order:
        if attr in attributes and attr in templates:
            template = random.choice(templates[attr])
            bio.append(template.format(name=name, **{attr: attributes[attr]}))

    return ' '.join(bio)


def kg_to_text(
    kg_json: str, 
    templates: Dict[str, List[str]],
    permute_individuals: bool = False, 
    permute_attributes: bool = False
) -> List[str]:
    """
    Convert an entire knowledge graph to a list of text descriptions.
    
    Args:
        kg_json: JSON representation of a knowledge graph
        templates: Dictionary mapping attribute names to lists of templates
        permute_individuals: Whether to randomize the order of individuals
        permute_attributes: Whether to randomize the order of attributes
        
    Returns:
        List of text descriptions, one per individual
    """
    kg = json.loads(kg_json)
    individuals = kg['individuals']

    if permute_individuals:
        random.shuffle(individuals)

    return [
        kg_individual_to_text(
            individual,
            templates,
            random.sample(list(individual['attributes'].keys()), k=len(individual['attributes']))
            if permute_attributes else None
        )
        for individual in individuals
    ]


def create_pretraining_data(
    kg: KnowledgeGraph, 
    templates: Dict[str, List[str]],
    config: Dict[str, Any]
) -> tuple[str, str]:
    """
    Create pretraining data from a knowledge graph.
    
    Args:
        kg: Knowledge graph to use
        templates: Dictionary mapping attribute names to lists of templates
        config: Configuration dictionary with the following keys:
            - n_epochs: Number of epochs (data variations) to create
            - data_dir: Directory to store generated data
            - vocab_dir: Directory to store vocabulary
            - permute_individuals: Whether to randomize individual order
            - permute_attributes: Whether to randomize attribute order
            
    Returns:
        Tuple of (data_dir, vocab_dir) paths
    """
    n_epochs = config.get("n_epochs", 10)
    create_vocab = config.get("create_vocab", True)
    data_dir = config.get("data_dir", "data")
    vocab_dir = config.get("vocab_dir", "vocab")
    permute_individuals = config.get("permute_individuals", True)
    permute_attributes = config.get("permute_attributes", True)

    # Create vocabulary
    if create_vocab:
        os.makedirs(vocab_dir, exist_ok=True)

        vocab = kg.get_vocabulary()
        for k, v in vocab.items():
            with open(os.path.join(vocab_dir, f"{k}.txt"), "w") as f:
                f.write("\n".join(v))

    # Create pretraining data
    os.makedirs(data_dir, exist_ok=True)

    logger.info(f"Creating {n_epochs} epochs of pretraining data")
    for epoch in tqdm(range(n_epochs), desc="Generating training data"):
        bios = kg_to_text(
            kg.to_json(), 
            templates,
            permute_individuals=permute_individuals, 
            permute_attributes=permute_attributes
        )
        
        with open(os.path.join(data_dir, f"epoch_{epoch}.txt"), "w") as f:
            f.write("\n".join(bios))

    logger.info(f"Created pretraining data in {data_dir}")
    return data_dir, vocab_dir


def get_new_vocab(vocab_dir: str) -> set:
    """
    Get all vocabulary terms from a directory.
    
    Args:
        vocab_dir: Directory containing vocabulary files
        
    Returns:
        Set of all vocabulary terms
    """
    vocab = set()
    for file in os.listdir(vocab_dir):
        with open(os.path.join(vocab_dir, file), "r") as f:
            vocab.update(f.read().splitlines())
    return vocab
