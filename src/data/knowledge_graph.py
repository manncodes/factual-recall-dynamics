"""
Knowledge graph implementation for representing entities and their attributes.
"""

import json
import os
import random
import logging
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Set, Tuple, Optional

from src.data.attribute_generator import AttributeGenerator

logger = logging.getLogger(__name__)


class Individual:
    """
    Represents an individual entity in the knowledge graph.
    """
    
    def __init__(self, id: int, name: str, attributes: Dict[str, str]):
        """
        Initialize an individual.
        
        Args:
            id: Unique identifier
            name: Name of the individual
            attributes: Dictionary of attribute name -> attribute value
        """
        self.id = id
        self.name = name
        self.attributes = attributes
    
    def __repr__(self) -> str:
        """String representation of the individual."""
        return f"Individual(id={self.id}, name='{self.name}', attributes={self.attributes})"


class KGConfig:
    """
    Configuration for knowledge graph generation.
    """
    
    def __init__(self, attribute_generators: Dict[str, AttributeGenerator], name_generator: AttributeGenerator):
        """
        Initialize knowledge graph configuration.
        
        Args:
            attribute_generators: Dictionary of attribute name -> generator
            name_generator: Generator for entity names
        """
        self.attribute_generators = attribute_generators
        self.name_generator = name_generator


class KnowledgeGraph:
    """
    Knowledge graph representing entities and their attributes.
    """
    
    def __init__(self):
        """Initialize an empty knowledge graph."""
        self.individuals: List[Individual] = []
        self.graph = nx.Graph()
    
    def add_individual(self, individual: Individual):
        """
        Add an individual to the knowledge graph.
        
        Args:
            individual: Individual to add
        """
        self.individuals.append(individual)
        self.graph.add_node(individual.id, name=individual.name, attributes=individual.attributes)
    
    def remove_individual(self, individual_id: int):
        """
        Remove an individual from the knowledge graph.
        
        Args:
            individual_id: ID of the individual to remove
        """
        self.individuals = [ind for ind in self.individuals if ind.id != individual_id]
        self.graph.remove_node(individual_id)
    
    def update_individual(self, individual_id: int, new_attributes: Dict[str, str]):
        """
        Update an individual's attributes.
        
        Args:
            individual_id: ID of the individual to update
            new_attributes: New attributes to set or update
        """
        for ind in self.individuals:
            if ind.id == individual_id:
                ind.attributes.update(new_attributes)
                self.graph.nodes[individual_id]['attributes'].update(new_attributes)
                break
    
    def query_by_attribute(self, attribute: str, value: str) -> List[Individual]:
        """
        Find individuals with a specific attribute value.
        
        Args:
            attribute: Attribute name to query
            value: Attribute value to match
            
        Returns:
            List of matching individuals (sorted by ID for determinism)
        """
        return sorted([ind for ind in self.individuals if ind.attributes.get(attribute) == value],
                     key=lambda x: x.id)
    
    def get_individual(self, individual_id: int) -> Optional[Individual]:
        """
        Get an individual by ID.
        
        Args:
            individual_id: ID to look up
            
        Returns:
            The matching individual or None if not found
        """
        for ind in self.individuals:
            if ind.id == individual_id:
                return ind
        return None
    
    def merge(self, other_kg: 'KnowledgeGraph'):
        """
        Merge another knowledge graph into this one.
        
        Args:
            other_kg: Knowledge graph to merge in
        """
        self.individuals.extend(other_kg.individuals)
        self.graph = nx.compose(self.graph, other_kg.graph)
    
    def to_dict(self) -> Dict:
        """
        Convert knowledge graph to a dictionary.
        
        Returns:
            Dictionary representation of the knowledge graph
        """
        return {
            "individuals": sorted([  # Sort for deterministic order
                {
                    "id": ind.id,
                    "name": ind.name,
                    "attributes": dict(sorted(ind.attributes.items()))  # Sort attributes
                } for ind in self.individuals
            ], key=lambda x: x["id"])
        }
    
    def to_json(self) -> str:
        """
        Convert knowledge graph to JSON.
        
        Returns:
            JSON representation of the knowledge graph
        """
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)
    
    def from_json(self, json_data: str):
        """
        Load knowledge graph from JSON.
        
        Args:
            json_data: JSON representation of a knowledge graph
        """
        data = json.loads(json_data)
        for ind_data in sorted(data['individuals'], key=lambda x: x['id']):  # Sort for deterministic loading
            individual = Individual(ind_data['id'], ind_data['name'],
                                 dict(sorted(ind_data['attributes'].items())))  # Sort attributes
            self.add_individual(individual)
    
    def get_vocabulary(self) -> Dict[str, List[str]]:
        """
        Get vocabulary of entity names, attributes, and values.
        
        Returns:
            Dictionary mapping vocabulary type to sorted list of terms
        """
        vocabulary = {
            "names": set(),
            "attributes": set(),
            "values": set(),
            "relationship_types": set()
        }

        for individual in sorted(self.individuals, key=lambda x: x.id):
            vocabulary["names"].add(individual.name)
            for attr, value in sorted(individual.attributes.items()):
                vocabulary["attributes"].add(attr)
                vocabulary["values"].add(value)

        for _, _, data in sorted(self.graph.edges(data=True)):
            if 'type' in data:
                vocabulary["relationship_types"].add(data['type'])

        # Convert sets to sorted lists for deterministic output
        return {k: sorted(list(v)) for k, v in vocabulary.items()}
    
    def visualize(self, output_file: str = 'knowledge_graph.png'):
        """
        Visualize the knowledge graph.
        
        Args:
            output_file: Path where to save the visualization
        """
        # Set seed for spring layout to make it deterministic
        random.seed(42)
        
        pos = nx.spring_layout(self.graph, seed=42)
        plt.figure(figsize=(12, 8))
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue',
                node_size=500, font_size=8)
        
        edge_labels = nx.get_edge_attributes(self.graph, 'type')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        
        plt.title("Knowledge Graph Visualization")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
        logger.info(f"Knowledge graph visualization saved to {output_file}")


def calculate_max_entities(config: KGConfig) -> int:
    """
    Calculate the maximum number of entities that can be generated.
    
    Args:
        config: Knowledge graph configuration
        
    Returns:
        Maximum number of entities
    """
    return min(len(generator.get_all_possible_values()) for generator in config.attribute_generators.values())


def generate_synthetic_kg(config: KGConfig, random_seed: Optional[int] = None) -> KnowledgeGraph:
    """
    Generate a synthetic knowledge graph.
    
    Args:
        config: Knowledge graph configuration
        random_seed: Optional random seed for reproducibility
        
    Returns:
        Generated knowledge graph
    """
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
    
    kg = KnowledgeGraph()
    max_entities = calculate_max_entities(config)
    
    logger.info(f"Generating knowledge graph with up to {max_entities} entities")

    # Get all possible attribute values up front
    all_possible_values = {
        attr: generator.get_all_possible_values()
        for attr, generator in sorted(config.attribute_generators.items())
    }
    
    # Keep track of used values to ensure uniqueness
    used_values = {attr: set() for attr in all_possible_values.keys()}
    
    # Get all name values
    name_values = config.name_generator.get_all_possible_values()

    for i in range(max_entities):
        # Get name deterministically by index
        name = name_values[i % len(name_values)]
        attributes = {}

        for attr, values in sorted(all_possible_values.items()):
            # Get values deterministically
            available_values = sorted([v for v in values if v not in used_values[attr]])
            if not available_values:
                logger.warning(f"Ran out of unique values for attribute {attr}. Generated {i} entities.")
                return kg

            value = available_values[0]  # Take first available value
            attributes[attr] = value
            used_values[attr].add(value)

        individual = Individual(i, name, attributes)
        kg.add_individual(individual)
    
    logger.info(f"Generated knowledge graph with {len(kg.individuals)} entities")
    return kg


def load_kg_config(config_file: str) -> KGConfig:
    """
    Load knowledge graph configuration from a JSON file.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Knowledge graph configuration
        
    Raises:
        FileNotFoundError: If configuration file not found
        json.JSONDecodeError: If configuration file contains invalid JSON
    """
    from src.data.attribute_generator import FileBasedGenerator, TemplateGenerator
    
    with open(config_file, 'r') as f:
        config_data = json.load(f)

    attribute_generators = {}
    for attr, gen_config in config_data['attribute_generators'].items():
        if 'file_path' in gen_config:
            # Use FileBasedGenerator if file_path is specified
            attribute_generators[attr] = FileBasedGenerator(gen_config['file_path'])
        else:
            # Use TemplateGenerator for template-based generation
            attribute_generators[attr] = TemplateGenerator(
                gen_config['template'],
                gen_config['num_values']
            )
    
    # Handle name generator similarly
    name_gen_config = config_data['name_generator']
    if 'file_path' in name_gen_config:
        name_generator = FileBasedGenerator(name_gen_config['file_path'])
    else:
        name_generator = TemplateGenerator(
            name_gen_config['template'],
            name_gen_config['num_values']
        )
    
    return KGConfig(attribute_generators, name_generator)


def save_kg_to_file(kg: KnowledgeGraph, output_file: str):
    """
    Save a knowledge graph to a JSON file.
    
    Args:
        kg: Knowledge graph to save
        output_file: Path where to save the knowledge graph
    """
    with open(output_file, 'w') as f:
        f.write(kg.to_json())
    
    logger.info(f"Saved knowledge graph to {output_file}")


def load_kg_from_file(input_file: str) -> KnowledgeGraph:
    """
    Load a knowledge graph from a JSON file.
    
    Args:
        input_file: Path to the JSON file
        
    Returns:
        Loaded knowledge graph
        
    Raises:
        FileNotFoundError: If file not found
        json.JSONDecodeError: If file contains invalid JSON
    """
    with open(input_file, 'r') as f:
        json_data = f.read()
    
    kg = KnowledgeGraph()
    kg.from_json(json_data)
    
    logger.info(f"Loaded knowledge graph from {input_file} with {len(kg.individuals)} entities")
    return kg
