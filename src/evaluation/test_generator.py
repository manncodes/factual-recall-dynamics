"""
Utilities for generating test cases to evaluate factual recall.
"""

import re
import json
import logging
import random
from typing import List, Dict, Any, Optional
from collections import Counter

logger = logging.getLogger(__name__)


def word_appears_before(text: str, word1: str, word2: str) -> bool:
    """
    Check if word1 appears before word2 in a text.
    
    Args:
        text: Text to search in
        word1: First word to find
        word2: Second word to find
        
    Returns:
        True if word1 appears before word2, False otherwise
    """
    pattern = rf"\b({word1}|{word2})\b"
    matches = re.findall(pattern, text)
    try:
        return matches.index(word1) < matches.index(word2)
    except ValueError:  # If either word is not found
        return False


def remove_after_word(text: str, word: str) -> str:
    """
    Remove everything after a word in a text.
    
    Args:
        text: Text to process
        word: Word after which to truncate
        
    Returns:
        Truncated text
    """
    try:
        match = re.search(rf"\b{re.escape(word)}\b", text)
        if match:
            index = match.start()
            return text[:index - 1]  # -1 to remove the preceding space
        else:
            return text
    except ValueError:
        return text


def create_triplet_tests(
    kg: Dict[str, Any], 
    kg_templates: Dict[str, List[str]], 
    num_individuals: Optional[int] = None,
    random_seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Create test cases for evaluating factual recall.
    
    Args:
        kg: Knowledge graph dictionary
        kg_templates: Templates for generating facts
        num_individuals: Optional limit on the number of individuals to test
        random_seed: Optional random seed for reproducibility
        
    Returns:
        List of test cases
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    tests = []
    individuals = kg["individuals"]
    
    # Limit the number of individuals if specified
    if num_individuals is not None:
        individuals = individuals[:num_individuals]
    
    # Create test cases for each individual
    for ind in individuals:
        test_templates = {}
        
        # Process each attribute
        for attr_name, attr_templates in kg_templates.items():
            if attr_name not in ind['attributes']:
                continue
                
            attr_test_templates = []
            
            # Process each template for the attribute
            for template in attr_templates:
                template = template.replace('.', '')
                
                if word_appears_before(template, 'name', attr_name):
                    # Test case where we provide the name and ask for the attribute value
                    # Example: "John was born on" -> expecting "1980/01/01"
                    prompt = remove_after_word(template, attr_name)
                    prompt = prompt.replace('{name}', ind['name'])
                else:
                    # Test case where we provide the attribute value and ask for the name
                    # Example: "Born on 1980/01/01" -> expecting "John"
                    prompt = remove_after_word(template, 'name')
                    prompt = prompt.replace(f'{{{attr_name}}}', ind['attributes'][attr_name])
                
                attr_test_templates.append(prompt)
            
            test_templates[attr_name] = attr_test_templates
        
        # Add the test case
        test = {
            "name": ind['name'],
            "individual": ind,
            "facts": test_templates
        }
        tests.append(test)
    
    logger.info(f"Created {len(tests)} test cases with {sum(len(facts) for test in tests for facts in test['facts'].values())} total prompts")
    return tests


def get_triplets_testing_data(epoch_data_file: str, kg_file: str) -> List[Dict[str, Any]]:
    """
    Parse training data to extract triplets for testing.
    
    Args:
        epoch_data_file: Path to an epoch data file
        kg_file: Path to the knowledge graph file
        
    Returns:
        List of triplet test data
    """
    # Read epoch data
    with open(epoch_data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Read knowledge graph
    with open(kg_file, 'r') as f:
        kg = json.load(f)
    
    # Create a mapping from name to individual
    name_to_individual = {ind["name"]: ind for ind in kg["individuals"]}
    
    # Parse each line to extract triplets
    triplets = []
    for line in lines:
        # Split into individual facts (sentences)
        facts = line.strip().split('.')[:-1]  # Exclude the last empty item
        facts = [fact.strip() for fact in facts]
        
        # Find the most common word, which is likely the name
        words = ' '.join(facts).split()
        word_counts = Counter(words)
        name = word_counts.most_common(1)[0][0]
        
        # Create test case
        test = {
            "name": name,
            "individual": name_to_individual.get(name, {}),
            "facts": facts
        }
        triplets.append(test)
    
    return triplets


def save_tests(tests: List[Dict[str, Any]], output_file: str):
    """
    Save test cases to a JSON file.
    
    Args:
        tests: List of test cases
        output_file: Path where to save the tests
    """
    with open(output_file, 'w') as f:
        json.dump(tests, f, indent=2)
    
    logger.info(f"Saved {len(tests)} test cases to {output_file}")


def load_tests(input_file: str) -> List[Dict[str, Any]]:
    """
    Load test cases from a JSON file.
    
    Args:
        input_file: Path to the JSON file
        
    Returns:
        List of test cases
    """
    with open(input_file, 'r') as f:
        tests = json.load(f)
    
    logger.info(f"Loaded {len(tests)} test cases from {input_file}")
    return tests


def sample_tests(
    tests: List[Dict[str, Any]], 
    num_samples: int,
    random_seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Sample a subset of test cases.
    
    Args:
        tests: List of all test cases
        num_samples: Number of test cases to sample
        random_seed: Optional random seed for reproducibility
        
    Returns:
        Sampled test cases
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    # Ensure num_samples is not larger than the number of tests
    num_samples = min(num_samples, len(tests))
    
    # Sample test cases
    sampled_tests = random.sample(tests, num_samples)
    
    logger.info(f"Sampled {len(sampled_tests)} test cases")
    return sampled_tests
