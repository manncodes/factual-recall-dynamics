"""
Classes for generating attribute values for knowledge graph entities.
"""

import random
import string
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Set, Optional


class AttributeGenerator(ABC):
    """
    Abstract base class for attribute generators.
    """
    
    @abstractmethod
    def get_all_possible_values(self) -> List[str]:
        """
        Get all possible values this generator can produce.
        
        Returns:
            List of all possible attribute values
        """
        pass


class TemplateGenerator(AttributeGenerator):
    """
    Generate attribute values using templates with placeholders.
    
    Templates can contain placeholders like:
    - {int:start-end} - Random integer in range [start, end]
    - {letter} - Random uppercase letter
    - {string:length} - Random lowercase string of given length
    """
    
    def __init__(self, template: str, num_values: int, random_seed: Optional[int] = None):
        """
        Initialize the template generator.
        
        Args:
            template: Template string with placeholders
            num_values: Number of unique values to generate
            random_seed: Random seed for reproducibility
        """
        self.template = template
        self.num_values = num_values
        
        # Set random seed if provided
        if random_seed is not None:
            random.seed(random_seed)
            
        self.values = self.generate_values()
    
    def generate_values(self) -> List[str]:
        """
        Generate the requested number of unique values.
        
        Returns:
            List of generated values
        """
        values = set()
        placeholders = re.findall(r'\{([^}]+)\}', self.template)
        
        # Try to generate unique values up to a reasonable limit
        max_attempts = self.num_values * 10
        attempts = 0
        
        while len(values) < self.num_values and attempts < max_attempts:
            value = self.template
            for placeholder in placeholders:
                replacement = self._generate_placeholder_value(placeholder)
                value = value.replace('{' + placeholder + '}', replacement)
            
            values.add(value)
            attempts += 1
            
        # Sort for deterministic order
        return sorted(list(values))
    
    def _generate_placeholder_value(self, placeholder: str) -> str:
        """
        Generate a value for a specific placeholder.
        
        Args:
            placeholder: The placeholder string
            
        Returns:
            Generated value for the placeholder
            
        Raises:
            ValueError: If placeholder format is invalid
        """
        if placeholder.startswith('int'):
            try:
                range_start, range_end = map(int, placeholder.split(':')[1].split('-'))
                return str(random.randint(range_start, range_end))
            except (IndexError, ValueError):
                raise ValueError(f"Invalid int placeholder format: {placeholder}")
            
        elif placeholder == 'letter':
            return random.choice(string.ascii_uppercase)
            
        elif placeholder.startswith('string'):
            try:
                length = int(placeholder.split(':')[1])
                return ''.join(random.choices(string.ascii_lowercase, k=length))
            except (IndexError, ValueError):
                raise ValueError(f"Invalid string placeholder format: {placeholder}")
            
        else:
            raise ValueError(f"Unknown placeholder type: {placeholder}")
    
    def get_all_possible_values(self) -> List[str]:
        """
        Get all generated values.
        
        Returns:
            List of all generated values
        """
        return self.values


class FileBasedGenerator(AttributeGenerator):
    """
    Generate attribute values from a file.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize with a file containing values.
        
        Args:
            file_path: Path to file containing values (one per line)
        """
        self.file_path = file_path
        self.values = self.load_values()
    
    def load_values(self) -> List[str]:
        """
        Load values from the file.
        
        Returns:
            List of values loaded from the file
            
        Raises:
            FileNotFoundError: If the file does not exist
        """
        try:
            with open(self.file_path, 'r') as f:
                # Strip whitespace, filter empty lines, and sort for deterministic order
                return sorted([line.strip() for line in f if line.strip()])
        except FileNotFoundError:
            raise FileNotFoundError(f"Attribute value file not found: {self.file_path}")
    
    def get_all_possible_values(self) -> List[str]:
        """
        Get all loaded values.
        
        Returns:
            List of all loaded values
        """
        return self.values
