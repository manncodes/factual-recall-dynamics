"""
Tests for the knowledge graph module.
"""

import os
import sys
import unittest
import tempfile
import json

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.knowledge_graph import (
    Individual,
    KGConfig,
    KnowledgeGraph,
    generate_synthetic_kg,
    save_kg_to_file,
    load_kg_from_file
)
from src.data.attribute_generator import TemplateGenerator


class TestKnowledgeGraph(unittest.TestCase):
    """Test the KnowledgeGraph class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple knowledge graph
        self.kg = KnowledgeGraph()
        
        self.individual1 = Individual(
            id=1, 
            name="JohnDoe", 
            attributes={
                "birth_date": "1990/01/01",
                "birth_city": "NewYork",
                "university": "Harvard"
            }
        )
        
        self.individual2 = Individual(
            id=2, 
            name="JaneDoe", 
            attributes={
                "birth_date": "1995/05/05",
                "birth_city": "Boston",
                "university": "MIT"
            }
        )
        
        self.kg.add_individual(self.individual1)
        self.kg.add_individual(self.individual2)
    
    def test_add_individual(self):
        """Test adding an individual to the knowledge graph."""
        self.assertEqual(len(self.kg.individuals), 2)
        self.assertIn(self.individual1, self.kg.individuals)
        self.assertIn(self.individual2, self.kg.individuals)
    
    def test_query_by_attribute(self):
        """Test querying by attribute."""
        result = self.kg.query_by_attribute("university", "Harvard")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, 1)
        
        result = self.kg.query_by_attribute("birth_city", "NonExistentCity")
        self.assertEqual(len(result), 0)
    
    def test_get_individual(self):
        """Test getting an individual by ID."""
        result = self.kg.get_individual(1)
        self.assertEqual(result, self.individual1)
        
        result = self.kg.get_individual(99)
        self.assertIsNone(result)
    
    def test_to_json_and_from_json(self):
        """Test converting to and from JSON."""
        json_str = self.kg.to_json()
        
        # Create a new knowledge graph
        new_kg = KnowledgeGraph()
        new_kg.from_json(json_str)
        
        self.assertEqual(len(new_kg.individuals), 2)
        self.assertEqual(new_kg.individuals[0].id, 1)
        self.assertEqual(new_kg.individuals[0].name, "JohnDoe")
        self.assertEqual(new_kg.individuals[0].attributes["university"], "Harvard")
    
    def test_save_and_load(self):
        """Test saving to and loading from a file."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            temp_file = tmp.name
            
        try:
            # Save the knowledge graph
            save_kg_to_file(self.kg, temp_file)
            
            # Load the knowledge graph
            loaded_kg = load_kg_from_file(temp_file)
            
            # Check that the loaded knowledge graph matches the original
            self.assertEqual(len(loaded_kg.individuals), 2)
            self.assertEqual(loaded_kg.individuals[0].name, "JohnDoe")
            self.assertEqual(loaded_kg.individuals[1].name, "JaneDoe")
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)
                
    def test_get_vocabulary(self):
        """Test getting vocabulary from the knowledge graph."""
        vocab = self.kg.get_vocabulary()
        
        self.assertIn("names", vocab)
        self.assertIn("attributes", vocab)
        self.assertIn("values", vocab)
        
        self.assertIn("JohnDoe", vocab["names"])
        self.assertIn("JaneDoe", vocab["names"])
        self.assertIn("birth_date", vocab["attributes"])
        self.assertIn("university", vocab["attributes"])
        self.assertIn("Harvard", vocab["values"])
        self.assertIn("MIT", vocab["values"])
        
    def test_update_individual(self):
        """Test updating an individual's attributes."""
        # Update individual 1
        self.kg.update_individual(1, {"university": "Stanford", "major": "ComputerScience"})
        
        # Check that the update worked
        individual = self.kg.get_individual(1)
        self.assertEqual(individual.attributes["university"], "Stanford")
        self.assertEqual(individual.attributes["major"], "ComputerScience")
        self.assertEqual(individual.attributes["birth_date"], "1990/01/01")  # Original attribute preserved
        
    def test_remove_individual(self):
        """Test removing an individual."""
        # Remove individual 1
        self.kg.remove_individual(1)
        
        # Check that the individual was removed
        self.assertEqual(len(self.kg.individuals), 1)
        self.assertIsNone(self.kg.get_individual(1))
        self.assertEqual(self.kg.individuals[0].id, 2)


class TestSyntheticKG(unittest.TestCase):
    """Test the synthetic knowledge graph generation."""
    
    def test_generate_synthetic_kg(self):
        """Test generating a synthetic knowledge graph."""
        # Create attribute generators
        name_generator = TemplateGenerator("Name{int:1-100}", 10)
        birth_date_generator = TemplateGenerator("{int:1990-2000}/{int:01-12}/{int:01-28}", 10)
        city_generator = TemplateGenerator("City{string:5}", 10)
        
        attribute_generators = {
            "birth_date": birth_date_generator,
            "birth_city": city_generator
        }
        
        # Create KG config
        kg_config = KGConfig(attribute_generators, name_generator)
        
        # Generate synthetic KG
        kg = generate_synthetic_kg(kg_config, random_seed=42)
        
        # Check that the KG has the expected number of individuals
        self.assertEqual(len(kg.individuals), 10)
        
        # Check that all individuals have the expected attributes
        for individual in kg.individuals:
            self.assertIn("birth_date", individual.attributes)
            self.assertIn("birth_city", individual.attributes)


if __name__ == "__main__":
    unittest.main()
