"""
Functions and utilities for generating name data.
"""

import random
import re
from typing import List, Optional, Union, Sequence


def generate_names(
    num_names: Optional[int] = None,
    fname_sharing: Optional[Union[int, List[int]]] = None,
    lname_sharing: Optional[Union[int, List[int]]] = None,
    fnames: Optional[Sequence[str]] = None,
    lnames: Optional[Sequence[str]] = None,
    use_templates: bool = False,
    subword_mode: bool = True,
    random_seed: Optional[int] = None
) -> List[str]:
    """
    Generate names based on sharing patterns, either from provided names or templates.

    Args:
        num_names: Optional total number of names to generate
        fname_sharing: Either:
            - int: number of people who should share each first name (1 means unique)
            - List[int]: specific sharing pattern for first names
            - None: no sharing pattern (uses each name once)
        lname_sharing: Either:
            - int: number of people who should share each last name (1 means unique)
            - List[int]: specific sharing pattern for last names
            - None: no sharing pattern (uses each name once)
        fnames: Optional sequence of first names to use
        lnames: Optional sequence of last names to use
        use_templates: If True, generates names using templates when needed
        subword_mode: If True, create names that break down into first/last name tokens
        random_seed: Optional random seed for reproducibility

    Returns:
        List of generated full names

    Raises:
        ValueError: If no names provided and templates not enabled or if sharing pattern is invalid
    """
    if random_seed is not None:
        random.seed(random_seed)

    def calculate_required_unique_names(sharing: Optional[Union[int, List[int]]], total: int) -> int:
        """Calculate how many unique names needed for the sharing pattern"""
        if sharing is None or sharing == 1:
            return total
        if isinstance(sharing, int):
            return (total + sharing - 1) // sharing  # Ceiling division
        return len(sharing)

    def generate_template_names(prefix: str, count: int) -> List[str]:
        """Generate names using template pattern"""
        return [f"{prefix}_{i}" for i in range(count)]

    # Calculate required unique names if num_names is specified
    if num_names is not None:
        if fname_sharing is not None:
            required_fnames = calculate_required_unique_names(fname_sharing, num_names)
        else:
            required_fnames = num_names

        if lname_sharing is not None:
            required_lnames = calculate_required_unique_names(lname_sharing, num_names)
        else:
            required_lnames = num_names

        # Generate template names if needed and enabled
        if use_templates:
            if not fnames:
                fnames = generate_template_names("fname", required_fnames)
            if not lnames:
                lnames = generate_template_names("lname", required_lnames)

    # Validate inputs
    if not fnames or not lnames:
        raise ValueError("Must provide name lists or enable templates")

    # Make copies to avoid modifying the input lists
    name_pool_first = list(fnames)
    name_pool_last = list(lnames)

    # Generate shared names using deterministic approach for reproducibility
    def generate_shared_names(names: List[str], sharing: Optional[Union[int, List[int]]],
                            target_count: Optional[int] = None) -> List[str]:
        """Generate a list of names with the specified sharing pattern"""
        if sharing is None:
            # No sharing - use each name once
            result = names.copy()
        elif isinstance(sharing, int):
            # Uniform sharing - each name appears 'sharing' times
            num_complete_groups = min(len(names), (target_count + sharing - 1) // sharing if target_count else len(names))
            # Use the first N names deterministically instead of random sampling for reproducibility
            selected_names = names[:num_complete_groups]
            result = []
            for name in selected_names:
                result.extend([name] * sharing)
        else:
            # Custom sharing pattern - follow the specified counts
            # Use the first N names deterministically where N is the length of sharing
            selected_names = names[:len(sharing)]
            result = []
            for name, count in zip(selected_names, sharing):
                result.extend([name] * count)

        # Truncate to target count if specified
        if target_count is not None:
            result = result[:target_count]
            
        return result

    # Generate the first and last names
    fnames_result = generate_shared_names(name_pool_first, fname_sharing, num_names)
    lnames_result = generate_shared_names(name_pool_last, lname_sharing, num_names)

    # Ensure we have the same number of first and last names
    min_len = min(len(fnames_result), len(lnames_result))
    if num_names is not None:
        min_len = min(min_len, num_names)

    fnames_result = fnames_result[:min_len]
    lnames_result = lnames_result[:min_len]

    # Combine first and last names based on subword mode
    if subword_mode:
        # In subword mode, we create camel case names for better subword tokenization
        return [f"{f}{l}" for f, l in zip(fnames_result, lnames_result)]
    else:
        # In non-subword mode, we use space-separated names
        return [f"{f} {l}" for f, l in zip(fnames_result, lnames_result)]


def add_random_names(filename: str, num_names: int, prefix: str = "F") -> None:
    """
    Add random names to a file.
    
    Args:
        filename: File to add names to
        num_names: Number of random names to add
        prefix: Prefix character for names (F for first names, L for last names)
    """
    with open(filename, "a") as f:
        for _ in range(num_names):
            if prefix == "F":
                random_chars = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=6))
            else:
                random_chars = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=8))
            f.write(f"{prefix}{random_chars}\n")


def read_names_from_file(file_path: str) -> List[str]:
    """
    Read names from a file, filtering out invalid entries.
    
    Args:
        file_path: Path to the file containing names (one per line)
        
    Returns:
        List of valid names
        
    Raises:
        FileNotFoundError: If the file does not exist
    """
    words = set()
    with open(file_path, 'r') as f:
        for line in f:
            # Remove non-alphanumeric characters
            cleaned_line = re.sub(r'[^a-zA-Z0-9]', '', line.strip())
            if cleaned_line:
                words.add(cleaned_line)
    return list(words)


def ensure_name_files(
    first_name_file: str, 
    last_name_file: str, 
    num_required: int
) -> tuple[List[str], List[str]]:
    """
    Ensure that name files exist and contain enough names.
    
    Args:
        first_name_file: Path to file containing first names
        last_name_file: Path to file containing last names
        num_required: Minimum number of names needed
        
    Returns:
        Tuple of (first_names, last_names) lists
    """
    # Check if files exist, create if not
    for file_path, prefix in [(first_name_file, "F"), (last_name_file, "L")]:
        try:
            with open(file_path, 'r') as f:
                num_names = sum(1 for _ in f)
                
            if num_names < num_required:
                add_random_names(file_path, num_required - num_names, prefix)
        except FileNotFoundError:
            # Create new file with random names
            with open(file_path, 'w') as f:
                pass  # Create empty file
            add_random_names(file_path, num_required, prefix)
    
    # Read files
    fnames = read_names_from_file(first_name_file)
    lnames = read_names_from_file(last_name_file)
    
    # Ensure first names and last names don't overlap
    fnames = list(set(fnames) - set(lnames))
    
    # Sort for deterministic order
    fnames.sort()
    lnames.sort()
    
    # Limit to required number
    fnames = fnames[:num_required]
    lnames = lnames[:num_required]
    
    # Capitalize for consistency
    fnames = [name.capitalize() for name in fnames]
    lnames = [name.capitalize() for name in lnames]
    
    return fnames, lnames
