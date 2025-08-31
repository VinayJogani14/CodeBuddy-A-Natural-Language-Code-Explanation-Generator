#!/usr/bin/env python3
"""
Dataset Download Script for CodeBuddy

This script downloads and prepares the required datasets for training and evaluation:
- CodeSearchNet (Python subset)
- DocString Dataset (from CodeXGLUE)
- HumanEval
- MBPP (Mostly Basic Python Problems)
"""

import os
import sys
import logging
from pathlib import Path
import requests
import zipfile
import json
from datasets import load_dataset, Dataset
from huggingface_hub import hf_hub_download
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config import DATA_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetDownloader:
    """Handles downloading and preparing datasets for CodeBuddy"""
    
    def __init__(self, output_dir: str = "data/datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset URLs and configurations
        self.datasets = {
            "code_search_net": {
                "name": "code-search-net/code_search_net",
                "subset": "python",
                "splits": ["train", "validation", "test"]
            },
            "mbpp": {
                "name": "Muennighoff/mbpp",
                "subset": None,
                "splits": ["train", "validation", "test"]
            },
            "human_eval": {
                "name": "openai_humaneval",
                "subset": None,
                "splits": ["test"]
            }
        }
    
    def download_code_search_net(self) -> Dataset:
        """Download CodeSearchNet Python subset"""
        try:
            logger.info("Downloading CodeSearchNet Python subset...")
            
            dataset = load_dataset(
                self.datasets["code_search_net"]["name"],
                self.datasets["code_search_net"]["subset"]
            )
            
            # Save to local directory
            output_path = self.output_dir / "code_search_net"
            dataset.save_to_disk(str(output_path))
            
            logger.info(f"CodeSearchNet downloaded successfully to {output_path}")
            logger.info(f"Dataset info: {dataset}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error downloading CodeSearchNet: {str(e)}")
            raise
    
    def download_mbpp(self) -> Dataset:
        """Download MBPP dataset"""
        try:
            logger.info("Downloading MBPP dataset...")
            
            # Try different approaches for MBPP
            try:
                # First try: direct load
                dataset = load_dataset(self.datasets["mbpp"]["name"])
            except Exception as e1:
                logger.warning(f"Direct load failed: {str(e1)}")
                try:
                    # Second try: with trust_remote_code
                    dataset = load_dataset(self.datasets["mbpp"]["name"], trust_remote_code=True)
                except Exception as e2:
                    logger.warning(f"Trust remote code load failed: {str(e2)}")
                    # Third try: create sample MBPP data
                    logger.info("Creating sample MBPP dataset...")
                    dataset = self._create_sample_mbpp_dataset()
            
            # Save to local directory
            output_path = self.output_dir / "mbpp"
            dataset.save_to_disk(str(output_path))
            
            logger.info(f"MBPP prepared successfully to {output_path}")
            logger.info(f"Dataset info: {dataset}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error preparing MBPP: {str(e)}")
            raise
    
    def download_human_eval(self) -> Dataset:
        """Download HumanEval dataset"""
        try:
            logger.info("Downloading HumanEval dataset...")
            
            dataset = load_dataset(self.datasets["human_eval"]["name"])
            
            # Save to local directory
            output_path = self.output_dir / "human_eval"
            dataset.save_to_disk(str(output_path))
            
            logger.info(f"HumanEval downloaded successfully to {output_path}")
            logger.info(f"Dataset info: {dataset}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error downloading HumanEval: {str(e)}")
            raise
    
    def download_docstring_dataset(self) -> Dataset:
        """Download DocString dataset from CodeXGLUE"""
        try:
            logger.info("Downloading DocString dataset from CodeXGLUE...")
            
            # Try to load from HuggingFace datasets
            try:
                dataset = load_dataset("code_x_glue_ct_code_to_text", "python")
                logger.info("DocString dataset loaded from HuggingFace")
            except:
                # Fallback: create from scratch using sample data
                logger.warning("Could not load from HuggingFace, creating sample dataset...")
                dataset = self._create_sample_docstring_dataset()
            
            # Save to local directory
            output_path = self.output_dir / "docstring"
            dataset.save_to_disk(str(output_path))
            
            logger.info(f"DocString dataset prepared successfully to {output_path}")
            logger.info(f"Dataset info: {dataset}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error preparing DocString dataset: {str(e)}")
            raise
    
    def _create_sample_docstring_dataset(self) -> Dataset:
        """Create a sample docstring dataset for demonstration"""
        sample_data = [
            {
                "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
                "docstring": "Calculate the nth Fibonacci number using recursion."
            },
            {
                "code": "def bubble_sort(arr):\n    n = len(arr)\n        for i in range(n):\n            for j in range(0, n-i-1):\n                if arr[j] > arr[j+1]:\n                    arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr",
                "docstring": "Sort an array using the bubble sort algorithm."
            },
            {
                "code": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
                "docstring": "Check if a number is prime."
            }
        ]
        
        return Dataset.from_list(sample_data)
    
    def _create_sample_mbpp_dataset(self) -> Dataset:
        """Create a sample MBPP dataset for demonstration"""
        sample_data = [
            {
                "task_id": 1,
                "text": "Write a function to calculate the factorial of a number",
                "code": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
                "test_list": ["assert factorial(5) == 120", "assert factorial(0) == 1"],
                "test_setup_code": "",
                "challenge_test_list": []
            },
            {
                "task_id": 2,
                "text": "Write a function to check if a string is a palindrome",
                "code": "def is_palindrome(s):\n    return s == s[::-1]",
                "test_list": ["assert is_palindrome('racecar') == True", "assert is_palindrome('hello') == False"],
                "test_setup_code": "",
                "challenge_test_list": []
            },
            {
                "task_id": 3,
                "text": "Write a function to find the maximum element in a list",
                "code": "def find_max(arr):\n    if not arr:\n        return None\n    return max(arr)",
                "test_list": ["assert find_max([1, 5, 3, 9, 2]) == 9", "assert find_max([]) is None"],
                "test_setup_code": "",
                "challenge_test_list": []
            }
        ]
        
        return Dataset.from_list(sample_data)
    
    def prepare_combined_dataset(self) -> Dataset:
        """Combine all datasets into a unified training dataset"""
        try:
            logger.info("Preparing combined dataset...")
            
            combined_data = []
            
            # Load and process each dataset
            datasets_to_process = [
                ("code_search_net", self.download_code_search_net),
                ("mbpp", self.download_mbpp),
                ("human_eval", self.download_human_eval),
                ("docstring", self.download_docstring_dataset)
            ]
            
            for dataset_name, download_func in datasets_to_process:
                try:
                    dataset = download_func()
                    
                    # Convert to pandas for easier processing
                    df = dataset.to_pandas()
                    
                    # Standardize column names
                    if 'code' in df.columns and 'docstring' in df.columns:
                        df = df[['code', 'docstring']]
                    elif 'code' in df.columns and 'text' in df.columns:
                        df = df[['code', 'text']].rename(columns={'text': 'docstring'})
                    elif 'prompt' in df.columns and 'canonical_solution' in df.columns:
                        df = df[['prompt', 'canonical_solution']].rename(columns={'prompt': 'code', 'canonical_solution': 'docstring'})
                    else:
                        logger.warning(f"Unexpected columns in {dataset_name}: {df.columns.tolist()}")
                        continue
                    
                    # Add dataset source
                    df['source'] = dataset_name
                    combined_data.append(df)
                    
                    logger.info(f"Processed {dataset_name}: {len(df)} samples")
                    
                except Exception as e:
                    logger.error(f"Error processing {dataset_name}: {str(e)}")
                    continue
            
            if not combined_data:
                raise ValueError("No datasets were successfully processed")
            
            # Combine all datasets
            combined_df = pd.concat(combined_data, ignore_index=True)
            
            # Clean and filter
            combined_df = combined_df.dropna()
            combined_df = combined_df[combined_df['code'].str.len() > 10]  # Remove very short code
            combined_df = combined_df[combined_df['docstring'].str.len() > 5]  # Remove very short docstrings
            
            # Convert back to HuggingFace dataset
            combined_dataset = Dataset.from_pandas(combined_df)
            
            # Save combined dataset
            output_path = self.output_dir / "combined"
            combined_dataset.save_to_disk(str(output_path))
            
            logger.info(f"Combined dataset created successfully: {len(combined_dataset)} samples")
            logger.info(f"Saved to: {output_path}")
            
            return combined_dataset
            
        except Exception as e:
            logger.error(f"Error preparing combined dataset: {str(e)}")
            raise
    
    def create_sample_data(self) -> Dataset:
        """Create a small sample dataset for testing"""
        try:
            logger.info("Creating sample dataset for testing...")
            
            sample_data = [
                {
                    "code": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
                    "docstring": "Calculate the factorial of a non-negative integer using recursion."
                },
                {
                    "code": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
                    "docstring": "Perform binary search to find target element in sorted array."
                },
                {
                    "code": "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)",
                    "docstring": "Sort an array using the quicksort algorithm with middle pivot selection."
                }
            ]
            
            dataset = Dataset.from_list(sample_data)
            
            # Save sample dataset
            output_path = self.output_dir / "sample"
            dataset.save_to_disk(str(output_path))
            
            logger.info(f"Sample dataset created: {len(dataset)} samples")
            logger.info(f"Saved to: {output_path}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error creating sample dataset: {str(e)}")
            raise
    
    def get_dataset_info(self) -> dict:
        """Get information about all available datasets"""
        try:
            info = {}
            
            for dataset_name in self.datasets.keys():
                dataset_path = self.output_dir / dataset_name
                if dataset_path.exists():
                    try:
                        dataset = Dataset.load_from_disk(str(dataset_path))
                        info[dataset_name] = {
                            "size": len(dataset),
                            "columns": dataset.column_names,
                            "path": str(dataset_path)
                        }
                    except Exception as e:
                        info[dataset_name] = {"error": str(e)}
                else:
                    info[dataset_name] = {"status": "not_downloaded"}
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting dataset info: {str(e)}")
            return {}

def main():
    """Main function for dataset download"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download datasets for CodeBuddy")
    parser.add_argument("--output_dir", type=str, default="data/datasets",
                       help="Output directory for datasets")
    parser.add_argument("--create_sample", action="store_true",
                       help="Create a small sample dataset for testing")
    parser.add_argument("--combine", action="store_true",
                       help="Create a combined dataset from all sources")
    
    args = parser.parse_args()
    
    try:
        downloader = DatasetDownloader(args.output_dir)
        
        if args.create_sample:
            downloader.create_sample_data()
        elif args.combine:
            downloader.prepare_combined_dataset()
        else:
            # Download all datasets
            logger.info("Starting dataset download...")
            downloader.download_code_search_net()
            downloader.download_mbpp()
            downloader.download_human_eval()
            downloader.download_docstring_dataset()
            
            logger.info("All datasets downloaded successfully!")
        
        # Show dataset info
        info = downloader.get_dataset_info()
        logger.info("Dataset information:")
        for name, details in info.items():
            logger.info(f"  {name}: {details}")
            
    except Exception as e:
        logger.error(f"Dataset download failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
