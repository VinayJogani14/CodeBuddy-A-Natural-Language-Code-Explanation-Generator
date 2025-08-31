#!/usr/bin/env python3
"""
Demo Script for CodeBuddy

This script demonstrates the main features of CodeBuddy:
- Code explanation
- Code Q&A
- Code search
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_code_explanation():
    """Demonstrate code explanation functionality"""
    try:
        logger.info("🔍 Demonstrating Code Explanation...")
        
        from models.code_explainer import CodeExplainer
        
        explainer = CodeExplainer()
        
        # Example code snippets
        examples = [
            {
                "name": "Bubble Sort",
                "code": """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr"""
            },
            {
                "name": "Binary Search",
                "code": """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1"""
            },
            {
                "name": "Fibonacci",
                "code": """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)"""
            }
        ]
        
        for example in examples:
            logger.info(f"\n📝 {example['name']}:")
            logger.info(f"Code:\n{example['code']}")
            
            explanation = explainer.explain_code(example['code'])
            logger.info(f"Explanation: {explanation}")
            logger.info("-" * 50)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Code explanation demo failed: {str(e)}")
        return False

def demo_rag_pipeline():
    """Demonstrate RAG pipeline functionality"""
    try:
        logger.info("🔍 Demonstrating RAG Pipeline...")
        
        from models.rag_pipeline import RAGPipeline
        
        rag = RAGPipeline()
        
        # Add some example code snippets
        code_snippets = [
            "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
            "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
            "def is_prime(n): return n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1))",
            "def quick_sort(arr): return arr if len(arr) <= 1 else quick_sort([x for x in arr[1:] if x <= arr[0]]) + [arr[0]] + quick_sort([x for x in arr[1:] if x > arr[0]])"
        ]
        
        explanations = [
            "Calculate factorial using recursion",
            "Calculate Fibonacci number using recursion",
            "Check if a number is prime",
            "Sort array using quicksort algorithm"
        ]
        
        rag.add_code_snippets(code_snippets, explanations)
        
        # Test search functionality
        logger.info("Testing code search...")
        
        search_queries = [
            "factorial function",
            "recursive algorithms",
            "sorting algorithms"
        ]
        
        for query in search_queries:
            logger.info(f"\n🔍 Searching for: '{query}'")
            results = rag.search_by_query(query, top_k=2)
            
            for i, result in enumerate(results):
                logger.info(f"  Result {i+1}: {result['code'][:50]}...")
                logger.info(f"  Explanation: {result['explanation']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ RAG pipeline demo failed: {str(e)}")
        return False

def demo_code_processing():
    """Demonstrate code processing functionality"""
    try:
        logger.info("🔍 Demonstrating Code Processing...")
        
        from utils.code_processor import CodeProcessor
        
        processor = CodeProcessor()
        
        # Example code
        test_code = """import math

def calculate_area(radius):
    \"\"\"Calculate the area of a circle\"\"\"
    return math.pi * radius ** 2

class Circle:
    def __init__(self, radius):
        self.radius = radius
    
    def get_area(self):
        return calculate_area(self.radius)
    
    def get_circumference(self):
        return 2 * math.pi * self.radius"""
        
        logger.info(f"Original code:\n{test_code}")
        
        # Process code
        processed_code = processor.process_code(test_code)
        logger.info(f"\nProcessed code:\n{processed_code}")
        
        # Analyze code
        analysis = processor.analyze_code(test_code)
        logger.info(f"\nCode analysis:")
        for key, value in analysis.items():
            logger.info(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Code processing demo failed: {str(e)}")
        return False

def demo_data_processing():
    """Demonstrate data processing functionality"""
    try:
        logger.info("🔍 Demonstrating Data Processing...")
        
        from data.data_processor import DataProcessor
        
        processor = DataProcessor()
        
        # Create sample dataset
        sample_data = [
            {"code": "def hello(): print('Hello')", "docstring": "Print hello message"},
            {"code": "def add(a, b): return a + b", "docstring": "Add two numbers"},
            {"code": "def multiply(a, b): return a * b", "docstring": "Multiply two numbers"}
        ]
        
        from datasets import Dataset
        dataset = Dataset.from_list(sample_data)
        
        logger.info(f"Original dataset: {len(dataset)} samples")
        logger.info(f"Columns: {dataset.column_names}")
        
        # Preprocess for training
        processed_dataset = processor.preprocess_for_training(dataset)
        
        logger.info(f"Processed dataset: {len(processed_dataset)} samples")
        logger.info(f"New columns: {processed_dataset.column_names}")
        
        # Show sample
        sample = processed_dataset[0]
        logger.info(f"Sample processed data:")
        logger.info(f"  Code: {sample['code']}")
        logger.info(f"  Explanation: {sample['explanation']}")
        logger.info(f"  Code length: {sample['code_length']}")
        logger.info(f"  Explanation length: {sample['explanation_length']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data processing demo failed: {str(e)}")
        return False

def run_demo():
    """Run the complete demo"""
    logger.info("🚀 Starting CodeBuddy Demo...")
    logger.info("=" * 60)
    
    demos = [
        ("Code Processing", demo_code_processing),
        ("Data Processing", demo_data_processing),
        ("Code Explanation", demo_code_explanation),
        ("RAG Pipeline", demo_rag_pipeline),
    ]
    
    results = []
    
    for demo_name, demo_func in demos:
        logger.info(f"\n{'='*60}")
        logger.info(f"🎬 Running {demo_name} Demo")
        logger.info(f"{'='*60}")
        
        try:
            success = demo_func()
            results.append((demo_name, success))
        except Exception as e:
            logger.error(f"Demo {demo_name} crashed: {str(e)}")
            results.append((demo_name, False))
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("📊 Demo Results Summary:")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for demo_name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"  {demo_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} demos successful")
    
    if passed == total:
        logger.info("🎉 All demos completed successfully!")
        logger.info("CodeBuddy is ready to use!")
    else:
        logger.warning("⚠️  Some demos failed. Please check the errors above.")
    
    return passed == total

def main():
    """Main function"""
    try:
        success = run_demo()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during demo: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
