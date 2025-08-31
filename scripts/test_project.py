#!/usr/bin/env python3
"""
Test Script for CodeBuddy Project

This script tests all major components to ensure they work correctly:
- Model loading
- Code processing
- RAG pipeline
- Data processing
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

def test_imports():
    """Test that all required modules can be imported"""
    try:
        logger.info("Testing imports...")
        
        # Test core imports
        from models.code_explainer import CodeExplainer
        from models.rag_pipeline import RAGPipeline
        from utils.code_processor import CodeProcessor
        from data.data_processor import DataProcessor
        from training.train_code_explainer import CodeExplainerTrainer
        from evaluation.evaluator import CodeExplainerEvaluator
        
        logger.info("✅ All imports successful")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {str(e)}")
        return False

def test_code_processor():
    """Test the code processor utility"""
    try:
        logger.info("Testing CodeProcessor...")
        
        from utils.code_processor import CodeProcessor
        
        processor = CodeProcessor()
        
        # Test code processing
        test_code = "def hello():\n    print('Hello, World!')"
        processed = processor.process_code(test_code)
        
        # Test code analysis
        analysis = processor.analyze_code(test_code)
        
        logger.info("✅ CodeProcessor test successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ CodeProcessor test failed: {str(e)}")
        return False

def test_rag_pipeline():
    """Test the RAG pipeline"""
    try:
        logger.info("Testing RAG Pipeline...")
        
        from models.rag_pipeline import RAGPipeline
        
        # Initialize RAG pipeline
        rag = RAGPipeline()
        
        # Test adding code snippets
        test_snippets = [
            "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
            "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
        ]
        test_explanations = [
            "Calculate factorial using recursion",
            "Calculate Fibonacci number using recursion"
        ]
        
        rag.add_code_snippets(test_snippets, test_explanations)
        
        # Test search
        results = rag.search_similar_code("def fact(n): return 1 if n <= 1 else n * fact(n-1)")
        
        logger.info("✅ RAG Pipeline test successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ RAG Pipeline test failed: {str(e)}")
        return False

def test_code_explainer():
    """Test the code explainer model"""
    try:
        logger.info("Testing CodeExplainer...")
        
        from models.code_explainer import CodeExplainer
        
        # Initialize explainer
        explainer = CodeExplainer()
        
        # Test code explanation
        test_code = "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr"
        
        explanation = explainer.explain_code(test_code)
        
        logger.info(f"Generated explanation: {explanation[:100]}...")
        logger.info("✅ CodeExplainer test successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ CodeExplainer test failed: {str(e)}")
        return False

def test_data_processor():
    """Test the data processor"""
    try:
        logger.info("Testing DataProcessor...")
        
        from data.data_processor import DataProcessor
        
        processor = DataProcessor()
        
        # Test creating sample data
        sample_data = [
            {"code": "def test(): pass", "docstring": "Test function"},
            {"code": "def hello(): print('Hello')", "docstring": "Hello function"}
        ]
        
        from datasets import Dataset
        dataset = Dataset.from_list(sample_data)
        
        # Test preprocessing
        processed = processor.preprocess_for_training(dataset)
        
        logger.info("✅ DataProcessor test successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ DataProcessor test failed: {str(e)}")
        return False

def test_config():
    """Test configuration loading"""
    try:
        logger.info("Testing configuration...")
        
        from config import MODEL_CONFIG, DATA_CONFIG, TRAINING_CONFIG
        
        # Check that configs are loaded
        assert isinstance(MODEL_CONFIG, dict)
        assert isinstance(DATA_CONFIG, dict)
        assert isinstance(TRAINING_CONFIG, dict)
        
        logger.info("✅ Configuration test successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {str(e)}")
        return False

def run_all_tests():
    """Run all tests"""
    logger.info("🚀 Starting CodeBuddy project tests...")
    
    tests = [
        ("Configuration", test_config),
        ("Imports", test_imports),
        ("Code Processor", test_code_processor),
        ("Data Processor", test_data_processor),
        ("RAG Pipeline", test_rag_pipeline),
        ("Code Explainer", test_code_explainer),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} test...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("📊 Test Results Summary:")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! CodeBuddy project is ready to use.")
    else:
        logger.warning("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

def main():
    """Main function"""
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during testing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
