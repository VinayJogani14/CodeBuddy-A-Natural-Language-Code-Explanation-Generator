#!/usr/bin/env python3
"""
Setup Script for CodeBuddy

This script helps set up the CodeBuddy project environment:
- Install dependencies
- Create necessary directories
- Download sample data
- Test the installation
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    try:
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            logger.error("❌ Python 3.8 or higher is required")
            return False
        
        logger.info(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error checking Python version: {str(e)}")
        return False

def install_dependencies():
    """Install required dependencies"""
    try:
        logger.info("📦 Installing dependencies...")
        
        # Get the requirements.txt path
        project_root = Path(__file__).parent.parent
        requirements_path = project_root / "requirements.txt"
        
        if not requirements_path.exists():
            logger.error("❌ requirements.txt not found")
            return False
        
        # Install dependencies
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_path)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Dependencies installed successfully")
            return True
        else:
            logger.error(f"❌ Failed to install dependencies: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error installing dependencies: {str(e)}")
        return False

def create_directories():
    """Create necessary project directories"""
    try:
        logger.info("📁 Creating project directories...")
        
        project_root = Path(__file__).parent.parent
        directories = [
            "data/datasets",
            "models/saved",
            "logs",
            "evaluation_results",
            "training_outputs"
        ]
        
        for directory in directories:
            dir_path = project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"  Created: {directory}")
        
        logger.info("✅ Directories created successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating directories: {str(e)}")
        return False

def download_sample_data():
    """Download sample data for testing"""
    try:
        logger.info("📥 Downloading sample data...")
        
        # Import and run the dataset downloader
        project_root = Path(__file__).parent.parent
        sys.path.append(str(project_root))
        
        from scripts.download_datasets import DatasetDownloader
        
        downloader = DatasetDownloader()
        downloader.create_sample_data()
        
        logger.info("✅ Sample data downloaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error downloading sample data: {str(e)}")
        return False

def test_installation():
    """Test if the installation works correctly"""
    try:
        logger.info("🧪 Testing installation...")
        
        # Test imports
        project_root = Path(__file__).parent.parent
        sys.path.append(str(project_root))
        
        try:
            from models.code_explainer import CodeExplainer
            from models.rag_pipeline import RAGPipeline
            from utils.code_processor import CodeProcessor
            logger.info("✅ Core modules imported successfully")
        except ImportError as e:
            logger.error(f"❌ Import failed: {str(e)}")
            return False
        
        # Test basic functionality
        try:
            processor = CodeProcessor()
            test_code = "def hello(): print('Hello')"
            processed = processor.process_code(test_code)
            logger.info("✅ Basic functionality test passed")
        except Exception as e:
            logger.error(f"❌ Basic functionality test failed: {str(e)}")
            return False
        
        logger.info("✅ Installation test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing installation: {str(e)}")
        return False

def run_setup():
    """Run the complete setup process"""
    logger.info("🚀 Starting CodeBuddy Setup...")
    logger.info("=" * 50)
    
    setup_steps = [
        ("Python Version Check", check_python_version),
        ("Install Dependencies", install_dependencies),
        ("Create Directories", create_directories),
        ("Download Sample Data", download_sample_data),
        ("Test Installation", test_installation),
    ]
    
    results = []
    
    for step_name, step_func in setup_steps:
        logger.info(f"\n{'='*50}")
        logger.info(f"🔧 Running: {step_name}")
        logger.info(f"{'='*50}")
        
        try:
            success = step_func()
            results.append((step_name, success))
            
            if not success:
                logger.error(f"❌ {step_name} failed. Setup cannot continue.")
                break
                
        except Exception as e:
            logger.error(f"❌ {step_name} crashed: {str(e)}")
            results.append((step_name, False))
            break
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("📊 Setup Results Summary:")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for step_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"  {step_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} steps completed")
    
    if passed == total:
        logger.info("🎉 Setup completed successfully!")
        logger.info("\n🚀 Next steps:")
        logger.info("  1. Run the demo: python scripts/demo.py")
        logger.info("  2. Test the project: python scripts/test_project.py")
        logger.info("  3. Start the Streamlit app: streamlit run app.py")
        logger.info("  4. Download datasets: python scripts/download_datasets.py")
    else:
        logger.error("⚠️  Setup failed. Please check the errors above and try again.")
    
    return passed == total

def main():
    """Main function"""
    try:
        success = run_setup()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during setup: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
