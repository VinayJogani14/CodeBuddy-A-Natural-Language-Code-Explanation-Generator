"""
Evaluator: Comprehensive evaluation metrics for code explanation models
"""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np
from pathlib import Path
import json
import time
from datasets import Dataset
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config import EVALUATION_CONFIG, MODEL_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeExplainerEvaluator:
    """
    Comprehensive evaluator for code explanation models
    """
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the evaluator
        
        Args:
            model_path: Path to the trained model
            device: Device to use for evaluation
        """
        self.model_path = Path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.metrics = {}
        
        try:
            self._load_model()
            logger.info("Evaluator initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing evaluator: {str(e)}")
            raise
    
    def _load_model(self):
        """Load the trained model and tokenizer"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_path)
            
            # Load model
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def evaluate_explanations(self, 
                            test_dataset: Dataset,
                            code_column: str = 'code',
                            reference_column: str = 'explanation',
                            batch_size: int = 8) -> Dict[str, Any]:
        """
        Evaluate model performance on test dataset
        
        Args:
            test_dataset: Test dataset
            code_column: Column name containing code
            reference_column: Column name containing reference explanations
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary containing evaluation results
        """
        try:
            logger.info("Starting evaluation...")
            
            # Generate predictions
            predictions = self._generate_predictions(
                test_dataset, 
                code_column, 
                batch_size
            )
            
            # Get references
            references = test_dataset[reference_column]
            
            # Calculate metrics
            metrics = self._calculate_metrics(predictions, references)
            
            # Add metadata
            metrics['dataset_size'] = len(test_dataset)
            metrics['model_path'] = str(self.model_path)
            metrics['device'] = self.device
            
            self.metrics = metrics
            
            logger.info("Evaluation completed successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise
    
    def _generate_predictions(self, 
                             dataset: Dataset, 
                             code_column: str, 
                             batch_size: int) -> List[str]:
        """Generate predictions for the dataset"""
        predictions = []
        
        try:
            for i in range(0, len(dataset), batch_size):
                batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
                
                # Prepare inputs
                inputs = [f"Explain this Python code: {code}" for code in batch[code_column]]
                
                # Tokenize inputs
                encoded_inputs = self.tokenizer(
                    inputs,
                    return_tensors="pt",
                    max_length=MODEL_CONFIG["code_t5_max_length"],
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                # Generate predictions
                with torch.no_grad():
                    outputs = self.model.generate(
                        **encoded_inputs,
                        max_length=MODEL_CONFIG["max_explanation_length"],
                        temperature=MODEL_CONFIG["temperature"],
                        top_p=MODEL_CONFIG["top_p"],
                        top_k=MODEL_CONFIG["top_k"],
                        num_beams=MODEL_CONFIG["num_beams"],
                        early_stopping=MODEL_CONFIG["early_stopping"],
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode outputs
                batch_predictions = [
                    self.tokenizer.decode(output, skip_special_tokens=True).strip()
                    for output in outputs
                ]
                
                predictions.extend(batch_predictions)
                
                # Log progress
                if (i + batch_size) % (batch_size * 10) == 0:
                    logger.info(f"Generated predictions for {i + batch_size}/{len(dataset)} samples")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise
    
    def _calculate_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        try:
            metrics = {}
            
            # BLEU Score
            metrics['bleu'] = self._calculate_bleu(predictions, references)
            
            # BERTScore
            metrics['bertscore'] = self._calculate_bertscore(predictions, references)
            
            # Exact Match
            metrics['exact_match'] = self._calculate_exact_match(predictions, references)
            
            # Length metrics
            metrics['avg_prediction_length'] = np.mean([len(pred) for pred in predictions])
            metrics['avg_reference_length'] = np.mean([len(ref) for ref in references])
            
            # Length ratio
            metrics['length_ratio'] = metrics['avg_prediction_length'] / metrics['avg_reference_length']
            
            # Coverage metrics
            metrics['coverage'] = self._calculate_coverage(predictions, references)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise
    
    def _calculate_bleu(self, predictions: List[str], references: List[str]) -> float:
        """Calculate BLEU score"""
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            
            # Download required NLTK data
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            
            smoothing = SmoothingFunction().method1
            
            bleu_scores = []
            for pred, ref in zip(predictions, references):
                # Tokenize
                pred_tokens = pred.split()
                ref_tokens = ref.split()
                
                # Calculate BLEU
                score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
                bleu_scores.append(score)
            
            return np.mean(bleu_scores)
            
        except Exception as e:
            logger.warning(f"Could not calculate BLEU score: {str(e)}")
            return 0.0
    
    def _calculate_bertscore(self, predictions: List[str], references: List[str]) -> float:
        """Calculate BERTScore"""
        try:
            from bert_score import score
            
            # Calculate BERTScore
            P, R, F1 = score(predictions, references, lang='en', verbose=True)
            
            # Return F1 score (harmonic mean of precision and recall)
            return F1.mean().item()
            
        except Exception as e:
            logger.warning(f"Could not calculate BERTScore: {str(e)}")
            return 0.0
    
    def _calculate_exact_match(self, predictions: List[str], references: List[str]) -> float:
        """Calculate exact match accuracy"""
        try:
            exact_matches = sum(1 for pred, ref in zip(predictions, references) 
                              if pred.strip().lower() == ref.strip().lower())
            return exact_matches / len(predictions)
            
        except Exception as e:
            logger.warning(f"Could not calculate exact match: {str(e)}")
            return 0.0
    
    def _calculate_coverage(self, predictions: List[str], references: List[str]) -> float:
        """Calculate coverage of reference content in predictions"""
        try:
            coverage_scores = []
            
            for pred, ref in zip(predictions, references):
                pred_words = set(pred.lower().split())
                ref_words = set(ref.lower().split())
                
                if len(ref_words) > 0:
                    coverage = len(pred_words.intersection(ref_words)) / len(ref_words)
                    coverage_scores.append(coverage)
            
            return np.mean(coverage_scores) if coverage_scores else 0.0
            
        except Exception as e:
            logger.warning(f"Could not calculate coverage: {str(e)}")
            return 0.0
    
    def evaluate_response_time(self, test_codes: List[str], num_runs: int = 10) -> Dict[str, float]:
        """
        Evaluate model response time
        
        Args:
            test_codes: List of test code snippets
            num_runs: Number of runs for averaging
            
        Returns:
            Dictionary containing timing metrics
        """
        try:
            logger.info("Evaluating response time...")
            
            response_times = []
            
            for _ in range(num_runs):
                start_time = time.time()
                
                # Generate explanation for each test code
                for code in test_codes:
                    input_text = f"Explain this Python code: {code}"
                    
                    encoded_input = self.tokenizer(
                        input_text,
                        return_tensors="pt",
                        max_length=MODEL_CONFIG["code_t5_max_length"],
                        truncation=True,
                        padding=True
                    ).to(self.device)
                    
                    with torch.no_grad():
                        self.model.generate(
                            **encoded_input,
                            max_length=MODEL_CONFIG["max_explanation_length"],
                            temperature=MODEL_CONFIG["temperature"],
                            top_p=MODEL_CONFIG["top_p"],
                            top_k=MODEL_CONFIG["top_k"],
                            num_beams=MODEL_CONFIG["num_beams"],
                            early_stopping=MODEL_CONFIG["early_stopping"],
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                
                end_time = time.time()
                response_times.append(end_time - start_time)
            
            # Calculate timing metrics
            avg_time = np.mean(response_times)
            std_time = np.std(response_times)
            min_time = np.min(response_times)
            max_time = np.max(response_times)
            
            timing_metrics = {
                'avg_response_time': avg_time,
                'std_response_time': std_time,
                'min_response_time': min_time,
                'max_response_time': max_time,
                'throughput': len(test_codes) / avg_time  # codes per second
            }
            
            logger.info(f"Response time evaluation completed: {timing_metrics}")
            return timing_metrics
            
        except Exception as e:
            logger.error(f"Error evaluating response time: {str(e)}")
            raise
    
    def compare_with_baseline(self, 
                             predictions: List[str], 
                             references: List[str],
                             baseline_predictions: List[str]) -> Dict[str, Any]:
        """
        Compare model performance with baseline
        
        Args:
            predictions: Model predictions
            references: Reference explanations
            baseline_predictions: Baseline model predictions
            
        Returns:
            Dictionary containing comparison metrics
        """
        try:
            logger.info("Comparing with baseline...")
            
            # Calculate metrics for both models
            model_metrics = self._calculate_metrics(predictions, references)
            baseline_metrics = self._calculate_metrics(baseline_predictions, references)
            
            # Calculate improvements
            improvements = {}
            for metric in model_metrics:
                if metric in baseline_metrics and baseline_metrics[metric] != 0:
                    improvement = ((model_metrics[metric] - baseline_metrics[metric]) / 
                                 baseline_metrics[metric]) * 100
                    improvements[f"{metric}_improvement"] = improvement
            
            comparison = {
                'model_metrics': model_metrics,
                'baseline_metrics': baseline_metrics,
                'improvements': improvements
            }
            
            logger.info("Baseline comparison completed")
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing with baseline: {str(e)}")
            raise
    
    def save_results(self, output_path: str):
        """Save evaluation results to file"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise
    
    def generate_report(self, output_path: str):
        """Generate comprehensive evaluation report"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            report = f"""
# CodeBuddy Model Evaluation Report

## Model Information
- Model Path: {self.model_path}
- Device: {self.device}
- Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Evaluation Metrics

### Quality Metrics
- BLEU Score: {self.metrics.get('bleu', 'N/A'):.4f}
- BERTScore: {self.metrics.get('bertscore', 'N/A'):.4f}
- Exact Match: {self.metrics.get('exact_match', 'N/A'):.4f}

### Length Metrics
- Average Prediction Length: {self.metrics.get('avg_prediction_length', 'N/A'):.2f}
- Average Reference Length: {self.metrics.get('avg_reference_length', 'N/A'):.2f}
- Length Ratio: {self.metrics.get('length_ratio', 'N/A'):.2f}

### Coverage Metrics
- Content Coverage: {self.metrics.get('coverage', 'N/A'):.4f}

## Summary
The model achieved a BLEU score of {self.metrics.get('bleu', 'N/A'):.4f} and a BERTScore of {self.metrics.get('bertscore', 'N/A'):.4f}, 
indicating {'good' if self.metrics.get('bleu', 0) > 0.3 else 'moderate' if self.metrics.get('bleu', 0) > 0.1 else 'poor'} 
performance in generating code explanations.
"""
            
            with open(output_path, 'w') as f:
                f.write(report)
            
            logger.info(f"Report generated and saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise

def main():
    """Main function for evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate CodeT5 model for code explanation")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model")
    parser.add_argument("--test_dataset", type=str, required=True,
                       help="Path to test dataset")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = CodeExplainerEvaluator(args.model_path)
    
    # Load test dataset
    from data.data_processor import DataProcessor
    data_processor = DataProcessor()
    test_dataset = data_processor.load_processed_dataset(args.test_dataset)
    
    # Run evaluation
    metrics = evaluator.evaluate_explanations(test_dataset)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    evaluator.save_results(output_dir / "evaluation_results.json")
    evaluator.generate_report(output_dir / "evaluation_report.md")
    
    print(f"Evaluation completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
