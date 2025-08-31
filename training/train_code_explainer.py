"""
Training script for CodeT5 model fine-tuning on code explanation tasks
"""

import torch
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset, DatasetDict
import logging
import os
from pathlib import Path
import wandb
from typing import Dict, Any, Optional, List
import sys
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data.data_processor import DataProcessor
from config import MODEL_CONFIG, TRAINING_CONFIG, DATASET_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeExplainerTrainer:
    """
    Trainer class for fine-tuning CodeT5 on code explanation tasks
    """
    
    def __init__(self, 
                 model_name: str = "Salesforce/codet5-base",
                 output_dir: str = "models/code_explainer_finetuned",
                 use_wandb: bool = True):
        """
        Initialize the trainer
        
        Args:
            model_name: Pre-trained model name
            output_dir: Output directory for saving models
            use_wandb: Whether to use Weights & Biases logging
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.data_processor = DataProcessor()
        
        # Initialize wandb if requested
        if self.use_wandb:
            self._init_wandb()
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging"""
        try:
            wandb.init(
                project="codebuddy-training",
                name="code-explainer-finetuning",
                config={
                    "model_name": self.model_name,
                    "training_config": TRAINING_CONFIG,
                    "model_config": MODEL_CONFIG
                }
            )
            logger.info("Weights & Biases initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize Weights & Biases: {str(e)}")
            self.use_wandb = False
    
    def load_model_and_tokenizer(self):
        """Load the pre-trained model and tokenizer"""
        try:
            logger.info(f"Loading model and tokenizer: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            
            # Load model
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self.model.to(self.device)
            
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def prepare_dataset(self, dataset_path: Optional[str] = None) -> DatasetDict:
        """
        Prepare dataset for training
        
        Args:
            dataset_path: Path to pre-processed dataset (optional)
            
        Returns:
            DatasetDict with train/validation/test splits
        """
        try:
            if dataset_path and Path(dataset_path).exists():
                logger.info(f"Loading pre-processed dataset from {dataset_path}")
                dataset = self.data_processor.load_processed_dataset(dataset_path)
            else:
                logger.info("Loading and processing datasets...")
                
                # Load CodeSearchNet Python dataset
                code_search_dataset = self.data_processor.load_code_search_net(
                    subset='python',
                    split='train',
                    max_samples=DATASET_CONFIG["code_search_net"]["max_samples"]
                )
                
                # Load DocString dataset
                docstring_dataset = self.data_processor.load_docstring_dataset(
                    split='train',
                    max_samples=DATASET_CONFIG["docstring"]["max_samples"]
                )
                
                # Combine datasets
                combined_dataset = self._combine_datasets([code_search_dataset, docstring_dataset])
                
                # Preprocess for training
                processed_dataset = self.data_processor.preprocess_for_training(
                    combined_dataset,
                    code_column='code',
                    explanation_column='docstring'
                )
                
                # Create training pairs
                training_dataset = self.data_processor.create_training_pairs(processed_dataset)
                
                # Split dataset
                dataset = self.data_processor.split_dataset(training_dataset)
                
                # Save processed dataset
                self.data_processor.save_processed_dataset(
                    dataset['train'], 
                    self.output_dir / "processed_train.parquet"
                )
            
            logger.info(f"Dataset prepared: {dataset}")
            return dataset
            
        except Exception as e:
            logger.error(f"Error preparing dataset: {str(e)}")
            raise
    
    def _combine_datasets(self, datasets: List[Dataset]) -> Dataset:
        """Combine multiple datasets"""
        combined_data = []
        
        for dataset in datasets:
            # Convert to list and add to combined data
            combined_data.extend(dataset.to_list())
        
        return Dataset.from_list(combined_data)
    
    def tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenize function for dataset preprocessing
        
        Args:
            examples: Dataset examples
            
        Returns:
            Tokenized examples
        """
        # Tokenize inputs
        inputs = self.tokenizer(
            examples["input_text"],
            truncation=True,
            padding="max_length",
            max_length=MODEL_CONFIG["code_t5_max_length"],
            return_tensors="pt"
        )
        
        # Tokenize targets
        targets = self.tokenizer(
            examples["target_text"],
            truncation=True,
            padding="max_length",
            max_length=MODEL_CONFIG["max_explanation_length"],
            return_tensors="pt"
        )
        
        # Set labels
        labels = targets["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }
    
    def setup_training(self, train_dataset: Dataset, val_dataset: Dataset):
        """Setup training components"""
        try:
            logger.info("Setting up training...")
            
            # Tokenize datasets
            tokenized_train = train_dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=train_dataset.column_names
            )
            
            tokenized_val = val_dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=val_dataset.column_names
            )
            
            # Data collator
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                padding=True
            )
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=str(self.output_dir),
                num_train_epochs=TRAINING_CONFIG["num_epochs"],
                per_device_train_batch_size=TRAINING_CONFIG["batch_size"],
                per_device_eval_batch_size=TRAINING_CONFIG["batch_size"],
                learning_rate=TRAINING_CONFIG["learning_rate"],
                warmup_steps=TRAINING_CONFIG["warmup_steps"],
                weight_decay=TRAINING_CONFIG["weight_decay"],
                gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
                max_grad_norm=TRAINING_CONFIG["max_grad_norm"],
                save_steps=TRAINING_CONFIG["save_steps"],
                eval_steps=TRAINING_CONFIG["eval_steps"],
                logging_steps=TRAINING_CONFIG["logging_steps"],
                evaluation_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                report_to="wandb" if self.use_wandb else "none",
                remove_unused_columns=False,
                push_to_hub=False,
                dataloader_pin_memory=False
            )
            
            # Initialize trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_val,
                tokenizer=self.tokenizer,
                data_collator=data_collator
            )
            
            logger.info("Training setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up training: {str(e)}")
            raise
    
    def train(self, dataset_path: Optional[str] = None):
        """
        Main training function
        
        Args:
            dataset_path: Path to pre-processed dataset (optional)
        """
        try:
            logger.info("Starting training process...")
            
            # Load model and tokenizer
            self.load_model_and_tokenizer()
            
            # Prepare dataset
            dataset_dict = self.prepare_dataset(dataset_path)
            
            # Setup training
            self.setup_training(dataset_dict['train'], dataset_dict['validation'])
            
            # Start training
            logger.info("Starting model training...")
            train_result = self.trainer.train()
            
            # Save final model
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.output_dir)
            
            # Log training results
            logger.info(f"Training completed. Results: {train_result}")
            
            # Evaluate on test set
            if 'test' in dataset_dict:
                logger.info("Evaluating on test set...")
                test_results = self.trainer.evaluate(dataset_dict['test'])
                logger.info(f"Test results: {test_results}")
                
                # Save test results
                with open(self.output_dir / "test_results.json", 'w') as f:
                    json.dump(test_results, f, indent=2)
            
            # Save training results
            with open(self.output_dir / "training_results.json", 'w') as f:
                json.dump(train_result.metrics, f, indent=2)
            
            logger.info(f"Training completed successfully. Model saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
        finally:
            if self.use_wandb:
                wandb.finish()
    
    def evaluate_model(self, test_dataset: Dataset) -> Dict[str, float]:
        """
        Evaluate the trained model
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            logger.info("Evaluating model...")
            
            # Tokenize test dataset
            tokenized_test = test_dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=test_dataset.column_names
            )
            
            # Evaluate
            eval_results = self.trainer.evaluate(tokenized_test)
            
            logger.info(f"Evaluation results: {eval_results}")
            return eval_results
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise

def main():
    """Main function for training"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train CodeT5 model for code explanation")
    parser.add_argument("--model_name", type=str, default="Salesforce/codet5-base",
                       help="Pre-trained model name")
    parser.add_argument("--output_dir", type=str, default="models/code_explainer_finetuned",
                       help="Output directory for saving models")
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="Path to pre-processed dataset")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable Weights & Biases logging")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = CodeExplainerTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb
    )
    
    # Start training
    trainer.train(dataset_path=args.dataset_path)

if __name__ == "__main__":
    main()
