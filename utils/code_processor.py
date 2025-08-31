"""
CodeProcessor: Utility class for preprocessing and analyzing Python code
"""

import ast
import re
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeProcessor:
    """
    Utility class for processing and analyzing Python code
    """
    
    def __init__(self):
        """Initialize the CodeProcessor"""
        self.supported_languages = ['python']
    
    def process_code(self, code: str, language: str = 'python') -> str:
        """
        Process and clean code for better model understanding
        
        Args:
            code: Raw code string
            language: Programming language (default: python)
            
        Returns:
            Processed code string
        """
        if language.lower() not in self.supported_languages:
            raise ValueError(f"Unsupported language: {language}")
        
        try:
            # Clean and normalize code
            processed_code = self._clean_code(code)
            
            # Validate Python syntax if it's Python code
            if language.lower() == 'python':
                self._validate_python_syntax(processed_code)
            
            return processed_code
            
        except Exception as e:
            logger.error(f"Error processing code: {str(e)}")
            raise
    
    def _clean_code(self, code: str) -> str:
        """Clean and normalize code"""
        # Remove leading/trailing whitespace
        code = code.strip()
        
        # Normalize line endings
        code = code.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive blank lines
        code = re.sub(r'\n\s*\n\s*\n', '\n\n', code)
        
        # Ensure consistent indentation
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            if line.strip():  # Non-empty line
                cleaned_lines.append(line)
            else:  # Empty line
                cleaned_lines.append('')
        
        return '\n'.join(cleaned_lines)
    
    def _validate_python_syntax(self, code: str):
        """Validate Python syntax using AST"""
        try:
            ast.parse(code)
        except SyntaxError as e:
            logger.warning(f"Python syntax warning: {str(e)}")
            # Don't raise error, just log warning
    
    def extract_functions(self, code: str) -> List[Dict[str, Any]]:
        """
        Extract function definitions from Python code
        
        Args:
            code: Python code string
            
        Returns:
            List of function information dictionaries
        """
        try:
            tree = ast.parse(code)
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'lineno': node.lineno,
                        'args': [arg.arg for arg in node.args.args],
                        'docstring': ast.get_docstring(node),
                        'body_lines': len(node.body)
                    }
                    functions.append(func_info)
            
            return functions
            
        except Exception as e:
            logger.error(f"Error extracting functions: {str(e)}")
            return []
    
    def extract_classes(self, code: str) -> List[Dict[str, Any]]:
        """
        Extract class definitions from Python code
        
        Args:
            code: Python code string
            
        Returns:
            List of class information dictionaries
        """
        try:
            tree = ast.parse(code)
            classes = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'lineno': node.lineno,
                        'bases': [base.id for base in node.bases if isinstance(base, ast.Name)],
                        'docstring': ast.get_docstring(node),
                        'methods': [],
                        'attributes': []
                    }
                    
                    # Extract methods and attributes
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            class_info['methods'].append(item.name)
                        elif isinstance(item, ast.Assign):
                            for target in item.targets:
                                if isinstance(target, ast.Name):
                                    class_info['attributes'].append(target.id)
                    
                    classes.append(class_info)
            
            return classes
            
        except Exception as e:
            logger.error(f"Error extracting classes: {str(e)}")
            return []
    
    def extract_imports(self, code: str) -> List[str]:
        """
        Extract import statements from Python code
        
        Args:
            code: Python code string
            
        Returns:
            List of import statements
        """
        try:
            tree = ast.parse(code)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(f"import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        if alias.asname:
                            imports.append(f"from {module} import {alias.name} as {alias.asname}")
                        else:
                            imports.append(f"from {module} import {alias.name}")
            
            return imports
            
        except Exception as e:
            logger.error(f"Error extracting imports: {str(e)}")
            return []
    
    def get_code_statistics(self, code: str) -> Dict[str, Any]:
        """
        Get statistics about the code
        
        Args:
            code: Code string
            
        Returns:
            Dictionary containing code statistics
        """
        try:
            lines = code.split('\n')
            total_lines = len(lines)
            non_empty_lines = len([line for line in lines if line.strip()])
            comment_lines = len([line for line in lines if line.strip().startswith('#')])
            code_lines = non_empty_lines - comment_lines
            
            # Extract structural information
            functions = self.extract_functions(code)
            classes = self.extract_classes(code)
            imports = self.extract_imports(code)
            
            stats = {
                'total_lines': total_lines,
                'non_empty_lines': non_empty_lines,
                'comment_lines': comment_lines,
                'code_lines': code_lines,
                'function_count': len(functions),
                'class_count': len(classes),
                'import_count': len(imports),
                'functions': functions,
                'classes': classes,
                'imports': imports
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting code statistics: {str(e)}")
            return {}
    
    def format_code(self, code: str) -> str:
        """
        Format code using basic formatting rules
        
        Args:
            code: Raw code string
            
        Returns:
            Formatted code string
        """
        try:
            # Basic formatting
            lines = code.split('\n')
            formatted_lines = []
            indent_level = 0
            
            for line in lines:
                stripped = line.strip()
                
                if not stripped:
                    formatted_lines.append('')
                    continue
                
                # Handle indentation
                if stripped.endswith(':'):
                    formatted_lines.append('    ' * indent_level + stripped)
                    indent_level += 1
                elif stripped.startswith(('return', 'break', 'continue', 'pass')):
                    indent_level = max(0, indent_level - 1)
                    formatted_lines.append('    ' * indent_level + stripped)
                else:
                    formatted_lines.append('    ' * indent_level + stripped)
            
            return '\n'.join(formatted_lines)
            
        except Exception as e:
            logger.error(f"Error formatting code: {str(e)}")
            return code
    
    def is_valid_python(self, code: str) -> bool:
        """
        Check if code is valid Python
        
        Args:
            code: Code string to validate
            
        Returns:
            True if valid Python, False otherwise
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def get_code_complexity(self, code: str) -> Dict[str, Any]:
        """
        Calculate code complexity metrics
        
        Args:
            code: Code string
            
        Returns:
            Dictionary containing complexity metrics
        """
        try:
            tree = ast.parse(code)
            
            # Count different types of nodes
            node_counts = {
                'functions': 0,
                'classes': 0,
                'loops': 0,
                'conditionals': 0,
                'assignments': 0,
                'calls': 0
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    node_counts['functions'] += 1
                elif isinstance(node, ast.ClassDef):
                    node_counts['classes'] += 1
                elif isinstance(node, (ast.For, ast.While)):
                    node_counts['loops'] += 1
                elif isinstance(node, ast.If):
                    node_counts['conditionals'] += 1
                elif isinstance(node, ast.Assign):
                    node_counts['assignments'] += 1
                elif isinstance(node, ast.Call):
                    node_counts['calls'] += 1
            
            # Calculate cyclomatic complexity (simplified)
            complexity = (node_counts['functions'] + 
                         node_counts['loops'] + 
                         node_counts['conditionals'] + 1)
            
            return {
                'node_counts': node_counts,
                'cyclomatic_complexity': complexity,
                'overall_complexity': 'Low' if complexity < 5 else 'Medium' if complexity < 10 else 'High'
            }
            
        except Exception as e:
            logger.error(f"Error calculating code complexity: {str(e)}")
            return {}
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """
        Comprehensive code analysis
        
        Args:
            code: Code string to analyze
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        try:
            analysis = {
                'statistics': self.get_code_statistics(code),
                'complexity': self.get_code_complexity(code),
                'functions': self.extract_functions(code),
                'classes': self.extract_classes(code),
                'imports': self.extract_imports(code),
                'is_valid': self.is_valid_python(code),
                'formatted_code': self.format_code(code)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing code: {str(e)}")
            return {}
