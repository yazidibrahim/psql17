import os
import re
import time
import torch
import logging
import threading
import atexit
from dataclasses import dataclass
from typing import Optional, List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor

# Global variables
_generator_instance = None
_instance_lock = threading.Lock()

# Configure logging
logging.basicConfig(
    filename='/tmp/gemma_generator.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class ModelConfig:
    """Model configuration settings"""
    model_path: str = os.getenv('GEMMA_MODEL_PATH', '/home/cybrosys/gemma2')
    max_length: int = 256
    max_new_tokens: int = 128
    temperature: float = 0.3
    top_p: float = 0.95
    num_sequences: int = 1
    repetition_penalty: float = 1.1

class SQLGenerationError(Exception):
    """Custom exception for SQL generation errors"""
    pass

class ModelManager:
    """Manages the Gemma model lifecycle"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()
        self._is_initialized = False

    def initialize(self) -> None:
        """Initialize the model and tokenizer"""
        if not self._is_initialized:
            with self._lock:
                if not self._is_initialized:
                    try:
                        start_time = time.time()
                        logging.info("Initializing model...")
                        
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            self.config.model_path,
                            local_files_only=True
                        )
                        
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.config.model_path,
                            device_map="auto" if torch.cuda.is_available() else None,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            low_cpu_mem_usage=True,
                            local_files_only=True
                        ).to(self.device)
                        
                        self._is_initialized = True
                        logging.info(f"Model initialized in {time.time() - start_time:.2f}s")
                    
                    except Exception as e:
                        logging.error(f"Model initialization failed: {str(e)}")
                        raise SQLGenerationError(f"Failed to initialize model: {str(e)}")

    def cleanup(self) -> None:
        """Clean up model resources"""
        with self._lock:
            if self.model is not None:
                try:
                    del self.model
                    del self.tokenizer
                    torch.cuda.empty_cache()
                    self._is_initialized = False
                    logging.info("Model resources cleaned up")
                except Exception as e:
                    logging.error(f"Cleanup error: {str(e)}")

class SQLQueryCleaner:
    """Handles SQL query cleaning and validation"""
    
    SQL_PATTERNS = [
        (r'SELECT\s+.*?(?:FROM\s+.*?)?(?:;|\Z)', 'SELECT'),
        (r'INSERT\s+INTO\s+\w+\s*\([^)]*\)\s*VALUES\s*\([^)]*\)', 'INSERT'),
        (r'UPDATE\s+.*?(?:;|\Z)', 'UPDATE'),
        (r'DELETE\s+.*?(?:;|\Z)', 'DELETE'),
    ]

    @classmethod
    def clean_query(cls, text: str) -> str:
        """Clean and validate SQL query"""
        logging.debug(f"Starting query cleaning. Input: {text}")
        
        if not text:
            logging.warning("Empty query received")
            return cls._error_query("Empty query generated")

        # Remove comments and normalize whitespace
        text = re.sub(r'--.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        text = re.sub(r'\s+', ' ', text).strip()
        logging.debug(f"After cleaning: {text}")

        # Try to extract SQL query
        for pattern, cmd_type in cls.SQL_PATTERNS:
            logging.debug(f"Trying pattern for {cmd_type}")
            if match := re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                query = match.group(0).strip()
                query = re.sub(r';$', '', query)  # Remove trailing semicolon
                
                logging.debug(f"Matched query for {cmd_type}: {query}")
                
                if cls._validate_query(query, cmd_type):
                    logging.info(f"Valid query found: {query}")
                    return query
                else:
                    logging.warning(f"Query validation failed for {cmd_type}: {query}")

        # Special handling for INSERT statements
        if text.upper().startswith('INSERT INTO'):
            if cls._validate_query(text, 'INSERT'):
                return text

        logging.warning("No valid SQL query found in text")
        return cls._error_query("No valid SQL query found")

    @staticmethod
    def _validate_query(query: str, cmd_type: str) -> bool:
        """Basic SQL query validation"""
        logging.debug(f"Validating {cmd_type} query: {query}")
        
        # Check for basic SQL injection patterns
        blacklist = [';--', 'UNION', 'DROP', 'TRUNCATE']
        if any(pattern.upper() in query.upper() for pattern in blacklist):
            logging.warning(f"Query contains blacklisted pattern: {query}")
            return False
            
        # Validate command structure
        if cmd_type == 'SELECT':
            return True
        elif cmd_type == 'INSERT':
            query_upper = query.upper()
            return ('INSERT INTO' in query_upper and 
                   'VALUES' in query_upper)
        elif cmd_type == 'UPDATE':
            return 'SET' in query.upper()
        elif cmd_type == 'DELETE':
            return 'FROM' in query.upper()
        return False

    @staticmethod
    def _error_query(message: str) -> str:
        """Generate error query"""
        return f"SELECT 'Error: {message}' as error"

class QueryGenerator:
    """Main query generation class"""
    def __init__(self, config: ModelConfig = ModelConfig()):
        self.model_manager = ModelManager(config)
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=1)

    def generate_query(self, natural_query: str, schema_context: str) -> str:
        """Generate SQL query from natural language"""
        try:
            self.model_manager.initialize()

            # Construct prompt
            prompt = self._build_prompt(natural_query, schema_context)
            logging.debug(f"Generated prompt: {prompt}")

            # Generate query
            inputs = self.model_manager.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length
            ).to(self.model_manager.device)

            with torch.no_grad():
                outputs = self.model_manager.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=True,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    num_return_sequences=self.config.num_sequences,
                    pad_token_id=self.model_manager.tokenizer.eos_token_id,
                    eos_token_id=self.model_manager.tokenizer.eos_token_id,
                    repetition_penalty=self.config.repetition_penalty
                )

            # Process output
            generated_text = self.model_manager.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logging.debug(f"Raw generated text: {generated_text}")
            
            # Extract SQL query
            sql_query = self._extract_sql(generated_text)
            if not sql_query:
                return SQLQueryCleaner._error_query("No valid SQL query generated")
            
            # Clean and return query
            return SQLQueryCleaner.clean_query(sql_query)

        except Exception as e:
            logging.error(f"Query generation failed: {str(e)}")
            return SQLQueryCleaner._error_query(str(e))

    @staticmethod
    def _build_prompt(natural_query: str, schema_context: str) -> str:
        """Build the prompt for the model"""
        return f"""### System: You are a SQL query generator. Generate only the SQL query without any additional text or explanations.
### Database Schema:
{schema_context}

### User: {natural_query}

### Assistant: Here is the SQL query:"""

    @staticmethod
    def _extract_sql(generated_text: str) -> str:
        """Extract SQL query from generated text"""
        if "Here is the SQL query:" in generated_text:
            generated_text = generated_text.split("Here is the SQL query:")[-1]
        
        lines = generated_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('--') and not line.startswith('/*'):
                return line
        
        return ""

def get_generator(config: Optional[ModelConfig] = None) -> QueryGenerator:
    """Get or create the global generator instance"""
    global _generator_instance
    if _generator_instance is None:
        with _instance_lock:
            if _generator_instance is None:
                try:
                    _generator_instance = QueryGenerator(config or ModelConfig())
                except Exception as e:
                    logging.error(f"Failed to create generator: {str(e)}")
                    raise
    return _generator_instance

def generate_sql_query(natural_language_query: str, schema_context: str) -> str:
    """Entry point function for PostgreSQL"""
    try:
        generator = get_generator()
        return generator.generate_query(natural_language_query, schema_context)
    except Exception as e:
        logging.error(f"SQL query generation failed: {str(e)}")
        return SQLQueryCleaner._error_query(str(e))

# Cleanup handler
def cleanup():
    """Cleanup resources on exit"""  
    global _generator_instance
    if _generator_instance is not None:
        _generator_instance.model_manager.cleanup()

# Register cleanup handler
atexit.register(cleanup)

# Make generate_sql_query available at the module level
__all__ = ['generate_sql_query']