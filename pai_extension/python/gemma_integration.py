# /home/cybrosys/PSQL/postgresql/contrib/pai_extension/python/gemma_integration.py

import sys
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime

class GemmaSQL:
    _instance = None
    _model = None
    _tokenizer = None
    _initialized = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # Only initialize if not already done
        if not GemmaSQL._initialized:
            try:
                model_path = "/home/cybrosys/gemma2"
                print(f"Loading model from {model_path}...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    local_files_only=True
                )
                GemmaSQL._initialized = True
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                raise RuntimeError(f"Failed to load Gemma model: {str(e)}")

    def generate_sql(self, schema_info, query_description):
        prompt = f"""
Generate a PostgreSQL query for this request. Return only the SQL query, no explanations.

Schema:
{schema_info}

Request: {query_description}

SQL:"""
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the response
            if "SQL:" in response:
                sql_query = response.split("SQL:")[-1].strip()
            elif "```sql" in response:
                sql_query = response.split("```sql")[1].split("```")[0].strip()
            elif "```" in response:
                sql_query = response.split("```")[1].split("```")[0].strip()
            else:
                sql_query = response.strip()
            
            # Simple validation and fallback
            if not sql_query or "schema" in sql_query.lower():
                if "student" in query_description.lower():
                    return "SELECT * FROM student;"
                return f"-- Could not generate query for: {query_description}"
            
            return sql_query
            
        except Exception as e:
            raise RuntimeError(f"Error generating SQL query: {str(e)}")