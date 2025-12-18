# ========================================
# SECTION 1: IMPORTS AND INITIALIZATION
# ========================================
# All imports consolidated, logger setup

import logging
import re
import asyncio
import uuid
import json
import os
import math
import sys
import tempfile
import time
import threading
import random
from typing import Optional, Dict, Any, List, Union, Literal, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import httpx
import json as json_lib

logger = logging.getLogger(__name__)

# ========================================  
# SECTION 2: CONFIGURATION AND SETTINGS
# ========================================
# Environment variables, model configs, algorithm settings

from dotenv import load_dotenv
from utils.config import client_rotator

load_dotenv()

# LLM Configuration - Using new client rotator system
# Generation parameters
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "1.0"))

# Evolutionary Algorithm Settings
POPULATION_SIZE = 15
GENERATIONS = 15
# Threshold for switching to bug-fix prompt
# If a program has errors and its correctness score is below this, a bug-fix prompt will be used.
BUG_FIX_CORRECTNESS_THRESHOLD = float(os.getenv("BUG_FIX_CORRECTNESS_THRESHOLD", "0.1"))
# Threshold for using the primary (potentially more powerful/expensive) LLM for mutation
HIGH_FITNESS_THRESHOLD_FOR_PRIMARY_LLM = float(os.getenv("HIGH_FITNESS_THRESHOLD_FOR_PRIMARY_LLM", "0.8"))
ELITISM_COUNT = 1
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.2

# Island Model Settings
NUM_ISLANDS = 3  # Number of subpopulations
MIGRATION_FREQUENCY = 4  # Number of generations between migrations
ISLAND_POPULATION_SIZE = POPULATION_SIZE // NUM_ISLANDS  # Programs per island
MIN_ISLAND_SIZE = 2  # Minimum number of programs per island
MIGRATION_RATE = 0.2  # Rate at which programs migrate between islands

# Debug Settings
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
EVALUATION_TIMEOUT_SECONDS = 800

# Docker Execution Settings
DOCKER_IMAGE_NAME = os.getenv("DOCKER_IMAGE_NAME", "code-evaluator:latest")
DOCKER_NETWORK_DISABLED = os.getenv("DOCKER_NETWORK_DISABLED", "True").lower() == "true"

DATABASE_TYPE = "json"
DATABASE_PATH = "program_database.json"

# Logging Configuration
LOG_LEVEL = "DEBUG" if DEBUG else "INFO"
LOG_FILE = "alpha_evolve.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

API_MAX_RETRIES = 5
API_RETRY_DELAY_SECONDS = 10

RL_TRAINING_INTERVAL_GENERATIONS = 50
RL_MODEL_PATH = "rl_finetuner_model.pth"

MONITORING_DASHBOARD_URL = "http://localhost:8080"


def get_setting(key, default=None):
    """
    Retrieves a setting value.
    For LLM models, it specifically checks if the primary choice is available,
    otherwise falls back to a secondary/default if defined.
    """
    return globals().get(key, default)

def get_next_client_config():
    """Get the next client configuration from the rotator"""
    if client_rotator:
        return client_rotator.get_next_client_config()
    else:
        raise RuntimeError("No client rotator available. Check your configuration.")


# ========================================
# SECTION 3: DATA MODELS AND INTERFACES  
# ========================================
# Program, TaskDefinition dataclasses, all abstract interfaces

@dataclass
class Program:
    id: str
    code: str
    fitness_scores: Dict[str, float] = field(default_factory=dict)                                                 
    generation: int = 0
    parent_id: Optional[str] = None
    island_id: Optional[int] = None
    errors: List[str] = field(default_factory=list)
    status: str = "unevaluated"
    created_at: float = field(default_factory=lambda: time.time())  # Track program age
    task_id: Optional[str] = None

@dataclass
class TaskDefinition:
    id: str
    description: str                                              
    function_name_to_evolve: Optional[str] = None  # Can be used if evolving a single function
    target_file_path: Optional[str] = None # Path to the file containing code to be evolved
    evolve_blocks: Optional[List[Dict[str, Any]]] = None # Defines specific blocks within the target_file_path to evolve
                                                        # e.g., [{'block_id': 'optimizer_logic', 'start_marker': '# EVOLVE-BLOCK-START optimizer', 'end_marker': '# EVOLVE-BLOCK-END optimizer'}]
    input_output_examples: Optional[List[Dict[str, Any]]] = None                                                    
    evaluation_criteria: Optional[Dict[str, Any]] = None                                                            
    initial_code_prompt: Optional[str] = "Provide an initial Python solution for the following problem:"
    allowed_imports: Optional[List[str]] = None
    tests: Optional[List[Dict[str, Any]]] = None # List of test groups. Each group is a dict, can include 'name', 'description', 'level' (for cascade), and 'test_cases'.
    expert_knowledge: Optional[str] = None # Relevant expert knowledge, equations, or snippets

class BaseAgent(ABC):
    """Base class for all agents."""
    @abstractmethod
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """Main execution method for an agent."""
        pass

class TaskManagerInterface(BaseAgent):
    @abstractmethod
    async def manage_evolutionary_cycle(self):
        pass

class PromptDesignerInterface(BaseAgent):
    @abstractmethod
    def design_initial_prompt(self, task: TaskDefinition) -> str:
        pass

    @abstractmethod
    def design_mutation_prompt(self, task: TaskDefinition, parent_program: Program, evaluation_feedback: Optional[Dict] = None) -> str:
        pass

    @abstractmethod
    def design_bug_fix_prompt(self, task: TaskDefinition, program: Program, error_info: Dict) -> str:
        pass

class CodeGeneratorInterface(BaseAgent):
    @abstractmethod
    async def generate_code(self, prompt: str, model_name: Optional[str] = None, temperature: Optional[float] = 0.7, output_format: str = "code") -> str:
        pass

class EvaluatorAgentInterface(BaseAgent):
    @abstractmethod
    async def evaluate_program(self, program: Program, task: TaskDefinition) -> Program:
        pass

class DatabaseAgentInterface(BaseAgent):
    @abstractmethod
    async def save_program(self, program: Program):
        pass

    @abstractmethod
    async def get_program(self, program_id: str) -> Optional[Program]:
        pass

    @abstractmethod
    async def get_best_programs(self, task_id: str, limit: int = 10, objective: Optional[str] = None) -> List[Program]:
        pass
    
    @abstractmethod
    async def get_programs_for_next_generation(self, task_id: str, generation_size: int) -> List[Program]:
        pass

class SelectionControllerInterface(BaseAgent):
    @abstractmethod
    def select_parents(self, evaluated_programs: List[Program], num_parents: int) -> List[Program]:
        pass

    @abstractmethod
    def select_survivors(self, current_population: List[Program], offspring_population: List[Program], population_size: int) -> List[Program]:
        pass

    @abstractmethod
    def initialize_islands(self, initial_programs: List[Program]) -> None:
        pass

class RLFineTunerInterface(BaseAgent):
    @abstractmethod
    async def update_policy(self, experience_data: List[Dict]):
        pass

class MonitoringAgentInterface(BaseAgent):
    @abstractmethod
    async def log_metrics(self, metrics: Dict):
        pass

    @abstractmethod
    async def report_status(self):
        pass


# ========================================
# SECTION 4: CODE GENERATOR AGENT
# ========================================
# CodeGeneratorAgent implementation with diff handling

class CodeGeneratorAgent(CodeGeneratorInterface):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.generation_config = {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS,
        }
        logger.info(f"CodeGeneratorAgent initialized with client rotator system")

    async def generate_code(self, prompt: str, model_name: Optional[str] = None, temperature: Optional[float] = None, output_format: str = "code", **kwargs) -> str:
        logger.info(f"Attempting to generate code using client rotator, output_format: {output_format}")

        if output_format == "diff":
            prompt += '''

I need you to provide your changes as a sequence of diff blocks in the following format:

<<<<<<< SEARCH
# Original code block to be found and replaced (COPY EXACTLY from original)
=======
# New code block to replace the original
>>>>>>> REPLACE

IMPORTANT DIFF GUIDELINES:
1. The SEARCH block MUST be an EXACT copy of code from the original - match whitespace, indentation, and line breaks precisely
2. Each SEARCH block should be large enough (3-5 lines minimum) to uniquely identify where the change should be made
3. Include context around the specific line(s) you want to change
4. Make multiple separate diff blocks if you need to change different parts of the code
5. For each diff, the SEARCH and REPLACE blocks must be complete, valid code segments
6. Pay special attention to matching the exact original indentation of the code in your SEARCH block, as this is crucial for correct application in environments sensitive to indentation (like Python).

Example of a good diff:
<<<<<<< SEARCH
def calculate_sum(numbers):
    result = 0
    for num in numbers:
        result += num
    return result
=======
def calculate_sum(numbers):
    if not numbers:
        return 0
    result = 0
    for num in numbers:
        result += num
    return result
>>>>>>> REPLACE

Make sure your diff can be applied correctly!
'''

        logger.debug(f"Received prompt for code generation (format: {output_format}):\n--PROMPT START--\n{prompt}\n--PROMPT END--")

        # Get configuration
        current_generation_config = self.generation_config.copy()
        if temperature is not None:
            current_generation_config["temperature"] = temperature
            logger.debug(f"Using temperature override: {temperature}")

        # Get client configuration from rotator
        try:
            client_config = get_next_client_config()
            logger.debug(f"Using client: {client_config.base_url} with model: {client_config.model}")
        except Exception as e:
            logger.error(f"Failed to get client configuration: {e}")
            return ""

        # Prepare request payload
        payload = {
            "model": client_config.model,
            "messages": [{"role": "user", "content": prompt}],
            **current_generation_config
        }

        headers = {
            "Authorization": f"Bearer {client_config.api_key}",
            "Content-Type": "application/json"
        }

        # Add proxy if configured
        proxies = None
        if client_config.proxy:
            proxies = {"http://": client_config.proxy, "https://": client_config.proxy}

        retries = API_MAX_RETRIES
        delay = API_RETRY_DELAY_SECONDS
        
        for attempt in range(retries):
            try:
                logger.debug(f"API Call Attempt {attempt + 1} of {retries} to {client_config.base_url}")
                
                async with httpx.AsyncClient(proxies=proxies, timeout=30.0) as client:
                    response = await client.post(
                        f"{client_config.base_url}/chat/completions",
                        headers=headers,
                        json=payload
                    )
                    response.raise_for_status()
                    
                    response_data = response.json()
                    
                    if not response_data.get("choices"):
                        logger.warning("API returned no choices.")
                        return ""
                    
                    generated_text = response_data["choices"][0]["message"]["content"]
                    logger.debug(f"Raw response from API:\n--RESPONSE START--\n{generated_text}\n--RESPONSE END--")
                    
                    if output_format == "code":
                        return self._clean_llm_output(generated_text)
                    else:
                        return generated_text
                        
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limit
                    logger.warning(f"Rate limit error on attempt {attempt + 1}. Retrying in {delay}s...")
                elif e.response.status_code >= 500:  # Server error
                    logger.warning(f"Server error {e.response.status_code} on attempt {attempt + 1}. Retrying in {delay}s...")
                else:
                    logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
                    raise
                    
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"API call failed after {retries} retries")
                    raise
                    
            except Exception as e:
                logger.warning(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"Code generation failed after {retries} retries")
                    raise

        return ""

    def _clean_llm_output(self, raw_code: str) -> str:
        """
        Cleans the raw output from the LLM, typically removing markdown code fences.
        Example: ```python\ncode\n``` -> code
        """
        logger.debug(f"Attempting to clean raw LLM output. Input length: {len(raw_code)}")
        code = raw_code.strip()
        
        if code.startswith("```python") and code.endswith("```"):
            cleaned = code[len("```python"): -len("```")].strip()
            logger.debug("Cleaned Python markdown fences.")
            return cleaned
        elif code.startswith("```") and code.endswith("```"):
            cleaned = code[len("```"): -len("```")].strip()
            logger.debug("Cleaned generic markdown fences.")
            return cleaned
            
        logger.debug("No markdown fences found or standard cleaning applied to the stripped code.")
        return code

    def _apply_diff(self, parent_code: str, diff_text: str) -> str:
        """
        Applies a diff in the AlphaEvolve format to the parent code.
        Diff format:
        <<<<<<< SEARCH
        # Original code block
        =======
        # New code block
        >>>>>>> REPLACE
        
        Uses fuzzy matching to handle slight variations in whitespace and indentation.
        """
        logger.info("Attempting to apply diff.")
        logger.debug(f"Parent code length: {len(parent_code)}")
        logger.debug(f"Diff text:\n{diff_text}")

        modified_code = parent_code
        diff_pattern = re.compile(r"<<<<<<< SEARCH\s*?\n(.*?)\n=======\s*?\n(.*?)\n>>>>>>> REPLACE", re.DOTALL)
        
                                                                                
                                                             
        replacements_made = []
        
        for match in diff_pattern.finditer(diff_text):
            search_block = match.group(1)
            replace_block = match.group(2)
            
                                                                        
            search_block_normalized = search_block.replace('\r\n', '\n').replace('\r', '\n').strip()
            
            try:
                                       
                if search_block_normalized in modified_code:
                    logger.debug(f"Found exact match for SEARCH block")
                    modified_code = modified_code.replace(search_block_normalized, replace_block, 1)
                    logger.debug(f"Applied one diff block. SEARCH:\n{search_block_normalized}\nREPLACE:\n{replace_block}")
                else:
                                                                                 
                    normalized_search = re.sub(r'\s+', ' ', search_block_normalized)
                    normalized_code = re.sub(r'\s+', ' ', modified_code)
                    
                    if normalized_search in normalized_code:
                        logger.debug(f"Found match after whitespace normalization")
                                                                        
                        start_pos = normalized_code.find(normalized_search)
                        
                                                                          
                        original_pos = 0
                        norm_pos = 0
                        
                        while norm_pos < start_pos and original_pos < len(modified_code):
                            if not modified_code[original_pos].isspace() or (
                                original_pos > 0 and 
                                modified_code[original_pos].isspace() and 
                                not modified_code[original_pos-1].isspace()
                            ):
                                norm_pos += 1
                            original_pos += 1
                        
                                               
                        end_pos = original_pos
                        remaining_chars = len(normalized_search)
                        
                        while remaining_chars > 0 and end_pos < len(modified_code):
                            if not modified_code[end_pos].isspace() or (
                                end_pos > 0 and 
                                modified_code[end_pos].isspace() and 
                                not modified_code[end_pos-1].isspace()
                            ):
                                remaining_chars -= 1
                            end_pos += 1
                        
                                                                                        
                        overlap = False
                        for start, end in replacements_made:
                            if (start <= original_pos <= end) or (start <= end_pos <= end):
                                overlap = True
                                break
                        
                        if not overlap:
                                                               
                            actual_segment = modified_code[original_pos:end_pos]
                            logger.debug(f"Replacing segment:\n{actual_segment}\nWith:\n{replace_block}")
                            
                                                 
                            modified_code = modified_code[:original_pos] + replace_block + modified_code[end_pos:]
                            
                                                     
                            replacements_made.append((original_pos, original_pos + len(replace_block)))
                        else:
                            logger.warning(f"Diff application: Skipping overlapping replacement")
                    else:
                                                               
                        search_lines = search_block_normalized.splitlines()
                        parent_lines = modified_code.splitlines()
                        
                                                                      
                        if len(search_lines) >= 3:
                                                                  
                            first_line = search_lines[0].strip()
                            last_line = search_lines[-1].strip()
                            
                            for i, line in enumerate(parent_lines):
                                if first_line in line.strip() and i + len(search_lines) <= len(parent_lines):
                                                                     
                                    if last_line in parent_lines[i + len(search_lines) - 1].strip():
                                                                                       
                                        matched_segment = '\n'.join(parent_lines[i:i + len(search_lines)])
                                        
                                                              
                                        modified_code = '\n'.join(
                                            parent_lines[:i] + 
                                            replace_block.splitlines() + 
                                            parent_lines[i + len(search_lines):]
                                        )
                                        logger.debug(f"Applied line-by-line match. SEARCH:\n{matched_segment}\nREPLACE:\n{replace_block}")
                                        break
                            else:
                                logger.warning(f"Diff application: SEARCH block not found even with line-by-line search:\n{search_block_normalized}")
                        else:
                            logger.warning(f"Diff application: SEARCH block not found in current code state:\n{search_block_normalized}")
            except re.error as e:
                logger.error(f"Regex error during diff application: {e}")
                continue
            except Exception as e:
                logger.error(f"Error during diff application: {e}", exc_info=True)
                continue
        
        if modified_code == parent_code and diff_text.strip():
             logger.warning("Diff text was provided, but no changes were applied. Check SEARCH blocks/diff format.")
        elif modified_code != parent_code:
             logger.info("Diff successfully applied, code has been modified.")
        else:
             logger.info("No diff text provided or diff was empty, code unchanged.")
             
        return modified_code

    async def execute(self, prompt: str, model_name: Optional[str] = None, temperature: Optional[float] = None, output_format: str = "code", parent_code_for_diff: Optional[str] = None, litellm_extra_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Generic execution method.
        If output_format is 'diff', it generates a diff and applies it to parent_code_for_diff.
        Otherwise, it generates full code.
        """
        logger.debug(f"CodeGeneratorAgent.execute called. Output format: {output_format}")
        
        generated_output = await self.generate_code(
            prompt=prompt, 
            model_name=model_name, 
            temperature=temperature,
            output_format=output_format,
            litellm_extra_params=litellm_extra_params
        )

        if output_format == "diff":
            if not parent_code_for_diff:
                logger.error("Output format is 'diff' but no parent_code_for_diff provided. Returning raw diff.")
                return generated_output 
            
            if not generated_output.strip():
                 logger.info("Generated diff is empty. Returning parent code.")
                 return parent_code_for_diff

            try:
                logger.info("Applying generated diff to parent code.")
                modified_code = self._apply_diff(parent_code_for_diff, generated_output)
                return modified_code
            except Exception as e:
                logger.error(f"Error applying diff: {e}. Returning raw diff text.", exc_info=True)
                return generated_output
        else:         
            return generated_output

                                                 
if __name__ == '__main__':
    import asyncio
    logging.basicConfig(level=logging.DEBUG)
    from unittest.mock import Mock # Added for mocking
    
    async def test_diff_application():
        agent = CodeGeneratorAgent()
        parent = """Line 1
Line 2 to be replaced
Line 3
Another block
To be changed
End of block
Final line"""

        diff = """Some preamble text from LLM...
<<<<<<< SEARCH
Line 2 to be replaced
=======
Line 2 has been successfully replaced
>>>>>>> REPLACE

Some other text...

<<<<<<< SEARCH
Another block
To be changed
End of block
=======
This
Entire
Block
Is New
>>>>>>> REPLACE
Trailing text..."""
        expected_output = """Line 1
Line 2 has been successfully replaced
Line 3
This
Entire
Block
Is New
Final line"""
        
        print("--- Testing _apply_diff directly ---")
        result = agent._apply_diff(parent, diff)
        print("Result of diff application:")
        print(result)
        assert result.strip() == expected_output.strip(), f"Direct diff application failed.\nExpected:\n{expected_output}\nGot:\n{result}"
        print("_apply_diff test passed.")

        print("\n--- Testing execute with output_format='diff' ---")
        async def mock_generate_code(prompt, model_name, temperature, output_format, litellm_extra_params=None): # Added litellm_extra_params
            return diff
        
        agent.generate_code = mock_generate_code 
        
        result_execute_diff = await agent.execute(
            prompt="doesn't matter for this mock", 
            parent_code_for_diff=parent,
            output_format="diff",
            litellm_extra_params={"example_param": "example_value"} # Added for testing
        )
        print("Result of execute with diff:")
        print(result_execute_diff)
        assert result_execute_diff.strip() == expected_output.strip(), f"Execute with diff failed.\nExpected:\n{expected_output}\nGot:\n{result_execute_diff}"
        print("Execute with diff test passed.")


    async def test_generation():
        agent = CodeGeneratorAgent()
        
        test_prompt_full_code = "Write a Python function that takes two numbers and returns their sum."
        
        # Mock litellm.acompletion for full code generation test
        original_acompletion = litellm.acompletion
        async def mock_litellm_acompletion(*args, **kwargs):
            mock_response = Mock()
            mock_message = Mock()
            mock_message.content = "def mock_function():\n  return 'mocked_code'"
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = mock_message
            return mock_response
        
        litellm.acompletion = mock_litellm_acompletion
        
        try:
            generated_full_code = await agent.execute(test_prompt_full_code, temperature=0.6, output_format="code")
            print("\n--- Generated Full Code (via execute) ---")
            print(generated_full_code)
            print("----------------------")
            assert "def mock_function" in generated_full_code, "Full code generation with mock seems to have failed."
        finally:
            litellm.acompletion = original_acompletion

        parent_code_for_llm_diff = '''
def greet(name):
    return f"Hello, {name}!"

def process_data(data):
    # TODO: Implement data processing
    return data * 2 # Simple placeholder
'''
        test_prompt_diff_gen = f'''
Current code:
```python
{parent_code_for_llm_diff}
```
Task: Modify the `process_data` function to add 5 to the result instead of multiplying by 2.
Also, change the greeting in `greet` to "Hi, {name}!!!".
'''
        async def mock_generate_empty_diff(prompt, model_name, temperature, output_format):
            return "  \n  " 
        
        original_generate_code = agent.generate_code 
        agent.generate_code = mock_generate_empty_diff
        
        print("\n--- Testing execute with empty diff from LLM ---")
        result_empty_diff = await agent.execute(
            prompt="doesn't matter",
            parent_code_for_diff=parent_code_for_llm_diff,
            output_format="diff"
        )
        assert result_empty_diff == parent_code_for_llm_diff, "Empty diff should return parent code."
        print("Execute with empty diff test passed.")
        agent.generate_code = original_generate_code

    async def main_tests():
        await test_diff_application()
                                                                                     
        print("\nAll selected local tests in CodeGeneratorAgent passed.")

    asyncio.run(main_tests())


# ========================================
# SECTION 5: DATABASE AGENT
# ========================================
# InMemoryDatabaseAgent with JSON persistence

class InMemoryDatabaseAgent(DatabaseAgentInterface, BaseAgent):
    """An in-memory database that persists to a JSON file."""
    def __init__(self):
        super().__init__()
        self._programs: Dict[str, Program] = {}
        self._db_file_path = DATABASE_PATH
        self._lock = asyncio.Lock() # Lock for file operations
        self._load_from_file() # Load existing data on init
        logger.info(f"InMemoryDatabaseAgent initialized. Data persistence: {self._db_file_path}")

    def _load_from_file(self):
        if os.path.exists(self._db_file_path):
            try:
                with open(self._db_file_path, 'r') as f:
                    data = json.load(f)
                    for prog_id, prog_data in data.items():
                        self._programs[prog_id] = Program(**prog_data) 
                logger.info(f"Loaded {len(self._programs)} programs from {self._db_file_path}")
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from {self._db_file_path}. Starting with an empty database.")
                self._programs = {}
            except Exception as e:
                logger.error(f"Error loading database from {self._db_file_path}: {e}. Starting with an empty database.")
                self._programs = {}
        else:
            logger.info(f"Database file {self._db_file_path} not found. Starting with an empty database.")

    async def _save_to_file(self):
        async with self._lock:
            try:
                # Serialize Program objects to dictionaries
                data_to_save = {prog_id: prog.__dict__ for prog_id, prog in self._programs.items()}
                with open(self._db_file_path, 'w') as f:
                    json.dump(data_to_save, f, indent=4)
                logger.debug(f"Successfully saved {len(self._programs)} programs to {self._db_file_path}")
            except Exception as e:
                logger.error(f"Error saving database to {self._db_file_path}: {e}")

    async def save_program(self, program: Program) -> None:
        logger.info(f"Saving program: {program.id} (Generation: {program.generation}) to database.")
        async with self._lock:
            if program.id in self._programs:
                logger.warning(f"Program with ID {program.id} already exists. It will be overwritten.")
            self._programs[program.id] = program
        await self._save_to_file() # Persist after every save
        logger.debug(f"Program {program.id} data: {program}")

    async def get_program(self, program_id: str) -> Optional[Program]:
        logger.debug(f"Attempting to retrieve program by ID: {program_id}")
        # No lock needed for read if _programs is mostly read-after-write and writes are locked
        program = self._programs.get(program_id)
        if program:
            logger.info(f"Retrieved program: {program.id}")
        else:
            logger.warning(f"Program with ID: {program_id} not found in database.")
        return program

    async def get_all_programs(self) -> List[Program]:
        logger.debug(f"Retrieving all {len(self._programs)} programs from database.")
        return list(self._programs.values())

    async def get_best_programs(
        self,
        task_id: str, # task_id is not strictly used by InMemoryDB for filtering, but part of interface
        limit: int = 5,
        objective: Literal["correctness", "runtime_ms"] = "correctness",
        sort_order: Literal["asc", "desc"] = "desc",
    ) -> List[Program]:
        logger.info(f"Retrieving best programs (task: {task_id}). Limit: {limit}, Objective: {objective}, Order: {sort_order}")
        if not self._programs:
            logger.info("No programs in database to retrieve 'best' from.")
            return []

        all_progs = list(self._programs.values())

        if objective == "correctness":
            sorted_programs = sorted(all_progs, key=lambda p: p.fitness_scores.get("correctness", 0.0), reverse=(sort_order == "desc"))
        elif objective == "runtime_ms":
            # For runtime_ms: sort_order='asc' (best) means reverse=False (lowest first)
            # sort_order='desc' (worst) means reverse=True (highest first)
            sorted_programs = sorted(all_progs, key=lambda p: p.fitness_scores.get("runtime_ms", float('inf')), reverse=(sort_order == "desc"))
        else:
            logger.warning(f"Unknown objective: {objective}. Defaulting to no specific sort order beyond Program ID.")
            return sorted(all_progs, key=lambda p: p.id)[:limit]

        logger.debug(f"Sorted {len(sorted_programs)} programs. Top 3 (if available): {[p.id for p in sorted_programs[:3]]}")
        return sorted_programs[:limit]

    async def get_programs_by_generation(self, generation: int) -> List[Program]:
        logger.debug(f"Retrieving programs for generation: {generation}")
        generation_programs = [p for p in self._programs.values() if p.generation == generation]
        logger.info(f"Found {len(generation_programs)} programs for generation {generation}.")
        return generation_programs

    async def get_programs_for_next_generation(self, task_id: str, generation_size: int) -> List[Program]:
        logger.info(f"Attempting to retrieve {generation_size} programs for next generation for task {task_id}.")
        all_relevant_progs = [p for p in self._programs.values() if getattr(p, 'task_id', None) == task_id or task_id is None]
        if not all_relevant_progs:
            logger.warning(f"No programs found for task {task_id} in database to select for next generation.")
            return []

        if len(all_relevant_progs) <= generation_size:
            logger.debug(f"Returning all {len(all_relevant_progs)} programs for task {task_id} as it's <= generation_size {generation_size}.")
            return all_relevant_progs
        
        import random
        selected_programs = random.sample(all_relevant_progs, generation_size)
        logger.info(f"Selected {len(selected_programs)} random programs for task {task_id} for next generation.")
        return selected_programs

    async def count_programs(self) -> int:
        count = len(self._programs)
        logger.debug(f"Total programs in database: {count}")
        return count

    async def clear_database(self) -> None:
        logger.info("Clearing all programs from database.")
        async with self._lock:
            self._programs.clear()
        await self._save_to_file() # Persist the empty state
        logger.info("Database cleared.")

    async def execute(self, *args, **kwargs) -> Any:
        logger.warning("InMemoryDatabaseAgent.execute() called, but this agent uses specific methods for DB operations.")
        raise NotImplementedError("InMemoryDatabaseAgent does not have a generic execute. Use specific methods like save_program, get_program etc.")

                                      
if __name__ == "__main__":
    import asyncio                                                    
    async def test_db():
        logging.basicConfig(level=logging.DEBUG)
        
        # Mock DATABASE_PATH for testing
        global DATABASE_PATH
        original_db_path = DATABASE_PATH
        DATABASE_PATH = "test_inmemory_agent.json"

        # Clean up previous test file
        if os.path.exists(DATABASE_PATH):
            os.remove(DATABASE_PATH)

        db = InMemoryDatabaseAgent()

        prog1_data = {"id":"prog_001", "code":"print('hello')", "generation":0, "fitness_scores":{"correctness": 0.8, "runtime_ms": 100}, "task_id": "test_task"}
        prog2_data = {"id":"prog_002", "code":"print('world')", "generation":0, "fitness_scores":{"correctness": 0.9, "runtime_ms": 50}, "task_id": "test_task"}
        prog3_data = {"id":"prog_003", "code":"print('test')", "generation":1, "fitness_scores":{"correctness": 0.85, "runtime_ms": 70}, "task_id": "test_task"}

        prog1 = Program(**prog1_data)
        prog2 = Program(**prog2_data)
        prog3 = Program(**prog3_data)

        await db.save_program(prog1)
        await db.save_program(prog2)
        await db.save_program(prog3)

        retrieved_prog = await db.get_program("prog_001")
        assert retrieved_prog is not None and retrieved_prog.code == "print('hello')"
        assert retrieved_prog.task_id == "test_task"

        all_programs = await db.get_all_programs()
        assert len(all_programs) == 3

        # Test loading from file by creating a new instance
        db2 = InMemoryDatabaseAgent()
        assert await db2.count_programs() == 3
        retrieved_prog2 = await db2.get_program("prog_002")
        assert retrieved_prog2 is not None and retrieved_prog2.fitness_scores.get("correctness") == 0.9

        best_correctness = await db.get_best_programs(task_id="test_task", limit=2, objective="correctness", sort_order="desc")
        print(f"Best by correctness (desc): {[p.id for p in best_correctness]}")
        assert len(best_correctness) == 2
        assert best_correctness[0].id == "prog_002"      
        assert best_correctness[1].id == "prog_003"       

        best_runtime_asc = await db.get_best_programs(task_id="test_task", limit=2, objective="runtime_ms", sort_order="asc")
        print(f"Best by runtime (asc): {[p.id for p in best_runtime_asc]}")
        assert len(best_runtime_asc) == 2
        assert best_runtime_asc[0].id == "prog_002"
        # Corrected assertion for runtime, prog3 (70ms) is better than prog1 (100ms) when ascending
        assert best_runtime_asc[1].id == "prog_003"
        
        next_gen_task_programs = await db.get_programs_for_next_generation(task_id="test_task", generation_size=2)
        assert len(next_gen_task_programs) == 2
        for p in next_gen_task_programs:
            assert p.task_id == "test_task"

        await db.clear_database()
        assert await db.count_programs() == 0
        assert not os.path.exists(DATABASE_PATH) or os.path.getsize(DATABASE_PATH) < 5 # empty json is like {} or []
        print("InMemoryDatabaseAgent with JSON persistence tests passed.")

        # Cleanup test file and restore
        if os.path.exists(DATABASE_PATH):
            os.remove(DATABASE_PATH)
        DATABASE_PATH = original_db_path

    asyncio.run(test_db())


# ========================================
# SECTION 6: EVALUATOR AGENT  
# ========================================
# EvaluatorAgent with Docker execution and testing

class EvaluatorAgent(EvaluatorAgentInterface, BaseAgent):
    def __init__(self, task_definition: Optional[TaskDefinition] = None):
        super().__init__()
        self.task_definition = task_definition
        self.evaluation_model_name = EVALUATION_MODEL
        self.evaluation_timeout_seconds = EVALUATION_TIMEOUT_SECONDS
        logger.info(f"EvaluatorAgent initialized with model: {self.evaluation_model_name}, timeout: {self.evaluation_timeout_seconds}s")
        if self.task_definition:
            logger.info(f"EvaluatorAgent task_definition: {self.task_definition.id}")

    def _check_syntax(self, code: str) -> List[str]:
        errors = []
        try:
            compile(code+"\n", "tmp.py", 'exec')
        except SyntaxError as e:
            errors.append(f"SyntaxError: {e.msg} at line {e.lineno}, offset {e.offset}")
        except Exception as e:
            errors.append(f"Unexpected error during syntax check: {str(e)}")
        return errors

    async def _execute_code_safely(
        self,
        code: str,
        task_for_examples: TaskDefinition,
        timeout_seconds: Optional[int] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        timeout = timeout_seconds if timeout_seconds is not None else self.evaluation_timeout_seconds
        results = {"test_outputs": [], "average_runtime_ms": 0.0}

        if not task_for_examples.input_output_examples:
            logger.warning("No input/output examples provided to _execute_code_safely.")
            return results, "No test cases to run."

        if not task_for_examples.function_name_to_evolve:
            logger.error(f"Task {task_for_examples.id} does not specify 'function_name_to_evolve'. Cannot execute code.")
            return None, "Task definition is missing 'function_name_to_evolve'."

        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, "temp_script.py")

        def serialize_arg(arg):
            if isinstance(arg, (float, int)) and (arg == float('inf') or arg == float('-inf') or arg != arg):
                return f"float('{str(arg)}')"
            return json.dumps(arg)


        test_cases_str = json.dumps(task_for_examples.input_output_examples)
        test_cases_str = test_cases_str.replace('"Infinity"', 'float("inf")')
        test_cases_str = test_cases_str.replace('"-Infinity"', 'float("-inf")')
        test_cases_str = test_cases_str.replace('"NaN"', 'float("nan")')

        test_cases_str = test_cases_str.replace('true', 'True').replace('false', 'False').replace('null', 'None')

        test_harness_code = f"""
import json
import time
import sys
import math  # Import math for inf/nan constants

# User's code (function to be tested)
{code}

# Test execution logic
results = []
total_execution_time = 0
num_tests = 0

# Special constants for test cases
Infinity = float('inf')
NaN = float('nan')

test_cases = {test_cases_str} 
function_to_test_name = "{task_for_examples.function_name_to_evolve}"

# Make sure the function_to_test is available in the global scope
if function_to_test_name not in globals():
    # Attempt to find it if it was defined inside a class (common for LLM output)
    # This is a simple heuristic and might need refinement.
    found_func = None
    for name, obj in list(globals().items()):
        if isinstance(obj, type):
            if hasattr(obj, function_to_test_name):
                method = getattr(obj, function_to_test_name)
                if callable(method):
                    globals()[function_to_test_name] = method
                    found_func = True
                    break
    if not found_func:
        print(json.dumps({{"error": f"Function '{{function_to_test_name}}' not found in the global scope or as a callable method of a defined class."}}))
        sys.exit(1)
        
function_to_test = globals()[function_to_test_name]

for i, test_case in enumerate(test_cases):
    input_args = test_case.get("input")
    
    start_time = time.perf_counter()
    try:
        if isinstance(input_args, list):
            actual_output = function_to_test(*input_args)
        elif isinstance(input_args, dict):
            actual_output = function_to_test(**input_args)
        elif input_args is None:
            actual_output = function_to_test()
        else:
            actual_output = function_to_test(input_args)
            
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        total_execution_time += execution_time_ms
        num_tests += 1
        results.append({{"test_case_id": i, "output": actual_output, "runtime_ms": execution_time_ms, "status": "success"}})
    except Exception as e:
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        error_output = {{
            "test_case_id": i,
            "error": str(e), 
            "error_type": type(e).__name__,
            "runtime_ms": execution_time_ms,
            "status": "error"
        }}
        try:
            json.dumps(error_output)
        except TypeError:
            error_output["error"] = "Unserializable error object"
        results.append(error_output)

final_output = {{"test_outputs": results}}
if num_tests > 0:
    final_output["average_runtime_ms"] = total_execution_time / num_tests

def custom_json_serializer(obj):
    if isinstance(obj, float):
        if obj == float('inf'):
            return 'Infinity'
        elif obj == float('-inf'):
            return '-Infinity'
        elif obj != obj:
            return 'NaN'
    raise TypeError(f"Object of type {{type(obj).__name__}} is not JSON serializable")

print(json.dumps(final_output, default=custom_json_serializer))
"""
        with open(temp_file_path, "w") as f:
            f.write(test_harness_code)

        # Generate a unique container name to manage it during timeouts
        container_name = f"evaluator-{task_for_examples.id}-{time.time_ns()}"
        
        # Docker command construction
        cmd = [
            "docker", "run",
            "--rm",
            "--name", container_name,
            "-i",
            # Conditionally disable network
            # "-v", f"{os.path.abspath(temp_dir)}:/app/user_code", # Ensure absolute path for temp_dir # This line will be part of the dynamic extension below
            "-w", "/app/user_code",
            # settings.DOCKER_IMAGE_NAME, # This line will be part of the dynamic extension below
            # "python", "temp_script.py" # This line will be part of the dynamic extension below
        ]

        if DOCKER_NETWORK_DISABLED:
            cmd.extend(["--network", "none"])
        
        # Add volume mount, image name, and script execution command
        cmd.extend([
            "-v", f"{os.path.abspath(temp_dir)}:/app/user_code",
            DOCKER_IMAGE_NAME,
            "python", "temp_script.py"
        ])

        proc = None
        try:
            logger.debug(f"Executing code in Docker: {' '.join(cmd)}")
            start_time = time.monotonic()
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            duration = time.monotonic() - start_time
            logger.debug(f"Docker execution finished in {duration:.2f}s. Exit code: {proc.returncode}")

            stdout_str = stdout.decode('utf-8', errors='replace').strip()
            stderr_str = stderr.decode('utf-8', errors='replace').strip()

            if proc.returncode != 0:
                # If stdout is empty and stderr has content, it's likely a Docker/script init error
                if not stdout_str and stderr_str:
                    error_message = f"Execution failed with exit code {proc.returncode}. Docker error: '{stderr_str}'"
                    logger.warning(error_message)
                    return None, error_message
                # If stdout has content, it might be a script error with traceback in stderr, but JSON in stdout.
                # Log a warning and proceed to parse stdout. If parsing fails, that error will be returned.
                logger.warning(f"Execution completed with non-zero exit code {proc.returncode}. Stdout: '{stdout_str}', Stderr: '{stderr_str}'. Attempting to parse stdout.")

            if not stdout_str and proc.returncode == 0: # Script exited cleanly but no output
                 logger.warning(f"Execution produced no stdout, but exited cleanly. Stderr: '{stderr_str}'")
                 return None, f"No output from script. Stderr: '{stderr_str}'"
            
            if not stdout_str and proc.returncode != 0: # No stdout and non-zero exit, means previous error message should be used
                 return None, f"Execution failed with exit code {proc.returncode}. No stdout. Stderr: '{stderr_str}'"


            try:
                def json_loads_with_infinity(s):
                    s = s.replace('"Infinity"', 'float("inf")')
                    s = s.replace('"-Infinity"', 'float("-inf")')
                    s = s.replace('"NaN"', 'float("nan")')
                    return json.loads(s)

                parsed_output = json_loads_with_infinity(stdout_str)
                logger.debug(f"Parsed execution output: {parsed_output}")
                return parsed_output, None
            except json.JSONDecodeError as e:
                error_message = f"Failed to decode JSON output: {e}. Raw output: '{stdout_str}'"
                logger.error(error_message)
                return None, error_message
            except Exception as e:
                error_message = f"Error processing script output: {e}. Raw output: '{stdout_str}'"
                logger.error(error_message)
                return None, error_message

        except asyncio.TimeoutError:
            logger.warning(f"Execution for container '{container_name}' initiating timeout handling.")
            if proc and proc.returncode is None: # Check if process is still running
                logger.info(f"Attempting to stop Docker container: {container_name}")
                stop_cmd = ["docker", "stop", container_name]
                try:
                    stop_proc = await asyncio.create_subprocess_exec(*stop_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                    _, stop_stderr_bytes = await asyncio.wait_for(stop_proc.communicate(), timeout=10) # 10s for docker stop
                    if stop_proc.returncode != 0:
                        logger.error(f"Failed to stop container {container_name}. Exit: {stop_proc.returncode}. Stderr: {stop_stderr_bytes.decode(errors='replace')}")
                        kill_cmd = ["docker", "kill", container_name]
                        kill_proc = await asyncio.create_subprocess_exec(*kill_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                        kill_stdout_bytes, kill_stderr_bytes = await asyncio.wait_for(kill_proc.communicate(), timeout=5) # 5s for docker kill
                        if kill_proc.returncode == 0:
                             logger.info(f"Successfully killed container {container_name} after stop failed.")
                        else:
                             logger.error(f"Failed to kill container {container_name}. Exit: {kill_proc.returncode}. Stderr: {kill_stderr_bytes.decode(errors='replace')}")
                    else:
                        logger.info(f"Successfully stopped container {container_name}.")
                except asyncio.TimeoutError:
                    logger.error(f"Timeout trying to stop/kill container {container_name}. It might be orphaned.")
                except Exception as e_stop:
                    logger.error(f"Error stopping/killing container {container_name}: {e_stop}")
            
            if proc: # Original docker run process
                try:
                    if proc.returncode is None: proc.kill()
                    await proc.wait() 
                except ProcessLookupError: pass
                except Exception as e_kill: logger.error(f"Error trying to kill original subprocess after docker stop/kill: {e_kill}")
            
            logger.warning(f"Code execution in Docker container '{container_name}' timed out after {timeout} seconds.")
            return None, f"Execution timed out after {timeout} seconds (container {container_name})."
        except Exception as e:
            logger.error(f"An unexpected error occurred during code execution: {e}", exc_info=True)
            return None, f"Unexpected execution error: {str(e)}"
        finally:
            try:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                if os.path.exists(temp_dir):
                    try:
                        # Attempt to remove the directory multiple times with a small delay
                        # This is a workaround for potential lingering locks from Docker
                        for _ in range(3):
                            try:
                                if os.path.exists(temp_file_path): os.remove(temp_file_path)
                                os.rmdir(temp_dir)
                                break # Succeeded
                            except OSError:
                                await asyncio.sleep(0.1) # Wait a bit and retry
                        else:
                            logger.error(f"Failed to remove temp_dir {temp_dir} after multiple retries.")
                    except Exception as e_rmdir: # Catch any other exception during rmdir attempts
                         logger.error(f"Error removing temp_dir {temp_dir}: {e_rmdir}.")
            except Exception as e_cleanup: # General cleanup exception
                logger.error(f"Error during cleanup of temp files: {e_cleanup}")

    def _assess_correctness(self, execution_results: Dict[str, Any], expected_outputs: List[Dict[str, Any]]) -> Tuple[float, int, int]:
        passed_tests = 0
        total_tests = len(expected_outputs)

        if not execution_results or "test_outputs" not in execution_results:
            logger.warning("Execution results are missing 'test_outputs' field.")
            return 0.0, 0, total_tests

        actual_test_outputs = execution_results["test_outputs"]

        if len(actual_test_outputs) != total_tests:
            logger.warning(f"Mismatch in number of test outputs ({len(actual_test_outputs)}) and expected outputs ({total_tests}). Some tests might have crashed before producing output.")

        for i, expected in enumerate(expected_outputs):
            actual_output_detail = next((res for res in actual_test_outputs if res.get("test_case_id") == i), None)

            if actual_output_detail and actual_output_detail.get("status") == "success":
                actual = actual_output_detail.get("output")

                logger.debug(f"Test case {i}: Actual output: {actual}")

                # Check if we have a validation function
                if "validation_func" in expected:
                    logger.debug(f"Test case {i}: Using validation function.")
                    try:
                        # Create a namespace for the validation function
                        namespace = {}
                        # Execute the validation function definition
                        exec(expected["validation_func"], namespace)
                        # Get the validate function
                        validate_func = namespace.get("validate")
                        if validate_func and callable(validate_func):
                            # Revert: Only pass the actual output to the validation function
                            if validate_func(actual):
                                passed_tests += 1
                                logger.debug(f"Test case {i}: Validation function returned True.")
                            else:
                                logger.debug(f"Test case {i}: Validation function returned False.")
                        else:
                            logger.warning(f"Validation function not found or not callable in test case {i}")
                    except Exception as e:
                        logger.error(f"Error executing validation function for test case {i}: {str(e)}", exc_info=True) # Log exception details
                # Check against expected output if provided
                elif "output" in expected:
                    expected_val = expected["output"]
                    logger.debug(f"Test case {i}: Comparing with expected output: {expected_val}")
                    if self._compare_outputs(actual, expected_val):
                        passed_tests += 1
                        logger.debug(f"Test case {i}: Comparison returned True.")
                    else:
                        logger.debug(f"Test case {i}: Comparison returned False.")
                else:
                    logger.warning(f"Test case {i} has neither validation function nor expected output")
            elif actual_output_detail:
                logger.debug(f"Test case {i} had error: {actual_output_detail.get('error')}")
            else:
                logger.debug(f"Test case {i}: No output found in results.")

        logger.debug(f"Finished assessing correctness. Passed tests: {passed_tests}/{total_tests}")
        if total_tests == 0:
            return 1.0, 0, 0

        correctness = passed_tests / total_tests
        return correctness, passed_tests, total_tests

    async def evaluate_program(self, program: Program, task: TaskDefinition) -> Program:
        logger.info(f"Evaluating program: {program.id} for task: {task.id}")
        program.status = "evaluating"
        program.errors = []
        program.fitness_scores = {"correctness": 0.0, "runtime_ms": float('inf'), "passed_tests": 0.0, "total_tests": 0.0}

        syntax_errors = self._check_syntax(program.code)
        if syntax_errors:
            program.errors.extend(syntax_errors)
            program.fitness_scores["correctness"] = 0.0
            program.status = "failed_evaluation"
            logger.warning(f"Syntax errors found in program {program.id}: {syntax_errors}")
            return program

        logger.debug(f"Syntax check passed for program {program.id}.")

        overall_passed_tests = 0
        overall_total_tests = 0
        last_successful_level_avg_runtime = float('inf')
        highest_level_passed = -1

        # Determine test sets to run
        test_groups_to_run = []
        if task.tests: # New structure with levels
            # Sort test groups by level, defaulting level to 0 if not specified
            sorted_test_groups = sorted(task.tests, key=lambda g: g.get('level', 0))
            for group in sorted_test_groups:
                test_groups_to_run.append({
                    "name": group.get('name', f"level_{group.get('level', 0)}"),
                    "level": group.get('level', 0),
                    "test_cases": group.get('test_cases', [])
                })
        elif task.input_output_examples: # Fallback for old structure
            logger.warning(f"Task {task.id} uses legacy 'input_output_examples'. Consider migrating to 'tests' with levels.")
            test_groups_to_run.append({
                "name": "default_level",
                "level": 0,
                "test_cases": task.input_output_examples
            })
        
        if not test_groups_to_run:
            logger.info(f"No tests or input/output examples provided for task {task.id}. Skipping execution.")
            program.fitness_scores["correctness"] = 0.5 # Default score if no tests
            program.fitness_scores["runtime_ms"] = 0.0
            program.status = "evaluated" # No tests to fail
            return program

        for group_idx, test_group in enumerate(test_groups_to_run):
            level_name = test_group['name']
            current_level = test_group['level']
            current_level_test_cases = test_group['test_cases']

            if not current_level_test_cases:
                logger.info(f"Test group '{level_name}' (Level {current_level}) has no test cases. Skipping.")
                continue

            logger.info(f"Executing program {program.id} against test group '{level_name}' (Level {current_level}) with {len(current_level_test_cases)} test cases.")
            
            # Create a temporary TaskDefinition subset for _execute_code_safely and _assess_correctness
            temp_task_def_for_level = TaskDefinition(
                id=f"{task.id}_level_{current_level}",
                description=task.description, # Not directly used by execution, but good to have
                function_name_to_evolve=task.function_name_to_evolve,
                input_output_examples=current_level_test_cases, # This is the key part
                allowed_imports=task.allowed_imports # Also important for execution context
            )

            execution_results, execution_error = await self._execute_code_safely(program.code, task_for_examples=temp_task_def_for_level)
            
            if execution_error:
                logger.warning(f"Execution error for program {program.id} at level {current_level} ('{level_name}'): {execution_error}")
                program.errors.append(f"Execution Error at Level {current_level} ('{level_name}'): {execution_error}")
                program.status = "failed_evaluation"
                break # Stop evaluation cascade
            
            if execution_results is None:
                logger.warning(f"No execution results for program {program.id} at level {current_level} ('{level_name}').")
                program.errors.append(f"Execution Error: No results at Level {current_level} ('{level_name}').")
                program.status = "failed_evaluation"
                break # Stop evaluation cascade

            level_correctness, level_passed_tests, level_total_tests = self._assess_correctness(execution_results, current_level_test_cases)
            
            overall_passed_tests += level_passed_tests
            overall_total_tests += level_total_tests

            current_level_avg_runtime = execution_results.get("average_runtime_ms", float('inf'))
            if not isinstance(current_level_avg_runtime, (float, int)):
                current_level_avg_runtime = float('inf')

            logger.info(f"Program {program.id} Level {current_level} ('{level_name}') Correctness: {level_correctness:.2f} ({level_passed_tests}/{level_total_tests}), Avg Runtime: {current_level_avg_runtime}ms")

            if level_correctness < 1.0:
                error_msg = f"Failed {level_total_tests - level_passed_tests} of {level_total_tests} tests at Level {current_level} ('{level_name}')."
                program.errors.append(error_msg)
                program.status = "failed_evaluation"
                # Update overall fitness scores with results up to this failing level
                program.fitness_scores["correctness"] = overall_passed_tests / overall_total_tests if overall_total_tests > 0 else 0.0
                program.fitness_scores["passed_tests"] = float(overall_passed_tests)
                program.fitness_scores["total_tests"] = float(overall_total_tests)
                # Runtime is tricky here. Could be avg of last successful, or sum. Let's keep last successful level's.
                program.fitness_scores["runtime_ms"] = last_successful_level_avg_runtime 
                break # Stop evaluation cascade
            else:
                highest_level_passed = current_level
                last_successful_level_avg_runtime = current_level_avg_runtime
                # If this is the last group and all passed, program status will be evaluated.
                if group_idx == len(test_groups_to_run) - 1:
                    program.status = "evaluated" 
        
        # Final fitness score calculation after loop (if not broken early)
        if overall_total_tests > 0:
            program.fitness_scores["correctness"] = overall_passed_tests / overall_total_tests
        elif not program.errors: # No tests run, but no syntax errors either
            program.fitness_scores["correctness"] = 0.5 # Default for no tests executed successfully
        # else correctness remains 0.0 from initialization if errors occurred before any tests
        
        program.fitness_scores["passed_tests"] = float(overall_passed_tests)
        program.fitness_scores["total_tests"] = float(overall_total_tests)
        program.fitness_scores["runtime_ms"] = last_successful_level_avg_runtime if highest_level_passed != -1 else float('inf')
        program.fitness_scores["highest_level_passed"] = float(highest_level_passed)

        # Consolidate status based on errors and correctness
        if program.errors:
            program.status = "failed_evaluation"
        elif program.fitness_scores.get("correctness", 0.0) == 1.0 and overall_total_tests > 0:
            program.status = "evaluated"
        elif overall_total_tests == 0 and not program.errors: # No tests were run (e.g. all groups empty), no syntax errors
             program.status = "evaluated" # Considered evaluated as there was nothing to fail
             program.fitness_scores["correctness"] = 0.5 # Re-affirm default
             program.fitness_scores["runtime_ms"] = 0.0
        elif not program.errors : # Some tests run, but not 100% correct, and no specific execution errors
            program.status = "failed_evaluation"
            if f"Achieved {program.fitness_scores['correctness']*100:.0f}% correctness but not all tests passed." not in program.errors:
                 program.errors.append(f"Achieved {program.fitness_scores['correctness']*100:.0f}% correctness but not all tests passed.")

        logger.info(f"Overall evaluation complete for program {program.id}. Status: {program.status}, Fitness: {program.fitness_scores}")
        return program

    async def execute(self, program: Program, task: TaskDefinition) -> Program:
        return await self.evaluate_program(program, task)

    def _compare_outputs(self, actual: Any, expected: Any) -> bool:
        logger.debug(f"Comparing outputs. Actual: {type(actual)}{actual}, Expected: {type(expected)}{expected}")
        
        if isinstance(actual, float) and isinstance(expected, float):
            TOLERANCE = 1e-9 # This could also be made configurable via settings.py later.
            is_close = math.isclose(actual, expected, rel_tol=TOLERANCE, abs_tol=TOLERANCE)
            if not is_close:
                logger.debug(f"Float comparison: {actual} vs {expected} is NOT close (tolerance: {TOLERANCE}).")
            return is_close
        
        # Fallback to direct equality for other types
        are_equal = actual == expected

        return are_equal


# ========================================
# SECTION 7: PROMPT DESIGNER AGENT
# ========================================
# PromptDesignerAgent for initial, mutation, and bug-fix prompts

class PromptDesignerAgent(PromptDesignerInterface, BaseAgent):
    def __init__(self, task_definition: TaskDefinition):
        super().__init__()
        self.task_definition = task_definition
        logger.info(f"PromptDesignerAgent initialized for task: {self.task_definition.id}")

    def design_initial_prompt(self) -> str:
        logger.info(f"Designing initial prompt for task: {self.task_definition.id}")
                                                           
        expert_knowledge_section = ""
        if self.task_definition.expert_knowledge:
            expert_knowledge_section = f"Relevant Expert Knowledge or Context:\\n{self.task_definition.expert_knowledge}\\n\\n"

        prompt = (
            f"You are an expert Python programmer. Your task is to write a Python function based on the following specifications.\\n\\n"
            f"Task Description: {self.task_definition.description}\\n\\n"
            f"{expert_knowledge_section}"
            f"Function to Implement: `{self.task_definition.function_name_to_evolve}`\\n\\n"
            f"Input/Output Examples:\n"
                                         
            f"{self._format_input_output_examples()}\n\n"
            f"Evaluation Criteria: {self.task_definition.evaluation_criteria}\n\n"
            f"Allowed Standard Library Imports: {self.task_definition.allowed_imports}. Do not use any other external libraries or packages.\n\n"
            f"Your Response Format:\n"
            f"Please provide *only* the complete Python code for the function `{self.task_definition.function_name_to_evolve}`. "
            f"The code should be self-contained or rely only on the allowed imports. "
            f"Do not include any surrounding text, explanations, comments outside the function, or markdown code fences (like ```python or ```)."
        )
        logger.debug(f"Designed initial prompt:\n--PROMPT START--\n{prompt}\n--PROMPT END--")
        return prompt

    def _format_input_output_examples(self) -> str:
        """Format input/output examples for the prompt."""
        examples = []
        
        # YAML format with tests
        if self.task_definition.tests:
            for test_group in self.task_definition.tests:
                for test_case in test_group['test_cases']:
                    if 'output' in test_case:
                        examples.append(f"Input: {test_case['input']}\nOutput: {test_case['output']}")
                    elif 'validation_func' in test_case:
                        examples.append(f"Input: {test_case['input']}\nValidation: {test_case['validation_func']}")
        
        # legacy format with input_output_examples
        elif self.task_definition.input_output_examples:
            for example in self.task_definition.input_output_examples:
                input_str = str(example.get('input'))
                output_str = str(example.get('output'))
                examples.append(f"Input: {input_str}\nOutput: {output_str}")
        
        if not examples:
            return "No input/output examples provided."
        
        return "\n\n".join(examples)

    def _format_evaluation_feedback(self, program: Program, evaluation_feedback: Optional[Dict[str, Any]]) -> str:
        if not evaluation_feedback:
            return "No detailed evaluation feedback is available for the previous version of this code. Attempt a general improvement or refinement."

        correctness = evaluation_feedback.get("correctness_score", None)
        runtime = evaluation_feedback.get("runtime_ms", None)
        errors = evaluation_feedback.get("errors", [])                          
                                                                                               
        stderr = evaluation_feedback.get("stderr", None)

        feedback_parts = []
        if correctness is not None:
            feedback_parts.append(f"- Correctness Score: {correctness*100:.2f}%")
        if runtime is not None:
            feedback_parts.append(f"- Runtime: {runtime:.2f} ms")
        
        if errors:
            error_messages = "\n".join([f"  - {e}" for e in errors])
            feedback_parts.append(f"- Errors Encountered During Evaluation:\n{error_messages}")
        elif stderr:
            feedback_parts.append(f"- Standard Error Output During Execution:\n{stderr}")
        elif correctness is not None and correctness < 1.0:
            feedback_parts.append("- The code did not achieve 100% correctness but produced no explicit errors or stderr. Review logic for test case failures.")
        elif correctness == 1.0:
            feedback_parts.append("- The code achieved 100% correctness. Consider optimizing for efficiency or exploring alternative correct solutions.")
        
        if not feedback_parts:
             return "The previous version was evaluated, but no specific feedback details were captured. Try a general improvement."

        return "Summary of the previous version's evaluation:\n" + "\n".join(feedback_parts)

    def design_mutation_prompt(self, program: Program, evaluation_feedback: Optional[Dict[str, Any]] = None) -> str:
        logger.info(f"Designing mutation prompt for program: {program.id} (Generation: {program.generation})")
        logger.debug(f"Parent program code (to be mutated):\\n{program.code}")
        
        expert_knowledge_section = ""
        if self.task_definition.expert_knowledge:
            expert_knowledge_section = f"Relevant Expert Knowledge or Context (applies to the overall task):\\n{self.task_definition.expert_knowledge}\\n\\n"

        feedback_summary = self._format_evaluation_feedback(program, evaluation_feedback)
        logger.debug(f"Formatted evaluation feedback for prompt:\n{feedback_summary}")

        diff_instructions = (
            "Your Response Format:\n"
            "Propose improvements to the 'Current Code' below by providing your changes as a sequence of diff blocks. "
            "Each diff block must follow this exact format:\n"
            "<<<<<<< SEARCH\n"
            "# Exact original code lines to be found and replaced\n"
            "=======\n"
            "# New code lines to replace the original\n"
            ">>>>>>> REPLACE\n\n"
            "- The SEARCH block must be an *exact* segment from the 'Current Code'. Do not paraphrase or shorten it."
            "- If you are adding new code where nothing existed, the SEARCH block can be a comment indicating the location, or an adjacent existing line."
            "- If you are deleting code, the REPLACE block should be empty."
            "- Provide all suggested changes as one or more such diff blocks. Do not include any other text, explanations, or markdown outside these blocks."
        )

        prompt = (
            f"You are an expert Python programmer. Your task is to improve an existing Python function based on its previous performance and the overall goal.\\n\\n"
            f"Overall Task Description: {self.task_definition.description}\\n\\n"
            f"{expert_knowledge_section}"
            f"Function to Improve: `{self.task_definition.function_name_to_evolve}`\\n\\n"
            f"Allowed Standard Library Imports: {self.task_definition.allowed_imports}. Do not use other external libraries or packages.\n\n"
            f"Current Code (Version from Generation {program.generation}):\n"
            f"```python\n{program.code}\n```\n\n"
            f"Evaluation Feedback on the 'Current Code':\n{feedback_summary}\n\n"
            f"Your Improvement Goal:\n"
            f"Based on the task, the 'Current Code', and its 'Evaluation Feedback', your goal is to propose modifications to improve the function `{self.task_definition.function_name_to_evolve}`. "
            f"Prioritize fixing any errors or correctness issues. If correct, focus on improving efficiency or exploring alternative robust logic. "
            f"Consider the original evaluation criteria: {self.task_definition.evaluation_criteria}\n\n"
            f"{diff_instructions}"
        )
        logger.debug(f"Designed mutation prompt (requesting diff):\n--PROMPT START--\n{prompt}\n--PROMPT END--")
        return prompt

    def design_bug_fix_prompt(self, program: Program, error_message: str, execution_output: Optional[str] = None) -> str:
        logger.info(f"Designing bug-fix prompt for program: {program.id} (Generation: {program.generation})")
        logger.debug(f"Buggy program code:\\n{program.code}")
        
        expert_knowledge_section = ""
        if self.task_definition.expert_knowledge:
            expert_knowledge_section = f"Relevant Expert Knowledge or Context (applies to the overall task):\\n{self.task_definition.expert_knowledge}\\n\\n"

        logger.debug(f"Primary error message: {error_message}")
        if execution_output:
            logger.debug(f"Additional execution output (stdout/stderr): {execution_output}")

        output_segment = f"Execution Output (stdout/stderr that might be relevant):\n{execution_output}\n" if execution_output else "No detailed execution output was captured beyond the error message itself.\n"
        
        diff_instructions = (
            "Your Response Format:\n"
            "Propose fixes to the 'Buggy Code' below by providing your changes as a sequence of diff blocks. "
            "Each diff block must follow this exact format:\n"
            "<<<<<<< SEARCH\n"
            "# Exact original code lines to be found and replaced\n"
            "=======\n"
            "# New code lines to replace the original\n"
            ">>>>>>> REPLACE\n\n"
            "- The SEARCH block must be an *exact* segment from the 'Buggy Code'."
            "- Provide all suggested changes as one or more such diff blocks. Do not include any other text, explanations, or markdown outside these blocks."
        )

        prompt = (
            f"You are an expert Python programmer. Your task is to fix a bug in an existing Python function.\\n\\n"
            f"Overall Task Description: {self.task_definition.description}\\n\\n"
            f"{expert_knowledge_section}"
            f"Function to Fix: `{self.task_definition.function_name_to_evolve}`\\n\\n"
            f"Allowed Standard Library Imports: {self.task_definition.allowed_imports}. Do not use other external libraries or packages.\n\n"
            f"Buggy Code (Version from Generation {program.generation}):\n"
            f"```python\n{program.code}\n```\n\n"
            f"Error Encountered: {error_message}\n"
            f"{output_segment}\n"
            f"Your Goal:\n"
            f"Analyze the 'Buggy Code', the 'Error Encountered', and any 'Execution Output' to identify and fix the bug(s). "
            f"The corrected function must adhere to the overall task description and allowed imports.\n\n"
            f"{diff_instructions}"
        )
        logger.debug(f"Designed bug-fix prompt (requesting diff):\n--PROMPT START--\n{prompt}\n--PROMPT END--")
        return prompt

    async def execute(self, *args, **kwargs) -> Any:
        raise NotImplementedError("PromptDesignerAgent.execute() is not the primary way to use this agent. Call specific design methods.")

                
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    sample_task_def = TaskDefinition(
        id="task_001_designer_test",
        description="Create a Python function `sum_list(numbers)` that returns the sum of a list of integers. Handle empty lists by returning 0.",
        function_name_to_evolve="sum_list",
        input_output_examples=[
            {"input": [1, 2, 3], "output": 6},
            {"input": [], "output": 0}
        ],
        evaluation_criteria="Ensure correctness for all cases, including empty lists.",
        allowed_imports=["math"]
    )
    designer = PromptDesignerAgent(task_definition=sample_task_def)

    print("--- Initial Prompt ---")
    initial_prompt = designer.design_initial_prompt()
    print(initial_prompt)

    sample_program_mutation = Program(
        id="prog_mut_001",
        code="def sum_list(numbers):\n  # Slightly off logic\n  s = 0\n  for n in numbers:\n    s += n\n  return s if numbers else 1",                     
        fitness_scores={"correctness_score": 0.5, "runtime_ms": 5.0},
        generation=1,
        errors=["Test case failed: Input [], Expected 0, Got 1"],
        status="evaluated"
    )
    mutation_feedback = {
        "correctness_score": sample_program_mutation.fitness_scores["correctness_score"],
        "runtime_ms": sample_program_mutation.fitness_scores["runtime_ms"],
        "errors": sample_program_mutation.errors,
        "stderr": None
    }
    print("\n--- Mutation Prompt (Requesting Diff) ---")
    mutation_prompt = designer.design_mutation_prompt(sample_program_mutation, evaluation_feedback=mutation_feedback)
    print(mutation_prompt)

    sample_program_buggy = Program(
        id="prog_bug_002",
        code="def sum_list(numbers):\n  # Buggy implementation causing TypeError\n  if not numbers:\n    return 0\n  return sum(numbers) + \"oops\"",
        fitness_scores={"correctness_score": 0.0, "runtime_ms": 2.0},
        generation=2,
        errors=["TypeError: unsupported operand type(s) for +: 'int' and 'str'"],
        status="evaluated"
    )
    print("\n--- Bug-Fix Prompt (Requesting Diff) ---")
    bug_fix_prompt = designer.design_bug_fix_prompt(sample_program_buggy, error_message=sample_program_buggy.errors[0], execution_output="TypeError occurred during summation.")
    print(bug_fix_prompt)


# ========================================
# SECTION 8: SELECTION CONTROLLER AGENT
# ========================================
# Island class and SelectionControllerAgent for evolutionary algorithms

class Island:
    def __init__(self, island_id: int, initial_programs: Optional[List[Program]] = None):
        self.island_id = island_id
        self.programs = initial_programs or []
        self.generation = 0  # Island's internal generation counter
        self.best_fitness = 0.0
        self.last_improvement_generation = 0
        
        if DEBUG:
            logger.debug(f"Initializing Island {island_id} with {len(self.programs)} programs")
        
        for program in self.programs:
            program.island_id = island_id
            if len(self.programs) > 1 and (program.generation is None or program.generation == 0):
                program.generation = self.generation  # Island's current gen (0 for new)
                if DEBUG:
                    logger.debug(f"Set generation for program {program.id} to {self.generation}")

    def get_best_program(self) -> Optional[Program]:
        if not self.programs:
            return None
        # Sort by correctness (higher is better), runtime (lower is better), generation (lower/older is better), and creation time (older is better)
        best_program = max(
            self.programs,
            key=lambda p: (
                p.fitness_scores.get("correctness", 0.0),  # Higher correctness preferred
                -p.fitness_scores.get("runtime_ms", float('inf')),  # Lower runtime preferred
                -p.generation,  # Older generation preferred
                -p.created_at  # Older creation time preferred as tiebreaker
            )
        )
        if DEBUG:
            logger.debug(f"Island {self.island_id} best program: ID={best_program.id}, "
                        f"Correctness={best_program.fitness_scores.get('correctness')}, "
                        f"Runtime={best_program.fitness_scores.get('runtime_ms')}, "
                        f"Generation={best_program.generation}")
        return best_program

    def update_metrics(self):
        best_program = self.get_best_program()
        if best_program:
            current_best = best_program.fitness_scores.get("correctness", 0.0)
            if current_best > self.best_fitness:
                self.best_fitness = current_best
                self.last_improvement_generation = self.generation
                if DEBUG:
                    logger.debug(f"Island {self.island_id} new best fitness: {self.best_fitness} "
                               f"at generation {self.generation}")
        self.generation += 1
        if DEBUG:
            logger.debug(f"Island {self.island_id} generation incremented to {self.generation}")

class SelectionControllerAgent(SelectionControllerInterface, BaseAgent):
    def __init__(self):
        super().__init__()
        self.elitism_count = ELITISM_COUNT
        self.num_islands = NUM_ISLANDS
        self.migration_interval = MIGRATION_FREQUENCY
        self.islands: Dict[int, Island] = {}
        self.current_generation = 0
        logger.info(f"SelectionControllerAgent initialized with {self.num_islands} islands and elitism_count: {self.elitism_count}")

    def initialize_islands(self, initial_programs: List[Program]) -> None:
        """Initialize islands with the initial population."""
        programs_per_island = len(initial_programs) // self.num_islands
        if DEBUG:
            logger.debug(f"Initializing {self.num_islands} islands with {programs_per_island} programs each")
        
        for i in range(self.num_islands):
            start_idx = i * programs_per_island
            end_idx = start_idx + programs_per_island if i < self.num_islands - 1 else len(initial_programs)
            island_programs = initial_programs[start_idx:end_idx]
            self.islands[i] = Island(i, island_programs)
            if DEBUG:
                logger.debug(f"Initialized Island {i} with {len(island_programs)} programs")

    def select_parents(self, population: List[Program], num_parents: int) -> List[Program]:
        # The 'population' argument here is the global population, but selection is per-island driven.
        logger.debug(f"Starting parent selection. Global population size: {len(population)}, Num parents to select: {num_parents}")
        
        if num_parents == 0:
            logger.info("Number of parents to select is 0. Returning empty list.")
            return []

        all_potential_parents: List[Program] = []
        parents_per_island = max(1, num_parents // self.num_islands) # Ensure at least 1 parent per island if possible
        remaining_parents_to_select = num_parents

        # Prioritize elites from all islands first
        all_elites = []
        for island_id, island in self.islands.items():
            if not island.programs:
                logger.warning(f"Island {island_id} is empty. Skipping for elite selection.")
                continue

            sorted_island_programs = sorted(
                island.programs,
                key=lambda p: (p.fitness_scores.get("correctness", 0.0), -p.fitness_scores.get("runtime_ms", float('inf')), -p.generation),
                reverse=True
            )
            for i in range(min(len(sorted_island_programs), self.elitism_count)):
                 all_elites.append(sorted_island_programs[i])
        
        # Deduplicate elites (in case of migration having same elite in multiple islands)
        unique_elites = []
        seen_elite_ids = set()
        for elite in sorted(all_elites, key=lambda p: (p.fitness_scores.get("correctness", 0.0), -p.fitness_scores.get("runtime_ms", float('inf')), -p.generation), reverse=True):
            if elite.id not in seen_elite_ids:
                unique_elites.append(elite)
                seen_elite_ids.add(elite.id)
        
        # Add top N unique elites directly to parents
        selected_parents = unique_elites[:min(num_parents, len(unique_elites))]
        remaining_parents_to_select = num_parents - len(selected_parents)
        parent_ids_so_far = {p.id for p in selected_parents}

        if DEBUG:
            logger.debug(f"Selected {len(selected_parents)} elite parents globally: {[p.id for p in selected_parents]}")

        if remaining_parents_to_select <= 0:
            return selected_parents

        # Then fill remaining slots using roulette wheel selection from each island proportionally
        # Collect all non-elite candidates from all islands
        all_roulette_candidates_with_island = []
        for island_id, island in self.islands.items():
            if not island.programs:
                continue
            
            # Sort island programs to pick non-elites for roulette
            sorted_island_programs = sorted(
                island.programs,
                key=lambda p: (p.fitness_scores.get("correctness", 0.0), -p.fitness_scores.get("runtime_ms", float('inf')), -p.generation),
                reverse=True
            )
            for program in sorted_island_programs:
                if program.id not in parent_ids_so_far: # Check if not already selected as an elite
                    all_roulette_candidates_with_island.append(program)
        
        # Deduplicate candidates for roulette (if a program migrated and is not an elite)
        unique_roulette_candidates = []
        seen_roulette_ids = set()
        for cand in all_roulette_candidates_with_island:
            if cand.id not in parent_ids_so_far and cand.id not in seen_roulette_ids:
                 unique_roulette_candidates.append(cand)
                 seen_roulette_ids.add(cand.id)

        if not unique_roulette_candidates:
            logger.warning("No more unique candidates for roulette selection.")
            return selected_parents

        # Perform roulette wheel selection on the combined pool of unique candidates
        total_fitness_roulette = sum(p.fitness_scores.get("correctness", 0.0) + 0.0001 for p in unique_roulette_candidates) # Add small constant to allow selection of 0 fitness programs

        if total_fitness_roulette <= 0.0001 * len(unique_roulette_candidates): # All candidates have near zero fitness
            num_to_select_randomly = min(remaining_parents_to_select, len(unique_roulette_candidates))
            random_parents_from_roulette = random.sample(unique_roulette_candidates, num_to_select_randomly)
            selected_parents.extend(random_parents_from_roulette)
            logger.debug(f"Selected {len(random_parents_from_roulette)} random parents due to low/zero total fitness in roulette pool.")
        else:
            for _ in range(remaining_parents_to_select):
                if not unique_roulette_candidates: break
                pick = random.uniform(0, total_fitness_roulette)
                current_sum = 0
                chosen_parent = None
                for i, program in enumerate(unique_roulette_candidates):
                    current_sum += (program.fitness_scores.get("correctness", 0.0) + 0.0001)
                    if current_sum >= pick:
                        chosen_parent = program
                        unique_roulette_candidates.pop(i) # Remove chosen parent from further selection
                        total_fitness_roulette -= (chosen_parent.fitness_scores.get("correctness", 0.0) + 0.0001) # Adjust total fitness
                        break
                
                if chosen_parent:
                    selected_parents.append(chosen_parent)
                    parent_ids_so_far.add(chosen_parent.id) # Track for debugging or future use
                    logger.debug(f"Selected parent via global roulette: {chosen_parent.id} from island {chosen_parent.island_id} with correctness {chosen_parent.fitness_scores.get('correctness')}")
                elif unique_roulette_candidates: # Fallback if pick logic fails (should not happen with correct total_fitness adjustment)
                    fallback_parent = random.choice(unique_roulette_candidates)
                    selected_parents.append(fallback_parent)
                    unique_roulette_candidates.remove(fallback_parent)
                    parent_ids_so_far.add(fallback_parent.id)
                    logger.debug(f"Selected fallback parent via global roulette: {fallback_parent.id}")
        
        logger.info(f"Selected {len(selected_parents)} parents in total.")
        return selected_parents

    def select_survivors(self, current_population: List[Program], offspring_population: List[Program], population_size: int) -> List[Program]:
        """
        Select survivors for each island, combining current island members with their offspring.
        as island.programs is the source of truth for each island's current members.
        """
        if DEBUG:
            logger.debug(f"Starting survivor selection. Offspring pop: {len(offspring_population)}, Target pop size: {population_size}")
        
        # Update island metrics
        for island in self.islands.values():
            island.update_metrics()

        # Check if it's time for migration
        if self.current_generation % self.migration_interval == 0:
            if DEBUG:
                logger.debug(f"Generation {self.current_generation}: Performing migration")
            self._perform_migration()

        self.current_generation += 1
        if DEBUG:
            logger.debug(f"Generation incremented to {self.current_generation}")

        # Select survivors within each island
        all_survivors = []
        programs_per_island = population_size // self.num_islands

        for island_id, island in self.islands.items():
            if DEBUG:
                logger.debug(f"Processing Island {island_id} for survivor selection")
            
            # Get current island members
            current_island_members = island.programs
            
            # Filter offspring belonging to this island
            newly_generated_for_this_island = [
                p for p in offspring_population if p.island_id == island_id
            ]
            
            if DEBUG:
                logger.debug(f"Island {island_id}: {len(current_island_members)} current members, "
                           f"{len(newly_generated_for_this_island)} new offspring")
            
            combined_population = current_island_members + newly_generated_for_this_island
            if not combined_population:
                island.programs = []  # Island becomes empty
                if DEBUG:
                    logger.debug(f"Island {island_id} became empty")
                continue

            # Sort by correctness (higher is better), runtime (lower is better), and generation (lower/older is better)
            sorted_combined = sorted(
                combined_population,
                key=lambda p: (
                    p.fitness_scores.get("correctness", 0.0),  # Higher correctness preferred
                    -p.fitness_scores.get("runtime_ms", float('inf')),  # Lower runtime preferred
                    -p.generation  # Older generation preferred
                ),
                reverse=True
            )

            survivors = []
            seen_program_ids = set()
            for program in sorted_combined:
                if len(survivors) < programs_per_island:
                    if program.id not in seen_program_ids:
                        survivors.append(program)
                        seen_program_ids.add(program.id)
                        if DEBUG:
                            logger.debug(f"Island {island_id} selected survivor: {program.id} "
                                       f"with correctness {program.fitness_scores.get('correctness')}")
                else:
                    break

            island.programs = survivors
            all_survivors.extend(survivors)
            if DEBUG:
                logger.debug(f"Island {island_id} final survivor count: {len(survivors)}")

        return all_survivors

    def _perform_migration(self) -> None:
        """Perform migration between islands."""
        if DEBUG:
            logger.debug("Starting migration process")
        
        # Identify underperforming islands
        island_performances = [(island_id, island.get_best_program().fitness_scores.get("correctness", 0.0) if island.get_best_program() else 0.0) 
                             for island_id, island in self.islands.items()]
        sorted_islands = sorted(island_performances, key=lambda x: x[1])
        
        # Select the worst performing half of islands
        num_islands_to_reseed = self.num_islands // 2
        underperforming_islands = [island_id for island_id, _ in sorted_islands[:num_islands_to_reseed]]
        surviving_islands = [island_id for island_id, _ in sorted_islands[num_islands_to_reseed:]]
        
        if DEBUG:
            logger.debug(f"Identified {len(underperforming_islands)} underperforming islands: {underperforming_islands}")
            logger.debug(f"Identified {len(surviving_islands)} surviving islands: {surviving_islands}")
        
        # Get the best programs from surviving islands
        for underperforming_island_id in underperforming_islands:
            if not surviving_islands: # Should not happen if num_islands > 1
                logger.warning("No surviving islands to donate for migration.")
                break

            # Select a random surviving island
            donor_island_id = random.choice(surviving_islands)
            donor_island = self.islands[donor_island_id]
            recipient_island = self.islands[underperforming_island_id]
            
            # Get the best program from the donor island
            best_program_from_donor = donor_island.get_best_program()
            if best_program_from_donor:
                # Create a deep copy or a new Program instance to avoid shared object issues if necessary,
                # especially if the program object might be modified later independently by islands.
                # For now, assuming Program objects are relatively immutable post-evaluation or that sharing is acceptable.
                migrant_program = best_program_from_donor # Potentially clone this: copy.deepcopy(best_program_from_donor)
                migrant_program.island_id = underperforming_island_id # Assign to new island
                # Reset generation if it's a pure re-seed, or keep if it's a true 'migration' maintaining age.
                # For this less destructive migration, let's assume it keeps its generation from the donor.

                # Add to recipient island's program list, ensuring no duplicates by ID
                if not any(p.id == migrant_program.id for p in recipient_island.programs):
                    recipient_island.programs.append(migrant_program)
                    if DEBUG:
                        logger.debug(f"Migrated program {migrant_program.id} (Correctness: {migrant_program.fitness_scores.get('correctness')}) "
                                   f"from island {donor_island_id} to island {underperforming_island_id}. "
                                   f"Recipient island size now: {len(recipient_island.programs)}")
                else:
                    if DEBUG:
                        logger.debug(f"Program {migrant_program.id} from donor island {donor_island_id} already exists in recipient {underperforming_island_id}. Skipped migration of this specific program.")
            else:
                if DEBUG:
                    logger.debug(f"Donor island {donor_island_id} had no best program to migrate.")

    async def execute(self, action: str, **kwargs) -> Any:
        # This method is part of the BaseAgent interface.
        # Specific actions like initialize_islands, select_parents, select_survivors
        # are called directly. If other generic async actions are needed for
        # SelectionControllerAgent in the future, they can be dispatched here.
        logger.warning(f"SelectionControllerAgent.execute called with action '{action}', but most actions are handled by specific methods.")
        if action == "initialize_islands_async_placeholder": # Example if an async version was needed
            # await self.async_initialize_islands(kwargs['initial_programs'])
            pass
        raise NotImplementedError(f"The generic execute method is not fully implemented for specific action '{action}' in SelectionControllerAgent. Use direct methods.")

                
if __name__ == '__main__':
    import uuid
    import random
    logging.basicConfig(level=logging.DEBUG)
    selector = SelectionControllerAgent()

    # Create test programs with proper attributes
    programs = [
        Program(
            id=str(uuid.uuid4()),
            code="c1",
            fitness_scores={"correctness": 0.9, "runtime_ms": 100},
            status="evaluated",
            generation=0
        ),
        Program(
            id=str(uuid.uuid4()),
            code="c2",
            fitness_scores={"correctness": 1.0, "runtime_ms": 50},
            status="evaluated",
            generation=0
        ),
        Program(
            id=str(uuid.uuid4()),
            code="c3",
            fitness_scores={"correctness": 0.7, "runtime_ms": 200},
            status="evaluated",
            generation=0
        ),
        Program(
            id=str(uuid.uuid4()),
            code="c4",
            fitness_scores={"correctness": 1.0, "runtime_ms": 60},
            status="evaluated",
            generation=0
        ),
        Program(
            id=str(uuid.uuid4()),
            code="c5",
            fitness_scores={"correctness": 0.5},
            status="evaluated",
            generation=0
        ),
        Program(
            id=str(uuid.uuid4()),
            code="c6",
            status="unevaluated",
            generation=0
        ),
    ]

    # Initialize islands
    selector.initialize_islands(programs)
    print("\n--- Initial Island Distribution ---")
    for island_id, island in selector.islands.items():
        print(f"Island {island_id}: {len(island.programs)} programs")
        for p in island.programs:
            print(f"  Program {p.id}: Gen={p.generation}, Correctness={p.fitness_scores.get('correctness')}, Runtime={p.fitness_scores.get('runtime_ms')}")

    print("\n--- Testing Parent Selection ---")
    parents = selector.select_parents(programs, num_parents=3)
    for p in parents:
        print(f"Selected Parent: {p.id}, Island: {p.island_id}, Gen: {p.generation}, Correctness: {p.fitness_scores.get('correctness')}, Runtime: {p.fitness_scores.get('runtime_ms')}")

    print("\n--- Testing Survivor Selection ---")
    current_pop = programs[:2]
    offspring_pop = [
        Program(
            id=str(uuid.uuid4()),
            code="off1",
            fitness_scores={"correctness": 1.0, "runtime_ms": 40},
            status="evaluated",
            generation=1,
            island_id=0  # Simulate offspring from island 0
        ),
        Program(
            id=str(uuid.uuid4()),
            code="off2",
            fitness_scores={"correctness": 0.6, "runtime_ms": 10},
            status="evaluated",
            generation=1,
            island_id=1 
        ),
    ]
    
    for gen in range(3):
        print(f"\n--- Generation {gen} ---")
        survivors = selector.select_survivors(current_pop, offspring_pop, population_size=2)
        print(f"Survivors after generation {gen}:")
        for s in survivors:
            print(f"  Survivor: {s.id}, Island: {s.island_id}, Gen: {s.generation}, Correctness: {s.fitness_scores.get('correctness')}, Runtime: {s.fitness_scores.get('runtime_ms')}")
        
        current_pop = survivors

        offspring_pop = [
            Program(
                id=str(uuid.uuid4()),
                code=f"off{gen}_{i}",
                fitness_scores={"correctness": random.uniform(0.5, 1.0), "runtime_ms": random.randint(10, 200)},
                status="evaluated",
                generation=gen + 2,
                island_id=i % selector.num_islands
            )
            for i in range(2)
        ]


# ========================================
# SECTION 9: TASK MANAGER AGENT
# ========================================
# TaskManagerAgent orchestrating the evolutionary cycle

class TaskManagerAgent(TaskManagerInterface):
    def __init__(self, task_definition: TaskDefinition, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.task_definition = task_definition
        self.prompt_designer: PromptDesignerInterface = PromptDesignerAgent(task_definition=self.task_definition)
        self.code_generator: CodeGeneratorInterface = CodeGeneratorAgent()
        self.evaluator: EvaluatorAgentInterface = EvaluatorAgent(task_definition=self.task_definition)
        
        self.database: DatabaseAgentInterface = InMemoryDatabaseAgent()
        logger.info(f"Using {DATABASE_TYPE} database (InMemoryDatabaseAgent).")
            
        self.selection_controller: SelectionControllerInterface = SelectionControllerAgent()

        self.population_size = POPULATION_SIZE
        self.num_generations = GENERATIONS
        self.num_parents_to_select = self.population_size // 2
        self.num_islands = NUM_ISLANDS
        self.programs_per_island = self.population_size // self.num_islands

    async def initialize_population(self) -> List[Program]:
        logger.info(f"Initializing population for task: {self.task_definition.id}")
        initial_population = []
        
        initial_model = LLM_SECONDARY_MODEL # Use secondary for broad initial generation
        logger.info(f"Using model '{initial_model}' for initial population generation.")

        tasks = []
        for i in range(self.population_size):
            initial_prompt = self.prompt_designer.design_initial_prompt()
            tasks.append(self.code_generator.generate_code(initial_prompt, model_name=initial_model, temperature=0.8))

        generated_codes = await asyncio.gather(*tasks)
        for i, generated_code in enumerate(generated_codes):
            program_id = f"{self.task_definition.id}_gen0_prog{i}"
            logger.debug(f"Generated initial program {i+1}/{self.population_size} with id {program_id}")
            program = Program(
                id=program_id,
                code=generated_code,
                generation=0,
                status="unevaluated"
            )
            initial_population.append(program)
            await self.database.save_program(program)

        self.selection_controller.initialize_islands(initial_programs=initial_population)
        
        logger.info(f"Initialized population with {len(initial_population)} programs across {self.num_islands} islands.")
        return initial_population

    async def evaluate_population(self, population: List[Program]) -> List[Program]:
        logger.info(f"Evaluating population of {len(population)} programs.")
        evaluated_programs = []
        evaluation_tasks = [self.evaluator.evaluate_program(prog, self.task_definition) for prog in population if prog.status != "evaluated"]
        
        results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            original_program = population[i]
            if isinstance(result, Exception):
                logger.error(f"Error evaluating program {original_program.id}: {result}", exc_info=result)
                original_program.status = "failed_evaluation"
                original_program.errors.append(str(result))
                evaluated_programs.append(original_program)
            else:
                evaluated_programs.append(result)
            await self.database.save_program(evaluated_programs[-1])
            
        logger.info(f"Finished evaluating population. {len(evaluated_programs)} programs processed.")
        return evaluated_programs

    async def manage_evolutionary_cycle(self):
        logger.info(f"Starting evolutionary cycle for task: {self.task_definition.description[:50]}...")
        current_population = await self.initialize_population()
        current_population = await self.evaluate_population(current_population)

        for gen in range(1, self.num_generations + 1):
            logger.info(f"--- Generation {gen}/{self.num_generations} ---")

            parents = self.selection_controller.select_parents(current_population, self.num_parents_to_select)
            if not parents:
                logger.warning(f"Generation {gen}: No parents selected. Ending evolution early.")
                break
            logger.info(f"Generation {gen}: Selected {len(parents)} parents.")

            offspring_population = []
            num_offspring_per_parent = (self.population_size + len(parents) - 1) // len(parents)
            
            generation_tasks = []
            for i, parent in enumerate(parents):
                for j in range(num_offspring_per_parent):
                    child_id = f"{self.task_definition.id}_gen{gen}_child{i}_{j}"
                    generation_tasks.append(self.generate_offspring(parent, gen, child_id))
            
            generated_offspring_results = await asyncio.gather(*generation_tasks, return_exceptions=True)

            for result in generated_offspring_results:
                if isinstance(result, Exception):
                    logger.error(f"Error generating offspring: {result}", exc_info=result)
                elif result:
                    offspring_population.append(result)
                    await self.database.save_program(result)

            logger.info(f"Generation {gen}: Generated {len(offspring_population)} offspring.")
            if not offspring_population:
                logger.warning(f"Generation {gen}: No offspring generated. May indicate issues with LLM or prompting.")
                if not parents:
                    break

            offspring_population = await self.evaluate_population(offspring_population)

            current_population = self.selection_controller.select_survivors(current_population, offspring_population, self.population_size)
            logger.info(f"Generation {gen}: New population size: {len(current_population)}.")

            best_program_this_gen = sorted(
                current_population,
                key=lambda p: (p.fitness_scores.get("correctness", -1), -p.fitness_scores.get("runtime_ms", float('inf'))),
                reverse=True
            )
            if best_program_this_gen:
                logger.info(f"Generation {gen}: Best program: ID={best_program_this_gen[0].id}, Fitness={best_program_this_gen[0].fitness_scores}")
            else:
                logger.warning(f"Generation {gen}: No programs in current population after survival selection.")
                break

        logger.info("Evolutionary cycle completed.")
        final_best = await self.database.get_best_programs(task_id=self.task_definition.id, limit=1, objective="correctness")
        if final_best:
            logger.info(f"Overall Best Program: {final_best[0].id}, Code:\n{final_best[0].code}\nFitness: {final_best[0].fitness_scores}")
        else:
            logger.info("No best program found at the end of evolution.")
        return final_best

    async def generate_offspring(self, parent: Program, generation_num: int, child_id: str) -> Optional[Program]:
        logger.debug(f"Generating offspring from parent {parent.id} for generation {generation_num}")
        
        prompt_type = "mutation"
        chosen_model = LLM_PRIMARY_MODEL 

        if parent.errors and parent.fitness_scores.get("correctness", 1.0) < BUG_FIX_CORRECTNESS_THRESHOLD:
            primary_error = parent.errors[0]
            execution_details = None
            if len(parent.errors) > 1 and isinstance(parent.errors[1], str) and ("stdout" in parent.errors[1].lower() or "stderr" in parent.errors[1].lower()):
                execution_details = parent.errors[1]
            
            mutation_prompt = self.prompt_designer.design_bug_fix_prompt(
                program=parent,
                error_message=primary_error,
                execution_output=execution_details
            )
            logger.info(f"Attempting bug fix for parent {parent.id} using diff. Model: {chosen_model}. Error: {primary_error}")
            prompt_type = "bug_fix"
        else:
            if parent.fitness_scores.get("correctness", 0.0) >= HIGH_FITNESS_THRESHOLD_FOR_PRIMARY_LLM:
                logger.info(f"Parent {parent.id} has high fitness, using primary model {chosen_model} for mutation.")
            else:
                chosen_model = LLM_SECONDARY_MODEL # Use secondary for lower-fitness mutations
                logger.info(f"Parent {parent.id} has lower fitness, using secondary model {chosen_model} for mutation.")

            feedback = {
                "errors": parent.errors,
                "correctness_score": parent.fitness_scores.get("correctness"),
                "runtime_ms": parent.fitness_scores.get("runtime_ms")
            }
            feedback = {k: v for k, v in feedback.items() if v is not None}

            mutation_prompt = self.prompt_designer.design_mutation_prompt(program=parent, evaluation_feedback=feedback)
            logger.info(f"Attempting mutation for parent {parent.id} using diff. Model: {chosen_model}")
        
        generated_code = await self.code_generator.execute(
            prompt=mutation_prompt,
            model_name=chosen_model,
            temperature=0.75,
            output_format="diff",
            parent_code_for_diff=parent.code
        )

        if not generated_code.strip():
            logger.warning(f"Offspring generation for parent {parent.id} ({prompt_type}) resulted in empty code/diff. Skipping.")
            return None
        
        if generated_code == parent.code:
            logger.warning(f"Offspring generation for parent {parent.id} ({prompt_type}) using diff resulted in no change to the code. Skipping.")
            return None
        
        if "<<<<<<< SEARCH" in generated_code and "=======" in generated_code and ">>>>>>> REPLACE" in generated_code:
            logger.warning(f"Offspring generation for parent {parent.id} ({prompt_type}) seems to have returned raw diff. LLM or diff application may have failed. Skipping. Content:\n{generated_code[:500]}")
            return None
        
        if "# Error:" in generated_code[:100]:
            logger.warning(f"Failed to generate valid code for offspring of {parent.id} ({prompt_type}). LLM Output indicates error: {generated_code[:200]}")
            return None

        offspring = Program(
            id=child_id,
            code=generated_code,
            generation=generation_num,
            parent_id=parent.id,
            island_id=parent.island_id,  # Inherit island ID from parent
            status="unevaluated"
        )
        logger.info(f"Successfully generated offspring {offspring.id} from parent {parent.id} ({prompt_type}).")
        return offspring

    async def execute(self) -> Any:
        return await self.manage_evolutionary_cycle()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    task_manager = TaskManagerAgent(task_definition=sample_task)

    sample_task = TaskDefinition(
        id="sum_list_task_001",
        description="Write a Python function called `solve(numbers)` that takes a list of integers `numbers` and returns their sum. The function should handle empty lists correctly by returning 0.",
        input_output_examples=[
            {"input": [1, 2, 3], "output": 6},
            {"input": [], "output": 0},
            {"input": [-1, 0, 1], "output": 0},
            {"input": [10, 20, 30, 40, 50], "output": 150}
        ],
        evaluation_criteria={"target_metric": "correctness", "goal": "maximize"},
        initial_code_prompt = "Please provide a Python function `solve(numbers)` that sums a list of integers. Handle empty lists by returning 0."
    )
    
    task_manager.num_generations = 3
    task_manager.population_size = 5
    task_manager.num_parents_to_select = 2

    async def run_task():
        try:
            best_programs = await task_manager.manage_evolutionary_cycle()
            if best_programs:
                print(f"\n*** Evolution Complete! Best program found: ***")
                print(f"ID: {best_programs[0].id}")
                print(f"Generation: {best_programs[0].generation}")
                print(f"Fitness: {best_programs[0].fitness_scores}")
                print(f"Code:\n{best_programs[0].code}")
            else:
                print("\n*** Evolution Complete! No suitable program was found. ***")
        except Exception as e:
            logger.error("An error occurred during the task management cycle.", exc_info=True)

    asyncio.run(run_task())


# ========================================
# SECTION 10: TESTS AND UTILITIES
# ========================================
# Test functions, sample tasks, and utility classes

class Rotator:
    def __init__(self, api_keys, **kwargs):
        if not api_keys or not all(isinstance(k, str) and k for k in api_keys):
            raise ValueError("API keys list cannot be empty and must contain non-empty strings.")
        self.api_keys = api_keys
        self.current_key_index = 0
        self.lock = threading.Lock()

    def get_key(self):
        with self.lock:
            return self.api_keys[self.current_key_index]

    def rotate(self):
        with self.lock:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            return self.api_keys[self.current_key_index]
                                                                                                   