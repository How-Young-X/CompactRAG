"""
AskCorpus - Generate QA pairs using vLLM llama8b model

This module processes corpus data and generates question-answer pairs
using the local vLLM llama8b model instead of external APIs.
"""

import argparse
import asyncio
import json
import jsonlines
import logging
import re
import time
import os
from pathlib import Path
from tqdm.asyncio import tqdm
from openai import OpenAI

import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from prompt.GenerateQA import GENERATEQA

# --------------------------
# Logging config
# --------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)

# --------------------------
# vLLM Configuration
# --------------------------
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_API_KEY = "EMPTY"  # vLLM doesn't require API key validation
MODEL_NAME = "llama8b"  # The served model name

def get_vllm_client():
    """Get vLLM OpenAI-compatible client."""
    try:
        client = OpenAI(
            api_key=VLLM_API_KEY,
            base_url=VLLM_BASE_URL,
            timeout=60  # Longer timeout for vLLM
        )
        return client
    except Exception as e:
        logging.error(f"Failed to initialize vLLM client: {e}")
        raise


# --------------------------
# Helper functions
# --------------------------
def validate_qa_pairs(qa_list):
    """Validate that the extracted JSON contains valid QA pairs."""
    if not isinstance(qa_list, list):
        return False
    
    for item in qa_list:
        if not isinstance(item, dict):
            return False
        if 'question' not in item or 'answer' not in item:
            return False
        if not isinstance(item['question'], str) or not isinstance(item['answer'], str):
            return False
        if not item['question'].strip() or not item['answer'].strip():
            return False
    
    return True

def extract_json(text):
    """Extract JSON list from model output with multiple fallback strategies."""
    if not text or not isinstance(text, str):
        logging.warning("Empty or invalid text input")
        return None
    
    # Strategy 1: Look for JSON array in markdown code block
    try:
        # Match ```json [...] ``` or ``` [...] ```
        code_block_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', text, re.DOTALL)
        if code_block_match:
            json_str = code_block_match.group(1).strip()
            result = json.loads(json_str)
            if isinstance(result, list) and validate_qa_pairs(result):
                logging.info(f"Successfully extracted JSON from code block: {len(result)} items")
                return result
    except Exception as e:
        logging.debug(f"Code block extraction failed: {e}")
    
    # Strategy 2: Look for outermost JSON array
    try:
        # Find the first complete JSON array
        bracket_count = 0
        start_idx = -1
        
        for i, char in enumerate(text):
            if char == '[':
                if bracket_count == 0:
                    start_idx = i
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0 and start_idx != -1:
                    json_str = text[start_idx:i+1]
                    result = json.loads(json_str)
                    if isinstance(result, list) and validate_qa_pairs(result):
                        logging.info(f"Successfully extracted JSON array: {len(result)} items")
                        return result
    except Exception as e:
        logging.debug(f"Array extraction failed: {e}")
    
    # Strategy 3: Look for any JSON object that might be a list
    try:
        # Try to find any valid JSON in the text, but only arrays
        json_patterns = [
            r'\[.*?\]',  # Array pattern only
        ]
        
        for pattern in json_patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    json_str = match.group(0)
                    result = json.loads(json_str)
                    if isinstance(result, list) and validate_qa_pairs(result):
                        logging.info(f"Successfully extracted JSON with pattern matching: {len(result)} items")
                        return result
                except:
                    continue
    except Exception as e:
        logging.debug(f"Pattern matching failed: {e}")
    
    # Strategy 4: Try to clean and parse the entire text
    try:
        # Remove common prefixes/suffixes and try to parse
        cleaned_text = text.strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "Here are the generated questions and answers:",
            "Generated questions and answers:",
            "Questions and answers:",
            "Here's the JSON:",
            "JSON:",
            "```json",
            "```",
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned_text.startswith(prefix):
                cleaned_text = cleaned_text[len(prefix):].strip()
        
        # Remove common suffixes
        suffixes_to_remove = [
            "```",
            "End of JSON",
            "That's all the questions and answers.",
        ]
        
        for suffix in suffixes_to_remove:
            if cleaned_text.endswith(suffix):
                cleaned_text = cleaned_text[:-len(suffix)].strip()
        
        # Try to parse the cleaned text
        result = json.loads(cleaned_text)
        if isinstance(result, list) and validate_qa_pairs(result):
            logging.info(f"Successfully extracted JSON after cleaning: {len(result)} items")
            return result
            
    except Exception as e:
        logging.debug(f"Cleaned text parsing failed: {e}")
    
    # If all strategies fail, log the raw text for debugging
    logging.warning(f"Failed to extract JSON from text. Raw text (first 500 chars): {text[:500]}")
    return None


async def call_vllm_with_retry(prompt_, max_retries=3, delay=2):
    """Call vLLM model with retries."""
    for attempt in range(max_retries):
        try:
            client = get_vllm_client()
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_},
                ],
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"vLLM call failed, attempt {attempt+1}/{max_retries}, error: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(delay)
            continue
    return None


MAX_RETRIES = 5   # Maximum retry count per passage


async def process_passage(line, queue, writer, failed_writer, pbar):
    """Process one passage. If failed, requeue until MAX_RETRIES reached."""
    retry_count = line.get("_retry_count", 0)
    passage = line.get("title", "") + "\n" + line.get("passage", "")
    input_ = GENERATEQA.format(chunk=passage)

    # Log processing attempt
    logging.debug(f"Processing passage (retry {retry_count}): {passage[:100]}...")

    llm_res = await call_vllm_with_retry(input_)
    if llm_res is None:
        if retry_count < MAX_RETRIES:
            line["_retry_count"] = retry_count + 1
            logging.warning(f"LLM call failed, requeuing record (retry {retry_count+1}/{MAX_RETRIES})")
            await queue.put(line)  # requeue
        else:
            logging.error(f"Max retries reached for passage, skipping record. Title: {line.get('title', 'Unknown')}")
            failed_writer.write(line)  # save to failed.jsonl
            pbar.update(1)  # mark as "done" to not block progress
        return False

    # Log successful LLM response
    logging.debug(f"LLM response received (length: {len(llm_res)})")

    json_result = extract_json(llm_res)
    if json_result is None:
        if retry_count < MAX_RETRIES:
            line["_retry_count"] = retry_count + 1
            line["_last_response"] = llm_res[:500]  # Store part of response for debugging
            logging.warning(f"JSON extraction failed, requeuing record (retry {retry_count+1}/{MAX_RETRIES})")
            await queue.put(line)  # requeue
        else:
            logging.error(f"JSON extraction failed after max retries. Title: {line.get('title', 'Unknown')}")
            line["_last_response"] = llm_res[:500]  # Store for debugging
            failed_writer.write(line)  # save to failed.jsonl
            pbar.update(1)
        return False

    # Success! Clean up and save
    line["qa"] = json_result
    if "_retry_count" in line:
        del line["_retry_count"]  # clean metadata
    if "_last_response" in line:
        del line["_last_response"]  # clean metadata
    
    writer.write(line)
    pbar.update(1)
    logging.info(f"Successfully processed passage: {line.get('title', 'Unknown')} -> {len(json_result)} QA pairs")
    return True


async def worker(queue, writer, failed_writer, pbar):
    """Worker coroutine: process items from the queue until empty."""
    while True:
        try:
            line = await asyncio.wait_for(queue.get(), timeout=5)
        except asyncio.TimeoutError:
            break  # stop if queue is empty for too long

        await process_passage(line, queue, writer, failed_writer, pbar)
        queue.task_done()


async def main(args):
    dataset_name = args.dataset
    corpus_file = args.corpus
    workers = args.workers
    start_index = args.start

    # Create data/QA directory structure
    qa_base_dir = Path("data/QA")
    dataset_dir = qa_base_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Input and output paths
    input_path = f"data/sampled/{corpus_file}"
    output_path = dataset_dir / f"llama8b-{dataset_name}-qa.jsonl"
    failed_path = dataset_dir / f"llama8b-{dataset_name}-qa_failed.jsonl"
    
    logging.info(f"Input: {input_path}")
    logging.info(f"Output: {output_path}")
    logging.info(f"Failed: {failed_path}")

    queue = asyncio.Queue()

    # Load data streamingly into queue (skip until start_index)
    total = 0
    with jsonlines.open(input_path, "r") as f:
        for idx, line in enumerate(f):
            if idx < start_index:
                continue  # 跳过前 start_index 条
            await queue.put(line)
            total += 1

    with jsonlines.open(output_path, "a") as writer, \
         jsonlines.open(failed_path, "a") as failed_writer:

        # Process with progress bar
        with tqdm(total=total, desc=f"Processing {dataset_name} passages", unit="passage") as pbar:
            tasks = [
                asyncio.create_task(worker(queue, writer, failed_writer, pbar))
                for _ in range(workers)
            ]
            await asyncio.gather(*tasks)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate QA pairs using vLLM llama8b model from sampled corpus files")
    parser.add_argument("--corpus", type=str, default="musique_sample_corpus.jsonl",
                        choices=["musique_sample_corpus.jsonl", "2wiki_sample_corpus.jsonl", "hotpotqa_sample_corpus.jsonl"],
                        help="Input corpus file from data/sampled/ directory")
    parser.add_argument("--dataset", type=str, default="musique", 
                        choices=["musique","2wiki","hotpotqa"],
                        help="Dataset name for output organization in data/QA/")
    parser.add_argument("--workers", type=int, default=4, 
                        help="Number of parallel workers (recommended: 2-4 for vLLM)")
    parser.add_argument("--start", type=int, default=0, 
                        help="Start index in dataset (for resuming)")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1",
                        help="vLLM server URL")
    args = parser.parse_args()

    # Update vLLM URL if provided
    if args.vllm_url != VLLM_BASE_URL:
        # Update the global variable
        import sys
        current_module = sys.modules[__name__]
        current_module.VLLM_BASE_URL = args.vllm_url
        logging.info(f"Using vLLM URL: {args.vllm_url}")

    logging.info(f"Starting QA generation for {args.dataset} dataset")
    logging.info(f"Input corpus: {args.corpus}")
    logging.info(f"Using vLLM model: {MODEL_NAME}")
    logging.info(f"Workers: {args.workers}")
    
    asyncio.run(main(args))
