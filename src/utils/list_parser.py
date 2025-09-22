import json
import re

def parse_model_output(output: str):
    """
    Parse the model's output and extract the list of sub-questions.
    
    Args:
        output (str): Raw string returned by the model.
    
    Returns:
        list[str]: List of sub-questions. Returns an empty list if parsing fails.
    """
    if not output or not isinstance(output, str):
        return []

    try:
        # Step 1: Try direct JSON parse
        return json.loads(output)
    except Exception:
        pass

    try:
        # Step 2: Extract JSON-like block inside triple backticks if present
        match = re.search(r"```json\s*(\[.*?\])\s*```", output, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except Exception:
        pass

    try:
        # Step 3: Extract first [...] JSON-like structure
        match = re.search(r"(\[.*\])", output, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except Exception:
        pass

    # Step 4: Fallback – attempt to clean and split manually
    try:
        # Extract quoted strings manually
        sub_questions = re.findall(r'"(.*?)"', output)
        return sub_questions if sub_questions else []
    except Exception:
        return []