import json
import re

def extract_model_direct_answer(output):
    try:
        data = json.loads(output)
        return data.get("answer")
    
    except json.JSONDecodeError:
        try:
            json_match = re.search(r'\{.*?\}', output, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                return data.get("answer")
            
            answer_match = re.search(r'"answer":\s*"([^"]+)"', output)
            if answer_match:
                return answer_match.group(1)
            
            return None
            
        except (json.JSONDecodeError, TypeError):
            return None


def extract_model_cot_answer(output):
    try:

        data = json.loads(output)
        return data.get("answer")
    
    except json.JSONDecodeError:
        try:
            json_match = re.search(r'\{.*?\}', output, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                return data.get("answer")
            
            answer_match = re.search(r'"answer":\s*"([^"]+)"', output)
            
            if answer_match:
                return answer_match.group(1)
            
            return None
            
        except (json.JSONDecodeError, TypeError):
            return None
def extract_model_cot_thought(output):
    try:

        data = json.loads(output)
        return data.get("thought")
    
    except json.JSONDecodeError:
        try:
            json_match = re.search(r'\{.*?\}', output, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                return data.get("thought")
            
            thought_match = re.search(r'"thought":\s*"([^"]+)"', output)
            
            if thought_match:
                return thought_match.group(1)
            
            return None
            
        except (json.JSONDecodeError, TypeError):
            return None

def extract_json_from_llm_response(llm_output: str):
    """
    Extract the first JSON object from an LLM response string.
    Handles trailing commas and unescaped quotes.
    """
    stack = []
    start_idx = None
    for idx, char in enumerate(llm_output):
        if char == '{':
            if not stack:
                start_idx = idx
            stack.append('{')
        elif char == '}':
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    json_str = llm_output[start_idx:idx+1]
                    # Try to clean common issues
                    json_str = json_str.replace('\n', ' ').replace('\t', ' ')
                    json_str = remove_trailing_commas(json_str)
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        # Fallback: escape inner quotes inside values
                        json_str_escaped = escape_inner_quotes(json_str)
                        try:
                            return json.loads(json_str_escaped)
                        except json.JSONDecodeError as e:
                            print(f"JSON parsing failed: {e}")
                            return None
    print("No JSON object found.")
    return None

def remove_trailing_commas(s: str) -> str:
    # Remove trailing commas before } or ]
    import re
    s = re.sub(r',\s*(\}|])', r'\1', s)
    return s

def escape_inner_quotes(s: str) -> str:
    # Escape quotes inside JSON string values (very naive)
    import re
    def replacer(match):
        inner = match.group(1)
        inner_escaped = inner.replace('"', '\\"')
        return f'"{inner_escaped}"'
    s = re.sub(r'"(.*?)"', replacer, s)
    return s
