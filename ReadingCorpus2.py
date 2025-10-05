import json
import re
import os
import time
from openai import OpenAI
import spacy
from collections import defaultdict

# --------- vLLM OpenAI API Client Configuration ---------
client = OpenAI(
    api_key="EMPTY",  # vLLM doesn't need a real API key
    base_url="http://localhost:8000/v1"
)
model_name = "llama8b"  # Corresponds to served-model-name in start_vllm_server.sh

# --------- SpaCy Model Initialization ---------
try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ SpaCy model loaded successfully")
except OSError:
    print("⚠️  SpaCy English model not installed, please run the following commands:")
    print("   1. pip install spacy")
    print("   2. python -m spacy download en_core_web_sm")
    print("   Or install a larger model: python -m spacy download en_core_web_trf")
    nlp = None

# --------- Enhanced Prompt (Based on Entities and Relations) ---------
ENHANCED_STAGE_PROMPT = """
You are a knowledge extraction and question generation system. You will receive:
1. Original passage text
2. Extracted entities and relationships from the passage

Your task is to generate atomic knowledge facts and corresponding QA pairs.

Output: A single JSON object only (nothing else) enclosed in a ```json ... ``` code block with exactly two keys:
  - "atomic_facts": an array of atomic knowledge facts (each fact should be independent, complete, and self-contained)
  - "qa": an array of objects {{"question": "...", "answer": "..." }}

Rules for "atomic_facts":
1. Each fact should be an independent, complete statement that can stand alone
2. Cover both explicit relationships (mentioned in entities/relations) and implicit relationships in the text
3. Include background knowledge and context that helps understand the entities
4. Each fact should be concise but informative (preferably one sentence)
5. Do not duplicate facts or add information not present in the passage

Rules for "qa":
1. Generate questions that test understanding of the atomic facts
2. Each question must be short (≤ 12 words), start with a question word (Who, What, When, Where, Which, How, How many)
3. Use explicit entity names from the entities list, avoid pronouns or vague references
4. Each answer must be an exact verbatim substring from the original passage
5. Ensure good coverage of all important entities and relationships
6. Avoid duplicate questions or answers

Example:

[Original Text]:
Lilli's Marriage (German: Lillis Ehe) is a 1919 German silent film directed by Jaap Speyer. It is a sequel to the film "Lilli", and premiered at the Marmorhaus in Berlin. The film's art direction was by Hans Dreier.

[Entity List]:
- Lilli's Marriage (WORK_OF_ART)
- Lillis Ehe (WORK_OF_ART) 
- Jaap Speyer (PERSON)
- Lilli (WORK_OF_ART)
- Marmorhaus in Berlin (FAC)
- Hans Dreier (PERSON)
- 1919 (DATE)

Output JSON:
```json
{{
  "atomic_facts": [
    "Lilli's Marriage is a 1919 German silent film",
    "Lilli's Marriage is also known as Lillis Ehe in German",
    "Jaap Speyer directed Lilli's Marriage",
    "Lilli's Marriage is a sequel to the film Lilli",
    "Lilli's Marriage premiered at the Marmorhaus in Berlin",
    "Hans Dreier was responsible for the art direction of Lilli's Marriage",
    "Lilli's Marriage was released in 1919"
  ],
  "qa": [
    {{"question": "What is Lilli's Marriage?", "answer": "Lilli's Marriage (German: Lillis Ehe) is a 1919 German silent film"}},
    {{"question": "Who directed Lilli's Marriage?", "answer": "directed by Jaap Speyer"}},
    {{"question": "Which film is Lilli's Marriage a sequel to?", "answer": "It is a sequel to the film \"Lilli\""}},
    {{"question": "Where did Lilli's Marriage premiere?", "answer": "premiered at the Marmorhaus in Berlin"}},
    {{"question": "Who was responsible for the art direction of Lilli's Marriage?", "answer": "The film's art direction was by Hans Dreier"}}
  ]
}}
```

Now process the following passage:

[Original Text]:
{passage}

[Entity List]:
{entity_info}
"""

# --------- SpaCy Entity Recognition and Relation Extraction ---------
def extract_entities_and_relations(text):
    """
    Use SpaCy to extract entities and basic relations
    Returns entity list and triple list
    """
    if nlp is None:
        return [], []
    
    doc = nlp(text)
    
    # Extract named entities
    entities = []
    entity_types = defaultdict(list)
    
    for ent in doc.ents:
        entity_info = {
            "text": ent.text.strip(),
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        }
        entities.append(entity_info)
        entity_types[ent.label_].append(ent.text.strip())
    
    # Extract noun phrases (supplement entities)
    noun_chunks = []
    for chunk in doc.noun_chunks:
        if chunk.text.strip() not in [e["text"] for e in entities]:  # Avoid duplicates
            noun_chunks.append(chunk.text.strip())
    
    # Extract triple relations based on dependency tree
    relations = []
    for token in doc:
        if token.dep_ in ["nsubj", "nsubjpass"]:  # Subject
            subject = token.text
            predicate = token.head.text if token.head else ""
            # Find object
            for child in token.head.children:
                if child.dep_ in ["dobj", "pobj", "attr"]:  # Direct object, prepositional object, attribute
                    object_ = child.text
                    relations.append({
                        "subject": subject,
                        "predicate": predicate,
                        "object": object_,
                        "relation_type": "SVO"
                    })
    
    return {
        "entities": entities,
        "entity_types": dict(entity_types),
        "noun_chunks": noun_chunks,
        "relations": relations
    }

def format_entities_for_prompt(extracted_info):
    """
    Format entity information for LLM prompt
    """
    if not extracted_info["entities"]:
        return "No clear entities identified"
    
    entity_text = "[Entity List]:\n"
    for entity in extracted_info["entities"]:
        entity_text += f"- {entity['text']} ({entity['label']})\n"
    
    if extracted_info["noun_chunks"]:
        entity_text += "\n[Important Noun Phrases]:\n"
        for chunk in extracted_info["noun_chunks"][:5]:  # Limit quantity
            entity_text += f"- {chunk}\n"
    
    if extracted_info["relations"]:
        entity_text += "\n[Basic Relations]:\n"
        for rel in extracted_info["relations"][:3]:  # Limit quantity
            entity_text += f"- {rel['subject']} → {rel['predicate']} → {rel['object']}\n"
    
    return entity_text

# --------- Helper Functions, Local Rules and Post-processing ---------
BAD_WORDS = [
# Vague pronouns that make questions unclear
"it","this","that","they","those","these",
# Vague references to objects without names
"the film","this film","the movie","this movie","the book","this book","the article","this article","the passage","this passage","the story","this story",
"the director","this director","the person","this person","the place","this place","the country","this country","the year","this year",
# Standalone pronouns at the start of questions
"^it ","^this ","^that ","^he ","^she ","^his ","^her "
]

BAD_WORDS_LOWER = [w.lower() for w in BAD_WORDS]

def has_bad_word(s):
    s_low = s.lower()
    
    # Check for regex patterns (those starting with ^)
    for pattern in BAD_WORDS_LOWER:
        if pattern.startswith('^'):
            # Remove the ^ and check if the pattern matches at the start
            regex_pattern = pattern[1:]
            if re.match(regex_pattern, s_low):
                return True
        else:
            # Regular substring check
            if pattern in s_low:
                return True
    
    return False

def coverage_check(required_phrases, qa_list):
    questions = [q["question"] for q in qa_list]
    missing = [p for p in required_phrases if not any(p in q for q in questions)]
    return missing

def call_llm(prompt, max_new_tokens=1024, max_retries=3):
    """
    Call vLLM API to generate text, return generation results and token statistics
    Includes retry mechanism to ensure stability
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=0.0,
                stream=False
            )
            generated_text = response.choices[0].message.content
            # Usage fields may exist depending on server
            input_tokens = getattr(response.usage, "prompt_tokens", 0) if hasattr(response, "usage") else response.usage.prompt_tokens if hasattr(response, "usage") else 0
            output_tokens = getattr(response.usage, "completion_tokens", 0) if hasattr(response, "usage") else response.usage.completion_tokens if hasattr(response, "usage") else 0
            total_tokens = getattr(response.usage, "total_tokens", 0) if hasattr(response, "usage") else response.usage.total_tokens if hasattr(response, "usage") else 0

            return {
                "text": generated_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            }
        except Exception as e:
            print(f"⚠️  API call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print("🔄 Waiting 2 seconds before retry...")
                time.sleep(2)
            else:
                print("❌ Reached maximum retry attempts, returning empty result")
                return {
                    "text": "",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0
                }

def extract_json_block(text):
    """
    Extract JSON from various formats that LLM might output
    """
    if not text:
        return ""
    
    # Try multiple extraction patterns
    patterns = [
        r"```json\s*(.*?)\s*```",  # Standard markdown code block
        r"```\s*(.*?)\s*```",      # Generic code block
        r"```\s*\{.*?\}\s*```",    # JSON in code block
        r"```\s*\[.*?\]\s*```",    # Array in code block
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.S | re.I)
        if match:
            return match.group(1).strip()
    
    # If no code block found, try to find JSON-like content
    # Look for the first complete JSON object or array
    json_start = None
    brace_count = 0
    bracket_count = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\':
            escape_next = True
            continue
            
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
            
        if not in_string:
            if char == '{':
                if json_start is None:
                    json_start = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
            elif char == '[':
                if json_start is None:
                    json_start = i
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
            
            # If we found a complete JSON object/array
            if json_start is not None and brace_count == 0 and bracket_count == 0:
                return text[json_start:i+1].strip()
    
    # Fallback: try to find any JSON-like structure
    start = None
    for i, ch in enumerate(text):
        if ch in ['{', '[']:
            start = i
            break
    
    if start is not None:
        return text[start:].strip()
    
    return text.strip()

def robust_json_parse(json_text):
    """
    Robustly parse JSON with multiple fallback strategies
    """
    if not json_text:
        return None
    
    # Strategy 1: Try parsing as-is
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Clean and try again
    cleaned_json = clean_json_strings(json_text)
    try:
        return json.loads(cleaned_json)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Try to extract partial JSON
    try:
        return extract_partial_json(json_text)
    except Exception:
        pass
    
    # Strategy 4: Manual parsing as last resort
    try:
        return manual_json_parse(json_text)
    except Exception:
        pass
    
    print("⚠️ All JSON parsing strategies failed")
    return None

def clean_json_strings(json_text):
    """
    Clean common JSON string issues
    """
    if not json_text:
        return json_text
    
    # Remove any leading/trailing non-JSON content
    json_text = json_text.strip()
    
    # Fix common quote issues in string values
    lines = json_text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip pure structural lines
        if line in ['{', '}', '[', ']', ',']:
            cleaned_lines.append(line)
            continue
        
        # Handle lines with string values that might have quote issues
        if '": "' in line:
            # Split into key and value parts
            key_value_match = re.match(r'^(\s*"[^"]+"\s*:\s*")(.*)(".*)$', line)
            if key_value_match:
                key_part = key_value_match.group(1)
                value_part = key_value_match.group(2)
                suffix_part = key_value_match.group(3)
                
                # Clean the value part
                # Remove any trailing comma or quote from value
                value_part = value_part.rstrip('",')
                
                # Escape quotes in the value
                value_part = value_part.replace('\\"', '"')  # Unescape first
                value_part = value_part.replace('"', '\\"')  # Escape all quotes
                
                # Reconstruct the line
                if line.endswith(','):
                    cleaned_line = key_part + value_part + '"'
                    cleaned_lines.append(cleaned_line)
                else:
                    cleaned_line = key_part + value_part + suffix_part
                    cleaned_lines.append(cleaned_line)
            else:
                cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def extract_partial_json(json_text):
    """
    Try to extract valid JSON parts from malformed JSON
    """
    # Try to find and extract just the atomic_facts array
    atomic_facts_match = re.search(r'"atomic_facts"\s*:\s*\[(.*?)\]', json_text, re.S)
    qa_match = re.search(r'"qa"\s*:\s*\[(.*?)\]', json_text, re.S)
    
    result = {}
    
    if atomic_facts_match:
        try:
            facts_text = atomic_facts_match.group(1)
            # Try to parse individual facts
            facts = []
            for fact_match in re.finditer(r'"([^"]+)"', facts_text):
                facts.append(fact_match.group(1))
            result["atomic_facts"] = facts
        except Exception:
            pass
    
    if qa_match:
        try:
            qa_text = qa_match.group(1)
            qa_pairs = []
            # Try to extract QA pairs
            for qa_match in re.finditer(r'\{"question":\s*"([^"]+)",\s*"answer":\s*"([^"]+)"\}', qa_text):
                qa_pairs.append({
                    "question": qa_match.group(1),
                    "answer": qa_match.group(2)
                })
            result["qa"] = qa_pairs
        except Exception:
            pass
    
    return result if result else None

def manual_json_parse(json_text):
    """
    Manual parsing as last resort - extract key information using regex
    """
    result = {"atomic_facts": [], "qa": []}
    
    # Extract atomic facts
    facts_pattern = r'"atomic_facts"\s*:\s*\[(.*?)\]'
    facts_match = re.search(facts_pattern, json_text, re.S)
    if facts_match:
        facts_content = facts_match.group(1)
        # Extract quoted strings
        for fact_match in re.finditer(r'"([^"]+)"', facts_content):
            result["atomic_facts"].append(fact_match.group(1))
    
    # Extract QA pairs
    qa_pattern = r'"qa"\s*:\s*\[(.*?)\]'
    qa_match = re.search(qa_pattern, json_text, re.S)
    if qa_match:
        qa_content = qa_match.group(1)
        # Try to find question-answer pairs
        qa_pairs = re.findall(r'\{"question":\s*"([^"]+)",\s*"answer":\s*"([^"]+)"\}', qa_content)
        for question, answer in qa_pairs:
            result["qa"].append({
                "question": question,
                "answer": answer
            })
    
    return result if result["atomic_facts"] or result["qa"] else None

# Small utility: split passage into sentences (very lightweight sentence segmentation)
def split_sentences(passage):
    # Keep punctuation, split by periods, question marks, exclamation marks
    parts = re.split(r'(?<=[.!?])\s+', passage.strip())
    return [p.strip() for p in parts if p.strip()]

# Extract shortest answerable substring from sentences based on rules
PATTERNS = [
    r"(directed by [^,.;\n]+)",
    r"(premiered at [^,.;\n]+)",
    r"(It is a sequel to [^,.;\n]+)",
    r"(sequel to [^,.;\n]+)",
    r"(The film's art direction was by [^,.;\n]+)",
    r"(art direction (?:was )?by [^,.;\n]+)",
    r"((?:is|was) [^,.;\n]+)", # Fallback: "is a ..." or "was a ..."
]

def extract_answer_by_pattern(sentence):
    for p in PATTERNS:
        m = re.search(p, sentence, flags=re.I)
        if m:
            return m.group(1).strip()
    return None

def find_sentence_with_phrase(passage, phrase):
    # Find sentences containing phrase (by earliest occurrence)
    for s in split_sentences(passage):
        if phrase in s:
            return s
    return None

def is_likely_title(phrase):
    # Simple judgment of whether it's a work title: contains quotes, apostrophes (Lilli's), or multiple words with capital letters
    if '"' in phrase or "'" in phrase:
        return True
    if re.search(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", phrase):
        # Multiple capitalized words
        return True
    return False

def create_question_answer_for_phrase(phrase, passage):
    """
    Create QA based on sentence patterns (local deterministic, no LLM call).
    Returns {question, answer} or None
    """
    sentence = find_sentence_with_phrase(passage, phrase)
    if not sentence:
        return None
    answer = extract_answer_by_pattern(sentence)
    # If answer exists but doesn't contain phrase (e.g., "directed by Jaap Speyer"), we can still use it as answer
    if answer:
        # Determine question type
        a_low = answer.lower()
        if "directed by" in a_low:
            q = f"Who directed {phrase}?"
            return {"question": q, "answer": answer}
        if "premiered at" in a_low:
            q = f"Where did {phrase} premiere?"
            return {"question": q, "answer": answer}
        if "sequel to" in a_low:
            q = f"Which film is {phrase} a sequel to?"
            return {"question": q, "answer": answer}
        if "art direction" in a_low:
            q = f"Who was responsible for the art direction of {phrase}?"
            return {"question": q, "answer": answer}
        # Fallback for "is/was ..." patterns
        if re.match(r"^(is|was)\b", answer, flags=re.I):
            q = f"What is {phrase}?"
            # Use the minimal answer starting with is/was
            return {"question": q, "answer": answer}
        # If no specific pattern matches, use the whole sentence as answer, use general question
        q = f"What is {phrase}?"
        return {"question": q, "answer": sentence}

def fallback_extract_required_phrases(passage):
    """
    Local fallback extraction when model fails: try to extract quoted titles, aliases in parentheses, and consecutive capitalized words (names/institutions/works)
    This function is a backup plan, model takes priority.
    """
    phrases = []
    # 1) Quoted titles
    for m in re.findall(r'"([^"]{2,80})"', passage):
        phrases.append(m.strip())
    for m in re.findall(r"'([^']{2,80})'", passage):
        phrases.append(m.strip())
    # 2) X (Y: Z) e.g., Lilli's Marriage (German: Lillis Ehe)
    for m in re.finditer(r'([A-Z][^(\n]{1,60})\s*\(([^)]{1,60})\)', passage):
        left = m.group(1).strip()
        right = m.group(2).strip()
        # Right might be like German: Lillis Ehe
        subparts = [p.strip() for p in re.split(r':', right) if p.strip()]
        phrases.append(left)
        for sp in subparts:
            if len(sp) > 1 and sp not in phrases:
                phrases.append(sp)
    # 3) Consecutive capitalized word groups (names / institutions / works)
    for m in re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\b', passage):
        if len(m) > 1 and m not in phrases:
            phrases.append(m)
    # Deduplicate and return reasonably sized phrases
    cleaned = []
    for p in phrases:
        p = p.strip()
        if 1 < len(p) < 120 and p not in cleaned:
            cleaned.append(p)
    return cleaned

def try_rewrite_invalid_question(qa, passage, required_phrases):
    """
    Try to locally deterministically replace questions containing vague pronouns/banned words
    - If question contains 'the film' etc., replace with preferred title (if exists)
    - Otherwise use answer in QA for local correction or use create_question_answer_for_phrase to generate alternative question
    """
    q = qa["question"]
    a = qa.get("answer", "")
    q_low = q.lower()
    had_bad = False
    for bad in BAD_WORDS_LOWER:
        if bad in q_low:
            had_bad = True
            break
    if not had_bad:
        return qa # No modification needed

    # 1) Prioritize title-type required_phrase to replace "the film" etc.
    title_candidates = [p for p in required_phrases if is_likely_title(p)]
    chosen_title = title_candidates[0] if title_candidates else (required_phrases[0] if required_phrases else None)
    new_q = q
    if chosen_title:
        # Replace a series of common vague words
        for bad in ["the film", "this film", "the movie", "this movie", "the book", "this book", "the article", "this article", "the passage", "this passage"]:
            new_q = re.sub(re.escape(bad), chosen_title, new_q, flags=re.I)
        # Ensure no pronouns like "it" at start
        new_q = re.sub(r'\b(it|this|that)\b', chosen_title, new_q, flags=re.I)
        # Clean up extra spaces
        new_q = re.sub(r'\s+', ' ', new_q).strip()
        # If still contains bad words, try to directly generate a clear question
        if has_bad_word(new_q):
            # If answer contains some required_phrase, use it
            for rp in required_phrases:
                if rp in a:
                    alt = create_question_answer_for_phrase(rp, passage)
                    if alt:
                        return alt
            # Otherwise directly generate Who/What template based on chosen_title (conservative)
            alt_q = {"question": f"What is {chosen_title}?", "answer": a if a else find_sentence_with_phrase(passage, chosen_title) or a}
            return alt_q
        else:
            return {"question": new_q, "answer": a}
    else:
        # No title to replace, try to find contained required_phrase based on answer
        for rp in required_phrases:
            if rp in a:
                alt = create_question_answer_for_phrase(rp, passage)
                if alt:
                    return alt
        # If everything fails, try to return original qa (as fallback)
        return qa

# --------- Enhanced Main Pipeline (SpaCy + LLM) ---------
def generate_qa(chunk):
    """
    Use SpaCy for entity recognition and relation extraction, then call LLM to generate atomic knowledge facts and QA pairs
    Returns QA list with token information and validation information.
    """
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0

    # Step 1: Use SpaCy to extract entities and relations
    print("🔍 Using SpaCy to extract entities and relations...")
    extracted_info = extract_entities_and_relations(chunk)
    entity_info = format_entities_for_prompt(extracted_info)
    
    # Extract entity text for subsequent validation
    required_phrases = [entity["text"] for entity in extracted_info["entities"]]
    required_phrases.extend(extracted_info["noun_chunks"][:3])  # Add some important noun phrases

    # Step 2: Call LLM to generate atomic knowledge facts and QA
    print("🤖 Calling LLM to generate atomic knowledge facts and QA...")
    prompt = ENHANCED_STAGE_PROMPT.format(passage=chunk, entity_info=entity_info)
    # print(prompt)
    stage_result = call_llm(prompt, max_new_tokens=5000)
    total_input_tokens += stage_result["input_tokens"]
    total_output_tokens += stage_result["output_tokens"]
    total_tokens += stage_result["total_tokens"]

    raw = stage_result["text"]
    json_text = extract_json_block(raw)
    atomic_facts = []
    qa_list = []

    # Try to parse model output using robust parsing
    print(f"🔍 Parsing LLM output: {json_text[:200]}...")
    
    parsed = robust_json_parse(json_text)
    
    if parsed:
        print("✅ Successfully parsed JSON output")
        if isinstance(parsed, dict):
            atomic_facts = parsed.get("atomic_facts", []) or []
            qa_list = parsed.get("qa", []) or []
        elif isinstance(parsed, list):
            # If model directly outputs array, treat as QA list
            qa_list = parsed
            atomic_facts = []
    else:
        print("⚠️ Failed to parse LLM output, using fallback extraction")
        atomic_facts = []
        qa_list = []

    # Standardize QA list (ensure it's list of dict with question & answer)
    safe_qa = []
    if isinstance(qa_list, list):
        for item in qa_list:
            if isinstance(item, dict) and "question" in item and "answer" in item:
                safe_qa.append({"question": item["question"].strip(), "answer": item["answer"].strip()})
    qa_list = safe_qa

    # Accept all LLM-generated QA pairs without bad word filtering
    qa_list = [{"question": qa["question"].strip(), "answer": qa["answer"].strip()} for qa in qa_list]
    final_invalid = []

    # Re-deduplicate QA (based on question text)
    seen_q = set()
    new_qa = []
    for qa in qa_list:
        qtxt = qa["question"].strip()
        if qtxt not in seen_q:
            seen_q.add(qtxt)
            new_qa.append(qa)
    qa_list = new_qa

    # Coverage check: which entities are not covered
    missing = coverage_check(required_phrases, qa_list)

    # Do deterministic repair for missing entities (local generation)
    auto_added = []
    # for rp in missing:
    #     auto = create_question_answer_for_phrase(rp, chunk)
    #     if auto:
    #         qa_list.append(auto)
    #         auto_added.append(rp)

    # Final validation: ensure answers are all passage substrings
    # for qa in qa_list:
    #     if qa["answer"] not in chunk:
    #         # Find the first entity in question and change answer to that complete sentence
    #         replaced = False
    #         for rp in required_phrases:
    #             if rp in qa["question"]:
    #                 sent = find_sentence_with_phrase(chunk, rp)
    #                 if sent:
    #                     qa["answer"] = sent
    #                     replaced = True
    #                     break
    #         if not replaced:
    #             # Worst case: use the entire passage as answer
    #             qa["answer"] = chunk

    validation_info = {
        "invalid_questions": final_invalid,
        "missing_phrases": missing,
        "required_phrases": required_phrases,
        "auto_added_phrases": auto_added,
        "atomic_facts": atomic_facts,
        "extracted_entities": extracted_info["entities"],
        "spacy_relations": extracted_info["relations"]
    }

    return {
        "qa_list": qa_list,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "validation_info": validation_info,
        "raw_model_output": raw  # Convenient for debugging
    }

# --------- Corpus Processing (Keep original output format and fields, only replace underlying generate_qa call) ---------
def process_corpus_file(input_file, output_file, dataset_name):
    """
    Process a single corpus file, generate QA and save incrementally
    """
    print(f"Starting to process {dataset_name} dataset...")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")

    failed_file = output_file.replace('.jsonl', '_failed.jsonl')
    print(f"Failed samples will be saved to: {failed_file}")

    corpus_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                corpus_data.append(json.loads(line.strip()))

    print(f"Total read {len(corpus_data)} corpus items")

    processed_count = 0
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            processed_count = sum(1 for line in f if line.strip())
        print(f"Found {processed_count} already processed corpus items, will start from item {processed_count + 1}")

    total_input_tokens = 0
    total_output_tokens = 0
    total_total_tokens = 0
    total_qa_count = 0
    failed_count = 0

    for i in range(processed_count, len(corpus_data)):
        item = corpus_data[i]
        title = item.get("title", "")
        passage = item.get("passage", "")

        print(f"\n🔄 Processing item {i + 1}/{len(corpus_data)}: {title[:50]}...")

        try:
            result = generate_qa("title: " + title + "\n" +"content: " + passage)

            if not result["qa_list"] or len(result["qa_list"]) == 0:
                print(f"⚠️  Item {i + 1} did not generate valid QA pairs, saving as failed sample")
                failed_count += 1
                failed_entry = {
                    "index": i + 1,
                    "title": title,
                    "passage": passage,
                    "reason": "no_valid_qa_generated",
                    "qa_list": result["qa_list"],
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "total_tokens": result["total_tokens"],
                    "validation_info": result.get("validation_info", {})
                }
                with open(failed_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(failed_entry, ensure_ascii=False) + '\n')
                continue

            validation_info = result.get("validation_info", {})
            has_validation_issues = (validation_info.get("invalid_questions") or validation_info.get("missing_phrases"))

            if has_validation_issues:
                print(f"⚠️  Item {i + 1} has validation issues, but still saving QA pairs")
                corpus_entry = {
                    "title": title,
                    "passage": passage,
                    "qa": result["qa_list"],
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "total_tokens": result["total_tokens"],
                    "validation_warnings": validation_info
                }
            else:
                corpus_entry = {
                    "title": title,
                    "passage": passage,
                    "qa": result["qa_list"],
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "total_tokens": result["total_tokens"]
                }

            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(corpus_entry, ensure_ascii=False) + '\n')

            total_input_tokens += result["input_tokens"]
            total_output_tokens += result["output_tokens"]
            total_total_tokens += result["total_tokens"]
            total_qa_count += len(result["qa_list"])

            print(f"✅ Successfully generated {len(result['qa_list'])} QA pairs, saved")
            print(f"📊 Current statistics: Total QA pairs={total_qa_count}, Total tokens={total_total_tokens}")

        except Exception as e:
            print(f"❌ Error processing item {i + 1}: {e}")
            print("Saving as failed sample, continuing with next item...")
            failed_count += 1
            failed_entry = {
                "index": i + 1,
                "title": title,
                "passage": passage,
                "reason": f"exception: {str(e)}",
                "qa_list": [],
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            }
            with open(failed_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(failed_entry, ensure_ascii=False) + '\n')
            error_log_file = output_file.replace('.jsonl', '_errors.log')
            with open(error_log_file, 'a', encoding='utf-8') as f:
                f.write(f"Error processing item {i+1} (title: {title[:50]}): {e}\n")
            continue

        if (i + 1) % 10 == 0:
            print(f"\n📈 Progress Report:")
            print(f"   Processed: {i + 1}/{len(corpus_data)} corpus items")
            print(f"   Success: {i + 1 - processed_count - failed_count} items")
            print(f"   Failed: {failed_count} items")
            print(f"   Generated QA pairs: {total_qa_count}")
            print(f"   Token usage: Input={total_input_tokens}, Output={total_output_tokens}, Total={total_total_tokens}")
            if i + 1 - processed_count - failed_count > 0:
                print(f"   Average per successful corpus: {total_qa_count/(i+1-processed_count-failed_count):.1f} QA pairs")

    print(f"\n🎉 {dataset_name} dataset processing completed!")
    print(f"📊 Final Statistics:")
    print(f"   Total corpus items: {len(corpus_data)}")
    print(f"   Successfully processed: {len(corpus_data) - processed_count - failed_count} items")
    print(f"   Failed samples: {failed_count} items")
    print(f"   Generated QA pairs: {total_qa_count}")
    print(f"   Token usage: Input={total_input_tokens}, Output={total_output_tokens}, Total={total_total_tokens}")
    if len(corpus_data) - processed_count - failed_count > 0:
        print(f"   Average per successful corpus: {total_qa_count/(len(corpus_data)-processed_count-failed_count):.1f} QA pairs")
    print(f"   Successful results saved to: {output_file}")
    print(f"   Failed samples saved to: {failed_file}")

def process_all_corpora():
    base_dir = "/root/autodl-tmp/ReadingCorpus/data/sampled"
    output_dir = "/root/autodl-tmp/ReadingCorpus/data/QA"
    os.makedirs(output_dir, exist_ok=True)
    datasets = [
        {
            "name": "hotpotqa",
            "input_file": os.path.join(base_dir, "hotpotqa_sample_corpus.jsonl"),
            "output_file": os.path.join(output_dir, "hotpotqa_qa_spacy.jsonl")
        },
         {
            "name": "musique",
            "input_file": os.path.join(base_dir, "musique_sample_corpus.jsonl"),
            "output_file": os.path.join(output_dir, "musique_qa_spacy.jsonl")
        }
    ]
    for dataset in datasets:
        if os.path.exists(dataset["input_file"]):
            process_corpus_file(
                dataset["input_file"],
                dataset["output_file"],
                dataset["name"]
            )
        else:
            print(f"❌ Input file does not exist: {dataset['input_file']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        chunk = "Lilli's Marriage (German: Lillis Ehe) is a 1919 German silent film directed by Jaap Speyer. It is a sequel to the film \"Lilli\", and premiered at the Marmorhaus in Berlin. The film's art direction was by Hans Dreier."
        print("🧪 Testing enhanced QA generation pipeline...")
        print(f"📝 Test text: {chunk}")
        print("\n" + "="*50)
        
        result = generate_qa(chunk)
        
        print("\n🎯 Generation Results:")
        print(f"📊 Token usage: Input={result['input_tokens']}, Output={result['output_tokens']}, Total={result['total_tokens']}")
        print(f"❓ Generated QA pairs count: {len(result['qa_list'])}")
        
        validation = result.get('validation_info', {})
        print(f"🔍 Extracted entities count: {len(validation.get('extracted_entities', []))}")
        print(f"🧠 Atomic knowledge facts count: {len(validation.get('atomic_facts', []))}")
        
        if result['qa_list']:
            print("\n📋 Generated QA pairs:")
            for i, qa in enumerate(result['qa_list'], 1):
                print(f"  {i}. Q: {qa['question']}")
                print(f"     A: {qa['answer']}")
        
        if validation.get('atomic_facts'):
            print("\n💡 Generated atomic knowledge facts:")
            for i, fact in enumerate(validation['atomic_facts'], 1):
                print(f"  {i}. {fact}")
        
        if validation.get('extracted_entities'):
            print("\n🏷️  SpaCy extracted entities:")
            for entity in validation['extracted_entities']:
                print(f"  - {entity['text']} ({entity['label']})")
        
        print("\n" + "="*50)
        print("✅ Test completed")
    elif len(sys.argv) > 1 and sys.argv[1] == "test-json":
        test_json_parsing()
    else:
        process_all_corpora()
