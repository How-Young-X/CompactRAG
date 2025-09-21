Synthesis_Answer_Extract = """
You are a data-generation assistant. Produce exactly one JSON object (and only the JSON; no extra text) that follows the schema and constraints below.

== Schema ==
{{
  "qa_list": [
    {{"id": "qa1", "q": "<question 1>", "a": "<sentence answer 1>", "is_distractor": false}},
    {{"id": "qa2", "q": "<question 2>", "a": "<sentence answer 2>", "is_distractor": true}},
    ...
  ],
  "question": "<final target question>",
  "answer": "<short correct answer>",
  "answer_from": "<qa id>"
}}

== Hard constraints (must obey exactly) ==
1. Output must be valid JSON ONLY. Do not output anything outside the JSON object. Use double quotes for all strings, no trailing commas, and no comments.
2. The top-level object must contain exactly the four keys shown above in the same order: "qa_list", "question", "answer", "answer_from".
3. qa_list:
   - Must contain **3 to 5** items. IDs must be sequential "qa1", "qa2", ...
   - Each item must have exactly the keys: "id", "q", "a", "is_distractor".
   - "q": ≤ 20 words, no pronouns ("this", "that", "he", "she", "it"), concise, semantically highly similar (paraphrases, like top-k retrieval).
   - "a": full sentence, ≤ 20 words, grounded in passage. Single line (no `\n`). 
   - "is_distractor": boolean. At least one must be true.
4. Exactly one qa_list entry (the one referenced by "answer_from") must provide the **correct factual basis** for answering the final question.  
   - The correct entry can be **any one** of the qa_list items (qa1, qa2, qa3, q4, ...).  
   - The position of the correct entry must vary naturally based on the passage.  
   - Do NOT always use the same qa id (e.g., do NOT always set "answer_from": "qa2").  
   - Ensure that "answer_from" is consistent with the actual qa_list content.
5. "answer":
   - Short phrase (few words).
   - It **must appear verbatim as a contiguous substring** inside the `"a"` value of the `qa_list` entry referenced by `"answer_from"`.
   - The same substring **may also appear in other `a` values** (especially distractors), but in those cases the sentence must be **semantically wrong** or misleading (so they cannot answer the final question correctly).
6. "answer_from": must be the qa{idx} , the  correct qa_list entry is {idx}.
7. "question": semantically similar to qa_list questions, ≤ 20 words, no pronouns.
8. No raw newlines in strings. Must be valid JSON parsable by strict JSON parsers.

== Content rules ==
- All factual claims in correct answers must be grounded in the passage.
- Paraphrase questions to be semantically close but not identical.
- Avoid adding extra keys, comments, or explanations.

== Example (illustrative only, do NOT copy) ==
{{
  "qa_list": [
    {{"id": "qa1", "q": "Which river does Lostock Dam cross?", "a": "Lostock Dam crosses the Paterson River.", "is_distractor": false}},
    {{"id": "qa2", "q": "What river lies beneath the Lostock Dam?", "a": "Lostock Dam is located on the Paterson River in New South Wales.", "is_distractor": false}}
  ],
  "question": "Which river does Lostock Dam cross?",
  "answer": "Paterson River",
  "answer_from": "qa1"
}}

== Now generate ==
Passage:
{passage}

Produce the single JSON object that satisfies all above constraints.
"""

Synthesis_question_rewrite="""
You are a data generation assistant. Your task is to create training examples for pronoun/coreference resolution in multi-hop question answering.

## Instructions
1. I will give you a passage.
2. From this passage, generate a **multi-hop question chain** (at least 2 sub-questions).
3. The **first sub-question** should be a normal factual question answerable from the passage.
4. The **second sub-question** must contain a pronoun or ambiguous reference (e.g., "this", "that", "he", "she", "it", "the river", "the scientist", etc.), so that its interpretation depends on the answer to the first sub-question.
5. Then, rewrite the second sub-question by replacing the pronoun/ambiguous reference with the explicit entity (the answer from sub-question 1).
6. Finally, output the result in JSON format:
   {{
     "input": "Q: <second_subquestion_with_pronoun> | prev_answer: <answer_to_first_subquestion>",
     "output": "<resolved_second_subquestion>"
   }}

## Example
Passage:
"Lostock Dam crosses the Paterson River. The Paterson River flows into the Hunter River."

Step 1: First sub-question: "Which river does Lostock Dam cross?"
Answer: "Paterson River"

Step 2: Second sub-question (with pronoun): "Which watercourse is the river that the Lostock Dam is located on the mouth of?"

Step 3: Resolved second sub-question: "Which watercourse is Paterson River the mouth of?"

Final JSON:
{{
  "input": "Q: Which watercourse is the river that the Lostock Dam is located on the mouth of? | prev_answer: Paterson River",
  "output": "Which watercourse is Paterson River the mouth of?"
}}

## Now your turn
Passage:
{passage}
Generate the JSON as instructed.
""".strip()