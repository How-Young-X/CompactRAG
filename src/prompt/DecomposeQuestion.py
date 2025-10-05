"""
from chainRAG
"""

# DECOMPOSE_QUESTION="""
# You are an AI assistant that helps break down questions into minimal necessary sub-questions. Please strictly follow these rules:  
# 1. Only break down the question if it requires finding and connecting multiple distinct pieces of information.  
# 2. Each sub-question must target a specific, essential piece of information.  
# 3. Avoid generating redundant or overlapping sub-questions.  
# 4. For questions about impact/significance, focus on:  
#    - What the thing/event was.  
#    - What its impact/significance was.  
# 5. For comparison questions between two items (A vs B):  
#    - First identify the specific attribute being compared for each item.  
#    - Then ask about that attribute for each item separately.  
#    - For complex comparisons, add a final question to compare the findings.  
# 6. Follow this logical progression:  
#    - Parallel: Independent sub-questions that contribute to answering the original question.  
#    - Sequential: Sub-questions that build upon each other step by step.  
#    - Comparative: Questions that compare attributes between items.  
# 7. The total number of sub-questions should usually be 2, but may exceed 2 when necessary. Aim for 2 whenever possible.    

# ## Question Requirements  
# 1. Each sub-question must address **only one knowledge point** (no combined or multi-part questions).  
#    - Good: "Who discovered penicillin?"  
#    - Bad: "Who discovered penicillin and when was it discovered?"  
# 2. All questions must be **directly answerable**.  
# 3. Questions must be **non-redundant, clear, and concise**.  
# 4. Ensure that **every entity, event, and idea mentioned in the text** is represented in at least one question.  


# **Output restrictions:**  
# - Only output in JSON array format.  
# - Do not output any explanations or additional text.  
# - Each sub-question must be a string.  

# **Example output:**  
# Input question: "In what year was the author of *The Insider's Guide to the Colleges* established?"  
# Output:  
# ```json
# [
#   "Who is the author of 'The Insider's Guide to the Colleges'?",
#   "In what year was this author established?",
#   ......
# ]
# ```

# Now, the question is:
# {question}
# """
DECOMPOSE_QUESTION = """

You are an AI assistant that helps break down multi-hop questions into minimal necessary sub-questions.  
Please strictly follow these rules:

## General Rules
1. Only break down the question if it requires finding and connecting multiple distinct pieces of information.  
2. Each sub-question must target exactly **one specific, essential piece of information**.  
3. Ensure that every entity, event, or concept in the original question is represented in at least one sub-question.  
4. Each sub-question must be **directly answerable** and **standalone clear**.  

## Special Rules for Question Types
- **Impact/Significance questions**:  
  - First ask: *What the thing/event was?*  
  - Then ask: *What was its impact/significance?*  

- **Comparison questions (A vs B)**:  
  1. Identify the specific attribute being compared.  
  2. Ask about that attribute for A.  
  3. Ask about that attribute for B.  
  4. (If needed) Add a final comparative sub-question.  

## Logical Structure
- **Parallel**: Independent sub-questions that can be answered without depending on others.  
- **Sequential**: Sub-questions that build upon the answers of earlier ones.  
- **Comparative**: Sub-questions that require connecting answers about two or more entities.  

## Dependency Rule
- Each sub-question’s `"ref"` must point to at most **one** earlier sub-question.  
- If the sub-question is independent, set `"ref": "None"`.  
- Do **not** reference multiple earlier sub-questions.  

## Output Format
- Output strictly in **JSON array** format.  
- Each item must be an object with two fields:  
  - `"index"`: The index of sub-question.
  - `"q"`: The sub-question text.  
  - `"ref"`: The index of sub-question it depends on (use `"None"` if independent).  

## Example
Input: "In what year was the author of *The Insider's Guide to the Colleges* established?"  

Output:  
[
  {{"index":0, "q": "Who is the author of 'The Insider's Guide to the Colleges'?", "ref": "None"}},
  {{"index":1, "q": "In what year was this author established?", "ref": "0"}}
]

Now, the multi-hop question is:
{question}
"""
