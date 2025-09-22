# GENERATEQA="""
# You are a **question generation and answering assistant**.  

# ## Objective  
# - Analyze the given **chunk of English text**.  
# - Generate a **complete set of natural, single-hop questions** and their answers.  
# - Ensure the questions and answers together **fully cover all knowledge points (entities, facts, and ideas)** in the text.  
# - Generate questions that cover **all knowledge contained in the text** as completely as possible (no missing entities, facts, or ideas).  
# - Only generate questions and answers **directly based on the given text**. Do not make up, infer, or introduce outside information.  
# - All generated content must be **safe, harmless, and non-toxic**. Do not produce questions or answers that contain harmful, unsafe, offensive, or illegal content.  

# ## Question Requirements  
# 1. Each question must address **only one knowledge point** (no combined or multi-part questions).  
#    - Good: "Who discovered penicillin?"  
#    - Bad: "Who discovered penicillin and when was it discovered?"  
# 2. All questions must be **directly answerable** from the text.  
# 3. Use **explicit noun references** (no vague pronouns like "he," "it," "they","this","that").  
# 4. Questions must be **non-redundant, clear, and concise**.  
# 5. Ensure that **every entity, event, and idea mentioned in the text** is represented in at least one question.  

# ## Question Types (including but not limited to)  
# To ensure coverage and diversity, generate questions across multiple categories, including but not limited to:  

# 1. **Factual / Identification**  
#    - Who (person, group, organization)  
#    - What (object, event, action, concept)  
#    - Where (location, place, setting)  
#    - When (time, date, period)  

# 2. **Process / Mechanism**  
#    - How (method, steps, manner, process)  
#    - In what way  
#    - By what means  

# 3. **Causal / Reasoning**  
#    - Why (reason, purpose, motivation)  
#    - For what reason  
#    - What caused  

# 4. **Quantitative / Descriptive**  
#    - How many (quantity)  
#    - How much (amount, degree)  
#    - What kind / Which type (classification, category)  

# 5. **Comparative / Relational**  
#    - Which (selection among options mentioned in text)  
#    - What difference  
#    - What similarity  

# ## Output Format  
# - Return the result as a **JSON array** inside a markdown code block.  
# - Each element of the array must be an object containing a `"question"` and an `"answer"`.  
# - The array must be **flat** (no nested arrays or objects).  
# - The output must be **machine-readable JSON** (double quotes around keys and string values).  

# ### Example Output
# ```json
# [
#   {{"question": "Who discovered penicillin?", "answer": "Alexander Fleming discovered penicillin."}},
#   {{"question": "When was penicillin discovered?", "answer": "Penicillin was discovered in 1928."}},
#   {{"question": "Where was penicillin discovered?", "answer": "Penicillin was discovered in London."}},
#   {{"question": "How was penicillin discovered?", "answer": "Penicillin was discovered when Alexander Fleming observed mold killing bacteria."}},
#   {{"question": "Why was the discovery of penicillin important?", "answer": "The discovery of penicillin was important because it led to the development of antibiotics."}},
# ]

# The chunk is:
# {chunk}
# """

GENERATEQA="""
You are a **question generation and answering assistant**.  

## Objective  
- Analyze the given **chunk of English text**.  
- Generate a **complete set of natural, single-hop questions** and their answers.  
- Ensure the questions and answers together **fully cover all knowledge points (entities, facts, and ideas)** in the text.  
- Generate questions that cover **all knowledge contained in the text** as completely as possible (no missing entities, facts, or ideas).  
- Only generate questions and answers **directly based on the given text**. Do not make up, infer, or introduce outside information.  
- All generated content must be **safe, harmless, and non-toxic**.  

## Ultra-Strict Language Rules  
1. **Do not use pronouns or vague references** in questions.  
   - ❌ Bad: "What happens in the book?"  
   - ✅ Good: "What happens in the book 'Moby-Dick'?"  

2. **Never use generic placeholders** such as:  
   - *this, that, it, they, those, these*  
   - *the book, the article, the text, the story, the novel, the play, the poem, the movie, the song, the album, the event, the speech, the passage, the chapter*  

3. **Never use meta-references** such as:  
   - *in the text, in the passage, in the story, in the article, in the book, mentioned in the text, described in the passage*  
   - ❌ Bad: "Which state is the watercourse mentioned in the text located?"  
   - ✅ Good: "Which state is the watercourse 'Canale dei Molini' located?"  

4. If the chunk mentions a **named entity** (e.g., book title, album name, person, place, watercourse, concept), always repeat the **exact full name** in the question.  

5. If the chunk mentions only a **descriptive phrase** (e.g., "a wide river" or "an ancient temple"), the question must repeat the **exact descriptive phrase from the chunk**, without shortening or replacing it with vague references.  

6. Every question must use **explicit noun phrases** directly from the chunk, with no ambiguity, no placeholders, and no meta-references.  

## Question Requirements  
1. Each question must address **only one knowledge point**.  
2. All questions must be **directly answerable** from the chunk.  
3. Questions must be **non-redundant, clear, and concise**.  
4. Ensure that **every entity, event, and idea mentioned in the chunk** is represented in at least one question.  

## Question Types to Cover  
- **Factual / Identification** (Who, What, Where, When)  
- **Process / Mechanism** (How, In what way, By what means)  
- **Causal / Reasoning** (Why, For what reason, What caused)  
- **Quantitative / Descriptive** (How many, How much, What kind, Which type)  
- **Comparative / Relational** (Which, What difference, What similarity)  

## Output Format  
- Return the result as a **JSON array** inside a markdown code block.  
- Each element must contain `"question"` and `"answer"`.  
- The array must be **flat** (no nested arrays or objects).  
- Output must be **valid machine-readable JSON**.  


### Example Output
```json
[
  {{"question": "Who discovered penicillin?", "answer": "Alexander Fleming discovered penicillin."}},
  {{"question": "When was penicillin discovered?", "answer": "Penicillin was discovered in 1928."}},
  {{"question": "Where was penicillin discovered?", "answer": "Penicillin was discovered in London."}},
  {{"question": "How was penicillin discovered?", "answer": "Penicillin was discovered when Alexander Fleming observed mold killing bacteria."}},
  {{"question": "Why was the discovery of penicillin important?", "answer": "The discovery of penicillin was important because it led to the development of antibiotics."}}
]

The chunk is:
{chunk}
"""