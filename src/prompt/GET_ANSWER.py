LLM_EVAL_PROMPT = """
You are an experienced linguist who is responsible for evaluating the correctness of the generated responses.
You are provided with question, the generated responses and the corresponding ground truth answer.
Your task is to compare the generated responses with the ground truth responses and evaluate the correctness of the generated responses.
Response in JSON format with key "response" and value "yes" or "no".

Question: {question}
Prediction: {prediction}
Ground-truth Answer: {answer}
Your response:
""".strip()

LLM_EXTRACT_ANSWER_PROMPT = """
Given a question, you should simplify the response to a more concise form of answer. If the response is already in a concise form, you can response with the same answer. If the response does not contain the answer, you can return "noanswer".
You should come out the simplified answer in JSON format with key "answer" and the answer string as the value. Your response should be in markdown code block. Like
```json
{"answer": "simplified answer"}
```

Question: {question}
Response: {response}
""".strip()

LLM_REASON_ANSWER_FROM_SUBQUESTIONS = """
You are an AI reasoning assistant.  
The user will provide:  
- A multi-hop question.  
- Its sub-questions.  
- Retrieved knowledge for each sub-question (which may contain multiple relevant or irrelevant facts).  

Your task:  
Carefully reason using the retrieved knowledge as well as your existing knowledge to answer each sub-question, and then combine them to answer the original multi-hop question. Produce a coherent, natural-language reasoning chain in the ‘thought’ field that explains how you used both the retrieved and your own knowledge to arrive at the final answer. Do not label facts as correct or incorrect; just synthesize them logically. Do not answer with 'I don't know', 'unknown', 'unclear'.
## Output format:
Return a JSON object as a string with three fields:
{{
  "question": "<the original multi-hop question>",
  "thought": "<a single, coherent paragraph explaining your reasoning process based on the retrieved knowledge>",
  "answer": "<the final concise answer: a word or short phrase>"
}}

## Example:
Multi-hop question:  
Which country is the capital of the city where the Eiffel Tower is located part of?

Sub-question 1: In which city is the Eiffel Tower located?  
Retrieved knowledge:  
In which city is the Eiffel Tower located?  
The Eiffel Tower is located in Paris.  

Where is the Eiffel Tower located?  
The Eiffel Tower is located in Berlin.

What is the Eiffel Tower?  
The Eiffel Tower is a wrought-iron lattice tower. 

Sub-question 2: Which country is this city the capital of?  
Retrieved knowledge:  
Paris is the capital of which country?  
Paris is the capital of France.  

What country is Paris the capital of?  
Paris is the capital of Spain.  

Berlin is the capital of which country?  
Berlin is the capital of Germany. 

Expected output (as a string):
{{
  "question": "Which country is the capital of the city where the Eiffel Tower is located part of?",
  "thought": "Based on the retrieved knowledge, the Eiffel Tower is identified as being located in Paris. Considering Paris as the relevant city, the retrieved facts indicate that it is the capital of France. By synthesizing these pieces of information, we can conclude that the country of the city where the Eiffel Tower is located is France.",
  "answer": "France"
}}

Now, it is your turn.
{input}
""".strip()
