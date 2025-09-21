QA_PROMPT = """
Answer the given question in JSON format, you can refer to the question answer pairs provided.
As an assistant, your task is to answer the question based on the given knowledge. Your answer should be after <Answer> in JSON format with key "answer" and its value should be string.
Your <Answer> must be wrapped by ```json and ```.
The given knowledge will be embraced by <doc> and </doc> tags. You can refer to the knowledge to answer the question. If the knowledge does not contain the answer, answer the question directly.

There are some examples for you to refer to:

<doc>
Who was the first man to walk on the moon?  
Neil Armstrong

Who wrote the play "Hamlet"?  
William Shakespeare

Which Greek poet wrote the Iliad and the Odyssey?  
Homer

What is the capital of Germany?  
Berlin

Mount Fuji is located in which country?  
Japan
</doc>
<Question>: Who is the author of "Hamlet"?  
<Answer>:  
```json
{{"answer": "William Shakespeare"}}
```

Now your question and reference knowledge are as follows.
<doc>
{qas}
</doc>
<Question>: {question}
<Answer>:
""".strip()