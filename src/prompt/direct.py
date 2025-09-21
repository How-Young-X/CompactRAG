DIRECT_PROMPT_HOTPOTQA = """
As an assistant, your task is to answer the question directly after <Question>. Your answer should be after <Answer> in JSON format with key "answer" and its value should be string.

There are some examples for you to refer to:
<Question>: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and born in December 5, 1932?
<Answer>:
```json
{{"answer": "Little Richard"}}
```

<Question>: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?
<Answer>:
```json
{{"answer": "Chinua Achebe"}}
```

<Question>: Remember Me Ballin' is a CD single by Indo G that features an American rapper born in what year?
<Answer>:
```json
{{"answer": "1979"}}
```

Now your Question is
<Question>: {question}
<Answer>:
""".strip()

DIRECT_PROMPT_WIKIMQA = """
As an assistant, your task is to answer the question directly after <Question>. Your answer should be after <Answer> in JSON format with key "answer" and its value should be string.

There are some examples for you to refer to:
<Question>: Which film came out first, Blind Shaft or The Mask Of Fu Manchu?
<Answer>:
```json
{{"answer": "The Mask Of Fu Manchu"}}
```

<Question>: When did John V, Prince Of Anhalt-Zerbst's father die?
<Answer>:
```json
{{"answer": "12 June 1516"}}
```

<Question>: Which film has the director who was born later, El Extrano Viaje or Love In Pawn?
<Answer>:
```json
{{"answer": "El Extrano Viaje"}}
```

Now your Question is
<Question>: {question}
<Answer>:
""".strip()

DIRECT_PROMPT_MUSIQUE = """
As an assistant, your task is to answer the question directly after <Question>. Your answer should be after <Answer> in JSON format with key "answer" and its value should be string.

There are some examples for you to refer to:
<Question>: In which year did the publisher of In Cold Blood form?
<Answer>:
```json
{{"answer": "2001"}}
```

<Question>: Who was in charge of the city where The Killing of a Sacred Deer was filmed?
<Answer>:
```json
{{"answer": "John Cranley"}}
```

<Question>: Where on the Avalon Peninsula is the city that Signal Hill overlooks?
<Answer>:
```json
{{"answer": "eastern tip"}}
```

Now your Question is
<Question>: {question}
<Answer>:
""".strip()


