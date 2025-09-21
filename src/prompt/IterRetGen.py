SYSTEM_PROMPT = """
Answer the questions based on given documents, you must give the answer in the format "So the final answer is".
Think step by step and answer the questions based on given documents. You must answer in JSON format with key "thought" and "answer".
""".strip()

ITER_RETGEN_MUSIQUE_PROMPT = """
You should think step by step and answer the question after <Question> based on given knowledge embraced with <doc> and </doc>. Your answer should be after <Answer> in JSON format with key "thought" and "answer", their value should be string.

Here are some examples for you to refer to:
<doc>
{{KNOWLEDGE FOR THE QUESTION}}
</doc>
<Question>: In which year did the publisher of In Cold Blood form?
Let's think step by step.
<Answer>:
```json
{{
    "thought": "In Cold Blood was first published in book form by Random House. Random House was form in 2001.",
    "answer": "2011"
}}
```

<doc>
{{KNOWLEDGE FOR THE QUESTION}}
</doc>
<Question>: Who was in charge of the city where The Killing of a Sacred Deer was filmed?
Let's think step by step.
<Answer>:
```json
{{
    "thought": "The Killing of a Sacred Deer was filmed in Cincinnati. The present Mayor of Cincinnati is John Cranley. Therefore, John Cranley is in charge of the city.",
    "answer": "John Cranley"
}}
```

<doc>
{{KNOWLEDGE FOR THE QUESTION}}
</doc>
<Question>: Where on the Avalon Peninsula is the city that Signal Hill overlooks?
Let's think step by step.
<Answer>:
{{
    "thought": "Signal Hill is a hill which overlooks the city of St. John's. St. John's is located on the eastern tip of the Avalon Peninsula.",
    "answer": "eastern tip"
}}
```

Now based on the given doc, answer the question after <Question>.
<doc>
{documents}
</doc>
<Question>: {question}
Let's think step by step.
<Answer>:
""".strip()

ITER_RETGEN_WIKIMQA_PROMPT = """
You should think step by step and answer the question after <Question> based on given knowledge embraced with <doc> and </doc>. Your answer should be after <Answer> in JSON format with key "thought" and "answer", their value should be string.

Here are some examples for you to refer to:
<doc>
{{KNOWLEDGE FOR THE QUESTION}}
</doc>
<Question>: Which film came out first, Blind Shaft or The Mask Of Fu Manchu?
Let's think step by step.
<Answer>:
```json
{{
    "thought": "Blind Shaft is a 2003 film, while The Mask Of Fu Manchu opened in New York on December 2, 1932. 2003 comes after 1932. Therefore, The Mask Of Fu Manchu came out earlier than Blind Shaft.",
    "answer": "The Mask Of Fu Manchu"
}}
```

<doc>
{{KNOWLEDGE FOR THE QUESTION}}
</doc>
<Question>: When did John V, Prince Of Anhalt-Zerbst's father die?
Let's think step by step.
<Answer>:
```json
{{
    "thought": "John V, Prince Of Anhalt-Zerbst was the son of Ernest I, Prince of Anhalt-Dessau. Ernest I, Prince of Anhalt-Dessau died on 12 June 1516.",
    "answer": "12 June 1516"
}}
```

<doc>
{{KNOWLEDGE FOR THE QUESTION}}
</doc>
<Question>: Which film has the director who was born later, El Extrano Viaje or Love In Pawn?
Let's think step by step.
<Answer>:
```json
{{
    "thought": "The director of El Extrano Viaje is Fernando Fernan Gomez, who was born on 28 August 1921. The director of Love In Pawn is Charles Saunders, who was born on 8 April 1904. 28 August 1921 comes after 8 April 1904. Therefore, Fernando Fernan Gomez was born later than Charles Saunders.",
    "answer": "El Extrano Viaje"
}}
```

Now based on the given doc, answer the question after <Question>
<doc>
{documents}
</doc>
<Question>: {question}
Let's think step by step.
<Answer>:
""".strip()

ITER_RETGEN_HOTPOTQA_PROMPT = """
You should think step by step and answer the question after <Question> based on given knowledge embraced with <doc> and </doc>. Your answer should be after <Answer> in JSON format with key "thought" and "answer", their value should be string.

Here are some examples for you to refer to:
<doc>
{{KNOWLEDGE FOR THE QUESTION}}
</doc>
<Question>: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and born in December 5, 1932?
Let's think step by step.
<Answer>:
```json
{{
    "thought": "Artists who worked with Modern Records include Etta James, Joe Houston, Little Richard, Ike and Tina Turner and John Lee Hooker in the 1950s and 1960s. Of these Little Richard, born in December 5, 1932, was an American musician, singer, actor, comedian, and songwriter.",
    "answer": "Little Richard"
}}
```

<doc>
{{KNOWLEDGE FOR THE QUESTION}}
</doc>
<Question>: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?
<Answer>:
```json
{{
    "thought": "Chinua Achebe was a Nigerian novelist, poet, professor, and critic. Rachel Carson was an American marine biologist, author, and conservationist. So Chinua Achebe had 4 jobs, while Rachel Carson had 3 jobs. Chinua Achebe had more diverse jobs than Rachel Carson.",
    "answer": "Chinua Achebe"
}}
```

<doc>
{{KNOWLEDGE FOR THE QUESTION}}
</doc>
<Question>: Remember Me Ballin' is a CD single by Indo G that features an American rapper born in what year?
<Answer>:
```json
{{
    "thought": "Remember Me Ballin' is the CD single by Indo G featuring Gangsta Boo. Gangsta Boo is Lola Mitchell's stage name, who was born in August 7, 1979, and is an American rapper.",
    "answer": "1979"
}}
```

Now based on the given doc, answer the question after <Question>.
<doc>
{documents}
</doc>
<Question>: {question}
Let's think step by step.
<Answer>:
""".strip()

