NAIVE_PROMPT_HOTPOTQA = """
Answer the given question in JSON format, you can refer to the document provided.
As an assistant, your task is to answer the question based on the given knowledge. Your answer should be after <Answer> in JSON format with key "answer" and its value should be string.
Your <Answer> must be wrapped by ```json and ```.
The given knowledge will be embraced by <doc> and </doc> tags. You can refer to the knowledge to answer the question. If the knowledge does not contain the answer, answer the question directly.

There are some examples for you to refer to:

<doc>
{{KNOWLEDGE FOR YOUR REFERENCE}}
</doc>
<Question>: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and born in December 5, 1932?
<Answer>:
```json
{{"thought":"Modern Record is a big R&B label with artists including Etta James, Joe Houston, Little Richard, Ike, Tina Turner and John Lee Hooker in the 1950s and 1960s. Little Richard is an American musician, signer actor and songwriter, born in December 5 1932. So the answer is Little Richard.","answer": "Little Richard"}}
```

<doc>
{{KNOWLEDGE FOR YOUR REFERENCE}}
</doc>
<Question>: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?
<Answer>:
```json
{{"thought":"Chinua Achebe was a Nigerian novelist, poet, professor, and critic. Rachel Carson was an American marine biologist, author, and conservationist. Chinua Achebe has 4 jobs while Rachel Carson has 3 jobs. So the answer is Chinua Achebe.","answer": "Chinua Achebe"}}
```

<doc>
{{KNOWLEDGE FOR YOUR REFERENCE}}
</doc>
<Question>: Remember Me Ballin' is a CD single by Indo G that features an American rapper born in what year?
<Answer>:
```json
{{"thought":"Remember Me Ballin' is the CD singer by Indo G that features Gangsta Boo, who is named Lola Mitchell, an American rapper born in 1979. So the answer is 1979.","answer": "1979"}}

Now your question and reference knowledge are as follows.
<doc>
{knowledge}
</doc>
<Question>: {question}
<Answer>:
""".strip()

NAIEV_PROMPT_WIKIMQA = """
Answer the given question in JSON format, you can refer to the document provided.
As an assistant, your task is to answer the question based on the given knowledge. Your answer should be after <Answer> in JSON format with key "answer" and its value should be string.
Your <Answer> must be wrapped by ```json and ```.
The given knowledge will be embraced by <doc> and </doc> tags. You can refer to the knowledge to answer the question. If the knowledge does not contain the answer, answer the question directly.

There are some examples for you to refer to:

<doc>
{{KNOWLEDGE FOR YOUR REFERENCE}}
</doc>
<Question>: Which film came out first, Blind Shaft or The Mask Of Fu Manchu?
<Answer>:
```json
{{"thought": "Blind Shaft is a 2003 Chinese film, and The Mask Of Fu Manchu is a 1932 American pre-Code adventure film. The Mask Of Fu Manchu came out first. So the answer is The Mask Of Fu Manchu.", "answer": "The Mask Of Fu Manchu"}}
```

<doc>
{{KNOWLEDGE FOR YOUR REFERENCE}}
</doc>
<Question>: When did John V, Prince Of Anhalt-Zerbst's father die?
<Answer>:
```json
{{"thought": "The father of John V, Prince Of Anhalt-Zerbst is Ernest I, Prince of Anhalt-Dessau. He died on 12 June 1516. So the answer is 12 June 1516.", "answer": "12 June 1516"}}
```

<doc>
{{KNOWLEDGE FOR YOUR REFERENCE}}
</doc>
<Question>: Which film has the director who was born later, El Extrano Viaje or Love In Pawn?
<Answer>:
```json
{{"thought": "The director of El Extrano Viaje is Fernando Fernan Gomez, he was born on 29 August 1921. The director of Love In Pawn is Charles Saunders, he was born on 8 April 1904. Fernando Fernan Gomez was born later, so film El Extrano Viaje has the director who was born later. So the answer is El Extrano Viaje.", "answer": "El Extrano Viaje"}}
```

Now your question and reference knowledge are as follows.
<doc>
{knowledge}
</doc>
<Question>: {question}
<Answer>:
""".strip()

NAIVE_PROMPT_MUSIQUE = """
Answer the given question in JSON format, you can refer to the document provided.
As an assistant, your task is to answer the question based on the given knowledge. Your answer should be after <Answer> in JSON format with key "answer" and its value should be string.
Your <Answer> must be wrapped by ```json and ```.
The given knowledge will be embraced by <doc> and </doc> tags. You can refer to the knowledge to answer the question. If the knowledge does not contain the answer, answer the question directly.

There are some examples for you to refer to:

<doc>
{{KNOWLEDGE FOR YOUR REFERENCE}}
</doc>
<Question>: In which year did the publisher of In Cold Blood form?
<Answer>:
```json
{{"thought": "The publisher of In Cold Blood is Random house, which was formed in 2001. So the answer is 2001.", "answer": "2001"}}
```

<doc>
{{KNOWLEDGE FOR YOUR REFERENCE}}
</doc>
<Question>: Who was in charge of the city where The Killing of a Sacred Deer was filmed?
<Answer>:
```json
{{"thought": "The killing of a Scared Deer was filmed in Cincinnati, Ohio, where John Cranley is the mayor. So the answer is John Cranley.", "answer": "John Cranley"}}
```

<doc>
{{KNOWLEDGE FOR YOUR REFERENCE}}
</doc>
<Question>: Where on the Avalon Peninsula is the city that Signal Hill overlooks?
<Answer>:
```json
{{"thought": "Signal Hill overlooks the city St. John's, which is located on the eastern tip of the Avalon Peninsula. So the answer is eastern tip.", "answer": "eastern tip"}}
```

Now your question and reference knowledge are as follows.
<doc>
{knowledge}
</doc>
<Question>: {question}
<Answer>:
""".strip()