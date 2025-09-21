SELF_ASK_PROMPT_MUSIEUQ = """
Solve the question with the given knowledge.
Each line should start with either "Intermediate answer:", "Follow up:", "So the final answer is:", or "Are follow up questions needed here:".
#
Question: In which year did the publisher of In Cold Blood form?
Are follow up questions needed here: Yes.
Follow up: What business published In Cold Blood?
Intermediate answer: In Cold Blood was published in book form by Random House.
Follow up: Which year witnessed the formation of Random House?
Intermediate answer: Random House was form in 2001.
So the final answer is: 2001
#
Question: Who was in charge of the city where The Killing of a Sacred Deer was filmed?
Are follow up questions needed here: Yes.
Follow up: In which city was The Killing of a Sacred Deer filmed
Intermediate answer: The Killing of a Sacred Deer was filmed in Cincinnati.
Follow up: Who was in charge of Cincinnati?
Intermediate answer: The present Mayor of Cincinnati is John Cranley, so John Cranley is in charge.
So the final answer is: John Cranley
#
Question: Where on the Avalon Peninsula is the city that Signal Hill overlooks?
Are follow up questions needed here: Yes.
Follow up: What city does Signal Hill overlook?
Intermediate answer: Signal Hill is a hill which overlooks the city of St. John's.
Follow up: Where on the Avalon Peninsula is St. John's located?
Intermediate answer: St. John's is located on the eastern tip of the Avalon Peninsula.
So the final answer is: eastern tip
#
Question: {question}
Are follow up questions needed here:
""".strip()

SELF_ASK_PROMPT_WIKIMQA = """
Solve the question with the given knowledge.
Each line should start with either "Intermediate answer:", "Follow up:", "So the final answer is:", or "Are follow up questions needed here:".
Follow the examples below to answer the questions with natural language.
#
Question: Which film came out first, Blind Shaft or The Mask Of Fu Manchu?
Are follow up questions needed here: Yes.
Follow up: When did Blind Shaft come out?
Intermediate answer: Blind Shaft came out in 2003.
Follow up: When did The Mask Of Fu Manchu come out?
Intermediate answer: The Mask Of Fu Manchu came out in 1932.
So the final answer is: The Mask Of Fu Manchu
#
Question: When did John V, Prince Of Anhalt-Zerbst's father die?
Are follow up questions needed here: Yes.
Follow up: Who is the father of John V, Prince Of Anhalt-Zerbst?
Intermediate answer: The father of John V, Prince Of Anhalt-Zerbst is Ernest I, Prince of Anhalt-Dessau.
Follow up: When did Ernest I, Prince of Anhalt-Dessau die?
Intermediate answer: Ernest I, Prince of Anhalt-Dessau died on 12 June 1516.
So the final answer is: 12 June 1516
#
Question: Which film has the director who was born later, El Extrano Viaje or Love In Pawn?
Are follow up questions needed here: Yes.
Follow up: Who is the director of El Extrano Viaje?
Intermediate answer: The director of El Extrano Viaje is Fernando Fernan Gomez.
Follow up: Who is the director of Love in Pawn?
Intermediate answer: The director of Love in Pawn is Charles Saunders.
Follow up: When was Fernando Fernan Gomez born?
Intermediate answer: Fernando Fernan Gomez was born on 28 August 1921.
Follow up: When was Charles Saunders (director) born?
Intermediate answer: Charles Saunders was born on 8 April 1904.
So the final answer is: El Extrano Viaje
#
Question: {question}
Are follow up questions needed here:
""".strip()

SELF_ASK_PROMPT_HOTPOTQA = """
Solve the question with the given knowledge.
Each line should start with either "Intermediate answer:", "Follow up:", "So the final answer is:", or "Are follow up questions needed here:".
#
Question: What is the name of this American musician, singer, actor, comedian, and songwriter, who worked with Modern Records and born in December 5, 1932?
Are follow up questions needed here: Yes.
Follow up: Who worked with Modern Records?
Intermediate answer: Artists worked with Modern Records include Etta James, Little Richard, Joe Houston, Ike and Tina Turner and John Lee Hooker.
Follow up: Is Etta James an American musician, singer, actor, comedian, and songwriter, and was born in December 5, 1932?
Intermediate answer: Etta James was born in January 25, 1938, not December 5, 1932, so the answer is no.
Follow up: Is Little Richard an American musician, singer, actor, comedian, and songwriter, and was born in December 5, 1932?
Intermediate answer: Yes, Little Richard, born in December 5, 1932, is an American musician, singer, actor, comedian and songwriter.
So the final answer is: Little Richard
#
Question: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?
Are follow up questions needed here: Yes.
Follow up: What jobs did Chinua Achebe have?
Intermediate answer: Chinua Achebe was a Nigerian (1) novelist, (2) poet, (3) professor, and (4) critic, so Chinua Achebe had 4 jobs.
Follow up: What jobs did Rachel Carson have?
Intermediate answer: Rachel Carson was an American (1) marine biologist, (2) author, and (3) conservationist, so Rachel Carson had 3 jobs.
Follow up: Did Chinua Achebe have more jobs than Rachel Carson?
Intermediate answer: Chinua Achebe had 4 jobs, while Rachel Carson had 3 jobs. 4 is greater than 3, so yes, Chinua Achebe had more jobs.
So the final answer is: Chinua Achebe
#
Question: Remember Me Ballin' is a CD single by Indo G that features an American rapper born in what year?
Are follow up questions needed here: Yes.
Follow up: Which American rapper is featured by Remember Me Ballin', a CD single by Indo G?
Intermediate answer: Gangsta Boo
Follow up: In which year was Gangsta Boo born?
Intermediate answer: Gangsta Boo was born in August 7, 1979, so the answer is 1979.
So the final answer is: 1979
#
Question: {question}
Are follow up questions needed here:
""".strip()


DIRECT_RETRIEVE_ANSWER_PROMPT_WIKIMQA = """
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
{{"answer": "The Mask Of Fu Manchu"}}
```

<doc>
{{KNOWLEDGE FOR YOUR REFERENCE}}
</doc>
<Question>: When did John V, Prince Of Anhalt-Zerbst's father die?
<Answer>:
```json
{{"answer": "12 June 1516"}}
```

<doc>
{{KNOWLEDGE FOR YOUR REFERENCE}}
</doc>
<Question>: Which film has the director who was born later, El Extrano Viaje or Love In Pawn?
<Answer>:
```json
{{"answer": "El Extrano Viaje"}}
```

Now your question and reference knowledge are as follows.
<doc>
{knowledge}
</doc>
<Question>: {question}
<Answer>:
""".strip()

DIRECT_RETRIEVE_ANSWER_PROMPT_MUSIQUE = """
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
{{"answer": "2001"}}
```

<doc>
{{KNOWLEDGE FOR YOUR REFERENCE}}
</doc>
<Question>: Who was in charge of the city where The Killing of a Sacred Deer was filmed?
<Answer>:
```json
{{"answer": "John Cranley"}}
```

<doc>
{{KNOWLEDGE FOR YOUR REFERENCE}}
</doc>
<Question>: Where on the Avalon Peninsula is the city that Signal Hill overlooks?
<Answer>:
```json
{{"answer": "eastern tip"}}
```

Now your question and reference knowledge are as follows.
<doc>
{knowledge}
</doc>
<Question>: {question}
<Answer>:
""".strip()

DIRECT_RETRIEVE_ANSWER_PROMPT_HOTPOTQA = """
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
{{"answer": "Little Richard"}}
```

<doc>
{{KNOWLEDGE FOR YOUR REFERENCE}}
</doc>
<Question>: Between Chinua Achebe and Rachel Carson, who had more diverse jobs?
<Answer>:
```json
{{"answer": "Chinua Achebe"}}
```

<doc>
{{KNOWLEDGE FOR YOUR REFERENCE}}
</doc>
<Question>: Remember Me Ballin' is a CD single by Indo G that features an American rapper born in what year?
<Answer>:
```json
{{"answer": "1979"}}
```

Now your question and reference knowledge are as follows.
<doc>
{knowledge}
</doc>
<Question>: {question}
<Answer>:
""".strip()



