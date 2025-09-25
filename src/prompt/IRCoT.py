"""
IRCoT (Interleaving Retrieval with Chain-of-Thought) prompts

基于IRCOT论文的prompt模板，用于交替检索和推理的问答任务。
参考IterRetGen的prompt结构，但内容与原IRCOT工程代码保持一致。
"""

# IRCoT HotpotQA Prompt - 参考IterRetGen结构，内容与原IRCOT工程一致
IRCOT_HOTPOTQA_PROMPT = """
You should think step by step and answer the question after <Question> based on given knowledge embraced with <doc> and </doc>. Your answer should be after <Answer> in JSON format with key "thought" and "answer", their value should be string.

Here are some examples for you to refer to:
<doc>
Wikipedia Title: John Lennon/Plastic Ono Band
John Lennon/Plastic Ono Band is the debut studio album by English rock musician John Lennon. It was released in 1970, after Lennon had issued three experimental albums with Yoko Ono and "Live Peace in Toronto 1969", a live performance in Toronto credited to the Plastic Ono Band. The album was recorded simultaneously with Ono's debut avant garde solo album, "Yoko Ono/Plastic Ono Band", at Ascot Sound Studios and Abbey Road Studios using the same musicians and production team and nearly identical cover artwork.

Wikipedia Title: Milk and Honey (album)
Milk and Honey is an album by John Lennon and Yoko Ono released in 1984. Following the compilation "The John Lennon Collection", it is Lennon's eighth and final studio album, and the first posthumous release of new Lennon music, having been recorded in the last months of his life during and following the sessions for their 1980 album "Double Fantasy". It was assembled by Yoko Ono in association with the Geffen label.

Wikipedia Title: Walls and Bridges
Walls and Bridges is the fifth studio album by English musician John Lennon. It was issued by Apple Records on 26 September 1974 in the United States and on 4 October in the United Kingdom. Written, recorded and released during his 18-month separation from Yoko Ono, the album captured Lennon in the midst of his "Lost Weekend". "Walls and Bridges" was an American "Billboard" number-one album and featured two hit singles, "Whatever Gets You thru the Night" and "#9 Dream". The first of these was Lennon's first number-one hit in the United States as a solo artist, and his only chart-topping single in either the US or Britain during his lifetime.

Wikipedia Title: Every Man Has a Woman Who Loves Him
"Every Man Has a Woman Who Loves Him" is a Yoko Ono song from her "Double Fantasy" album with John Lennon. The version released as a single, and on the compilation album "Every Man Has a Woman", is credited to Lennon, and has stripped away Ono's lead vocal, while bringing Lennon's backing vocal up in the mix (this version was also released as a bonus track on the CD reissue of "Milk and Honey"). The B-side featured "It's Alright" by Ono and Sean Lennon. The last Lennon song to be released as a single, it failed to chart in either the United Kingdom or the United States.

Wikipedia Title: Nobody Loves You (When You're Down and Out)
"Nobody Loves You (When You're Down and Out)" is a song written by John Lennon released on his 1974 album "Walls and Bridges". The song is included on the 1986 compilation "Menlove Ave.", the 1990 boxset "Lennon", the 1998 boxset "John Lennon Anthology", the 2005 two-disc compilation "", and the 2010 boxset "Gimme Some Truth".
</doc>
<Question>: Nobody Loves You was written by John Lennon and released on what album that was issued by Apple Records, and was written, recorded, and released during his 18 month separation from Yoko Ono?
Let's think step by step.
<Answer>:
```json
{{
    "thought": "The album issued by Apple Records, and written, recorded, and released during John Lennon's 18 month separation from Yoko Ono is Walls and Bridges. Nobody Loves You was written by John Lennon on Walls and Bridges album. So the answer is Walls and Bridges.",
    "answer": "Walls and Bridges"
}}
```

<doc>
Wikipedia Title: Missouri Route 413
Route 413 is a highway in Missouri running between Route 13 in Springfield and Route 13 in Reeds Spring. It is an older alignment of Route 13 which was later rerouted. Except for two sections (in Springfield between Route 13 and U.S. Route 60 and about a mile between US 60 and Route 265 southwest of Billings), the road is runs concurrent with other designations for its entire length.

Wikipedia Title: Cambodia
Cambodia ( ; Khmer: កម្ពុជា , or Kampuchea ] ), officially known as the Kingdom of Cambodia (Khmer: ព្រះរាជាណាចក្រកម្ពុជា , "Preăh Réachéanachâk Kâmpŭchéa", ] ), is a country located in the southern portion of the Indochina Peninsula in Southeast Asia. It is 181035 km2 in area, bordered by Thailand to the northwest, Laos to the northeast, Vietnam to the east, and the Gulf of Thailand to the southwest.

Wikipedia Title: National Route 13 (Vietnam)
National Route 13 (Vietnamese: "Quốc lộ 13" ) is a highway in southern Vietnam stretching from the northeastern outskirts of Ho Chi Minh City, the commercial centre of the country, towards the border to Cambodia. The highway starts around Thủ Đức on the northern outskirts of Ho Chi Minh City, once the site of the military academy of the Army of the Republic of Vietnam, and travels north through the provinces of Bình Dương and Bình Phước. The highway passes through the districts of Thuận An, Thủ Dầu Một town, Bến Cát, Chơn Thành, Đồng Phú, Bình Long, and Lộc Ninh.

Wikipedia Title: Gangbyeonbuk-ro
The Gangbyeon Expressway (Korean: 강변북로 ) is an eight-lane highway located in Seoul, South Korea. It is part of National Route 46 and National Route 77. This route connects Seoul from Goyang to Namyangju, with a total length of 37.37 km . The highway is directly connected with the Jayu Motorway (자유로) in the border city located on the Gayang Bridge. The Mapo Bridge–Namyangju section has been designated as National Route 46, while the Yanghwa Bridge–Goyang section is specified as National Route 77. The highway is the main arterial road that connects northern Seoul with satellite cities such as Goyang, Paju, and Namyangju.

Wikipedia Title: Route 13 (Laos)
Route 13 is the most important highway in the country of Laos. It begins at Boten in the North of Laos at the Chinese border. It connects the city of Vientiane to Luang Prabang in the north and roughly follows the line of the Mekong River down to the border with Cambodia. The road then continues at National Highway 7 in Cambodia. Also this national highway leads past all three international airports in Laos Vientiane Airport, Luang Prabang Airport and Pakse Airport. Between Boten and Nateuy the Route 13 is at a length of 20 km part of the Kunming-Bangkok Expressway, which is also part of the Asian Highway AH3. From Nateuy to Vientiane, the Route 13 is part of the AH12 and from Vientiane to the border of Cambodia AH11.
</doc>
<Question>: What is known as the Kingdom and has National Route 13 stretching towards its border?
Let's think step by step.
<Answer>:
```json
{{
    "thought": "Cambodia is officially known as the Kingdom of Cambodia. National Route 13 streches towards border to Cambodia. So the answer is Cambodia.",
    "answer": "Cambodia"
}}
```

<doc>
Wikipedia Title: Christopher Nolan
Christopher Edward Nolan ( ; born 30 July 1970) is an English-American film director, producer, and screenwriter. He is one of the highest-grossing directors in history, and among the most successful and acclaimed filmmakers of the 21st century.

Wikipedia Title: James Draper (settler)
James Draper (1618–1694) was an early settler of the Massachusetts Bay Colony. He was born and married in Heptonstall, Yorkshire, England, and came with his wife to New England shortly after 1647. He was a weaver by profession, and settled in the Massachusetts town of Roxbury, but also lived for a short while in the towns of Dedham and Charlestown. He and his wife had nine children and many notable descendants. They are buried in the Westerly Burial Ground in West Roxbury, now a neighborhood of Boston, and share the oldest marker in the cemetery.

Wikipedia Title: Downhill (2014 film)
Downhill is a British comedy film directed by James Rouse and written by Torben Betts about four old friends who reunite to walk the Coast to Coast Walk. It stars Richard Lumsden, Karl Theobald, Jeremy Swift and Ned Dennehy. It was produced by Benji Howell and is noted for its "innovative" release strategy

Wikipedia Title: Jeremy Theobald
Jeremy Theobald is a British actor best known for his portrayal of "The Young Man", the main character in Christopher Nolan's 1998 major picture debut "Following", and for which Theobald was also a producer, Filming was scheduled around their day jobs. Jonathan Romney, writing in the "New Statesman", noted that "Nolan and his cast are terrific finds: I wouldn't normally say this to struggling artists, but they might want to give up their day jobs."

Wikipedia Title: Semper Gestion
Semper is a Swiss wealth management company, asset management and Family Office established in 2001 and working in open architecture. Based in Geneva, with a team of highly skilled portfolio managers who share a common vision of the profession, its model and its values, Semper has a sister company in London (Beauregard Capital Limited) and a representative office in Shanghai (Shanghai Semper). Since 2001, she offers a highly personalized service of wealth management and Family Office. Managed by Grégoire Vaucher (CEO) and Jean-Evrard Dominicé (CIO), Semper currently has 20 employees.
</doc>
<Question>: Jeremy Theobald and Christopher Nolan share what profession?
Let's think step by step.
<Answer>:
```json
{{
    "thought": "Jeremy Theobald is an actor and producer. Christopher Nolan is a director, producer, and screenwriter. Therefore, they both share the profession of being a producer. So the answer is producer.",
    "answer": "producer"
}}
```

Now based on the given doc, answer the question after <Question>.
<doc>
{context}
</doc>
Previous reasoning: {generation_so_far}
<Question>: {question}
Let's think step by step.
<Answer>:
""".strip()

# IRCoT Musique Prompt - 参考IterRetGen结构，内容与原IRCOT工程一致
IRCOT_MUSIQUE_PROMPT = """
You should think step by step and answer the question after <Question> based on given knowledge embraced with <doc> and </doc>. Your answer should be after <Answer> in JSON format with key "thought" and "answer", their value should be string.

Here are some examples for you to refer to:
<doc>
Wikipedia Title: Jonathan Stanton
Jonathan Stanton (1834–1918) was an ornithologist and longtime professor of Greek and Latin at Bates College, librarian, and supporter of the debate program.

Wikipedia Title: Robichaud v Canada (Treasury Board)
Robichaud v Canada (Treasury Board), [1987] 2 S.C.R. 84 is a leading case decided by the Supreme Court of Canada on sexual harassment under the Canadian Human Rights Act. The Court found that a corporation can be found liable for the discriminatory conduct of its employees who are acting "in the course of their employment." It also found it necessary to impose liability, as the employer is the only one that is in the position to remedy the discriminatory conduct.

Wikipedia Title: Neville A. Stanton
Neville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton. Prof Stanton is a Chartered Engineer (C.Eng), Chartered Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). He has written and edited over a forty books and over three hundered peer-reviewed journal papers on applications of the subject. Stanton is a Fellow of the British Psychological Society, a Fellow of The Institute of Ergonomics and Human Factors and a member of the Institution of Engineering and Technology. He has been published in academic journals including "Nature". He has also helped organisations design new human-machine interfaces, such as the Adaptive Cruise Control system for Jaguar Cars.

Wikipedia Title: Southampton
The University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students. The university is ranked in the top 100 research universities in the world in the Academic Ranking of World Universities 2010. In 2010, the THES - QS World University Rankings positioned the University of Southampton in the top 80 universities in the world. The university considers itself one of the top 5 research universities in the UK. The university has a global reputation for research into engineering sciences, oceanography, chemistry, cancer sciences, sound and vibration research, computer science and electronics, optoelectronics and textile conservation at the Textile Conservation Centre (which is due to close in October 2009.) It is also home to the National Oceanography Centre, Southampton (NOCS), the focus of Natural Environment Research Council-funded marine research.

Wikipedia Title: Lanton Mills
Lanton Mills is an American comedy short film written and directed by Terrence Malick and starring Malick, Warren Oates, Harry Dean Stanton, and Paula Mandel. The film was Malick's thesis project for the American Film Institute, and was completed in 1969. The story concerns two apparently 19th-century cowboys (Stanton and Malick) plotting to rob a bank in Texas, which they do in the 20th century.
</doc>
<Question>: When was Neville A. Stanton's employer founded?
Let's think step by step.
<Answer>:
```json
{{
    "thought": "The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. So the answer is 1862.",
    "answer": "1862"
}}
```

<doc>
Wikipedia Title: ISO/TC 68
ISO/TC 68 is a technical committee formed within the International Organization for Standardization (ISO), of Geneva, Switzerland, tasked with developing and maintaining international standards covering the areas of banking, securities, and other financial services. As the standards organization under ISO responsible for the development of all international financial services standards, ISO/TC 68 plays a key role in the development and adoption of new technologies in the banking, brokerage and insurance industries. Many of its current work projects involve developing ecommerce standards such as better online security for financial transactions, XML standards for financial transactions and standards to reduce the cost and delays of international financial transactions. The membership of ISO/TC 68, consists of more than 30 organizations assigned by participating national standards bodies plus additional international standards development organizations that work collaboratively toward global financial services standards development.

Wikipedia Title: ISO 21500
ISO 21500:2012, Guidance on Project Management, is an international standard developed by the International Organization for Standardization, or ISO starting in 2007 and released in 2012. It was intended to provide generic guidance, explain core principles and what constitutes good practice in project management. The ISO technical committee dealing with project management, ISO/PC 236 was held by the American National Standards Institute (ANSI) which had approved four standards that used PMI materials. one of which was ANSI/PMI 99-001-2008, A Guide to the Project Management Body of Knowledge - 4th Edition (PMI BoK® Guide - 4th Edition) (revision and re-designation of ANSI/PMI 99-001-2004): 11/20/2008.

Wikipedia Title: Unicode
Unicode is developed in conjunction with the International Organization for Standardization and shares the character repertoire with ISO/IEC 10646: the Universal Character Set. Unicode and ISO/IEC 10646 function equivalently as character encodings, but The Unicode Standard contains much more information for implementers, covering—in depth—topics such as bitwise encoding, collation and rendering. The Unicode Standard enumerates a multitude of character properties, including those needed for supporting bidirectional text. The two standards do use slightly different terminology.

Wikipedia Title: ISO 3166-2:AO
ISO 3166-2:AO is the entry for Angola in ISO 3166-2, part of the ISO 3166 standard published by the International Organization for Standardization (ISO), which defines codes for the names of the principal subdivisions (e.g., provinces or states) of all countries coded in ISO 3166-1.

Wikipedia Title: ISO 3307
ISO 3307 is an international standard for date and time representations issued by the International Organization for Standardization (ISO). The standard was issued in 1975, then was superseded by ISO 8601 in 1988.
</doc>
<Question>: What is the headquarters for the organization who sets the standards for ISO 21500?
Let's think step by step.
<Answer>:
```json
{{
    "thought": "The standards for ISO 21500 were set by International Organization for Standardization. The International Organization for Standardization has headquarters in Geneva. So the answer is Geneva.",
    "answer": "Geneva"
}}
```

<doc>
Wikipedia Title: Bogotá
Bogotá (/ ˈboʊɡətɑː /, / ˌbɒɡəˈtɑː /, / ˌboʊ - /; Spanish pronunciation: (boɣoˈta) (listen)), officially Bogotá, Distrito Capital, abbreviated Bogotá, D.C., and formerly known as Santafé de Bogotá between 1991 and 2000, is the capital and largest city of Colombia, administered as the Capital District, although often thought of as part of Cundinamarca. Bogotá is a territorial entity of the first order, with the same administrative status as the departments of Colombia. It is the political, economic, administrative, industrial, artistic, cultural, and sports center of the country.

Wikipedia Title: Arrondissement of Mechelen
The Arrondissement of Mechelen (; ) is one of the three administrative arrondissements in the Province of Antwerp, Belgium. It is both an administrative and a judicial arrondissement, as the territory for both coincides.

Wikipedia Title: Guy Shepherdson
Guy Shepherdson (born 17 February 1982 in Jakarta, Indonesia) is an Australian former rugby union professional footballer. He played as a tight-head prop for the Brumbies and Reds in the Super Rugby competition and played for the Australian national team, the Wallabies.

Wikipedia Title: Vilnius County
Vilnius County () is the largest of the 10 counties of Lithuania, located in the east of the country around the city Vilnius. On 1 July 2010, the county administration was abolished, and since that date, Vilnius County remains as the territorial and statistical unit.

Wikipedia Title: SMA Negeri 68 Jakarta
SMA Negeri 68 Jakarta (SMANED) is a public high school located at Salemba Raya street in Central Jakarta, Indonesia. The school is in one complex with SMP Negeri 216 Jakarta, SD Negeri Kenari, and Menza functional building. It was established on August 29, 1981 after being inaugurated by President Soeharto. In 2006, it was appointed to become RSBI (Rintisan Sekolah Bertaraf Internasional). Today, there are 840 students and 103 teachers and staff.
</doc>
<Question>: What region of the state where Guy Shepherdson was born, contains SMA Negeri 68?
Let's think step by step.
<Answer>:
```json
{{
    "thought": "Guy Shepherdson was born in Jakarta. SMA Negeri 68 Jakarta is located in Central Jakarta. So the answer is Central Jakarta.",
    "answer": "Central Jakarta"
}}
```

Now based on the given doc, answer the question after <Question>.
<doc>
{context}
</doc>
Previous reasoning: {generation_so_far}
<Question>: {question}
Let's think step by step.
<Answer>:
""".strip()

# IRCoT 2Wiki Prompt - 参考IterRetGen结构，内容与原IRCOT工程一致
IRCOT_2WIKI_PROMPT = """
You should think step by step and answer the question after <Question> based on given knowledge embraced with <doc> and </doc>. Your answer should be after <Answer> in JSON format with key "thought" and "answer", their value should be string.

Here are some examples for you to refer to:
<doc>
Wikipedia Title: Kurram Garhi
Kurram Garhi is a small village located near the city of Bannu, which is the part of Khyber Pakhtunkhwa province of Pakistan. Its population is approximately 35000. Barren hills are near this village. This village is on the border of Kurram Agency. Other nearby villages are Peppal, Surwangi and Amandi Kala.

Wikipedia Title: Satellite tournament
A satellite tournament is either a minor tournament or event on a competitive sporting tour or one of a group of such tournaments that form a series played in the same country or region.

Wikipedia Title: 1992 Copa Libertadores
The 1992 Copa Libertadores was the 33rd edition of the Copa Libertadores, CONMEBOL's annual international club tournament. São Paulo won the competition. The participating teams were divided into five groups, in which teams of the same country were placed in the same group. Each country was represented by two teams. The countries were paired as follows:

Wikipedia Title: Trojkrsti
Trojkrsti is a village in Municipality of Prilep, Republic of Macedonia.

Wikipedia Title: 2001–02 UEFA Champions League second group stage
Eight winners and eight runners- up from the first group stage were drawn into four groups of four teams, each containing two group winners and two runners- up. Teams from the same country or from the same first round group could not be drawn together. The top two teams in each group advanced to the quarter- finals.
</doc>
<Question>: Are both Kurram Garhi and Trojkrsti located in the same country?
Let's think step by step.
<Answer>:
```json
{{
    "thought": "Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country. So the answer is no.",
    "answer": "no"
}}
```

<doc>
Wikipedia Title: Laughter in Hell
Laughter in Hell is a 1933 American Pre-Code drama film directed by Edward L. Cahn and starring Pat O'Brien. The film's title was typical of the sensationalistic titles of many Pre-Code films. Adapted from the 1932 novel of the same name buy Jim Tully, the film was inspired in part by "I Am a Fugitive from a Chain Gang" and was part of a series of films depicting men in chain gangs following the success of that film. O'Brien plays a railroad engineer who kills his wife and her lover in a jealous rage and is sent to prison. The movie received a mixed review in "The New York Times" upon its release. Although long considered lost, the film was recently preserved and was screened at the American Cinematheque in Hollywood, CA in October 2012. The dead man's brother ends up being the warden of the prison and subjects O'Brien's character to significant abuse. O'Brien and several other characters revolt, killing the warden and escaping from the prison. The film drew controversy for its lynching scene where several black men were hanged. Contrary to reports, only blacks were hung in this scene, though the actual executions occurred off-camera (we see instead reaction shots of the guards and other prisoners). The "New Age" (an African American weekly newspaper) film critic praised the scene for being courageous enough to depict the atrocities that were occurring in some southern states.

Wikipedia Title: Ian Barry (director)
Ian Barry is an Australian director of film and TV.

Wikipedia Title: Thomas Scott (diver)
Thomas Scott( 1907- date of death unknown) was an English diver.

Wikipedia Title: Etan Boritzer
Etan Boritzer( born 1950) is an American writer of children 's literature who is best known for his book" What is God?" first published in 1989. His best selling" What is?" illustrated children's book series on character education and difficult subjects for children is a popular teaching guide for parents, teachers and child- life professionals. Boritzer gained national critical acclaim after" What is God?" was published in 1989 although the book has caused controversy from religious fundamentalists for its universalist views. The other current books in the" What is?" series include What is Love?, What is Death?, What is Beautiful?, What is Funny?, What is Right?, What is Peace?, What is Money?, What is Dreaming?, What is a Friend?, What is True?, What is a Family?, What is a Feeling?" The series is now also translated into 15 languages. Boritzer was first published in 1963 at the age of 13 when he wrote an essay in his English class at Wade Junior High School in the Bronx, New York on the assassination of John F. Kennedy. His essay was included in a special anthology by New York City public school children compiled and published by the New York City Department of Education.

Wikipedia Title: Edward L. Cahn
Edward L. Cahn (February 12, 1899 – August 25, 1963) was an American film director.
</doc>
<Question>: When did the director of film Laughter In Hell die?
Let's think step by step.
<Answer>:
```json
{{
    "thought": "The film Laughter In Hell was directed by Edward L. Cahn. Edward L. Cahn died on August 25, 1963. So the answer is August 25, 1963.",
    "answer": "August 25, 1963"
}}
```

<doc>
Wikipedia Title: Grand Duke Alexei Alexandrovich of Russia
Grand Duke Alexei Alexandrovich of Russia,(Russian: Алексей Александрович; 14 January 1850 (2 January O.S.) in St. Petersburg – 14 November 1908 in Paris) was the fifth child and the fourth son of Alexander II of Russia and his first wife Maria Alexandrovna (Marie of Hesse). Chosen for a naval career, Alexei Alexandrovich started his military training at the age of 7. By the age of 20 he had been appointed lieutenant of the Imperial Russian Navy and had visited all Russia's European military ports. In 1871, he was sent as a goodwill ambassador to the United States and Japan. In 1883 he was appointed general-admiral. He had a significant contribution in the equipment of the Russian navy with new ships and in modernizing the naval ports. However, after the Russian defeat in the Battle of Tsushima in 1905, he was relieved of his command. He died in Paris in 1908.

Wikipedia Title: Maria Alexandrovna (Marie of Hesse)
Maria Alexandrovna , born Princess Marie of Hesse and by Rhine (8 August 1824 – 3 June 1880) was Empress of Russia as the first wife of Emperor Alexander II. She was the mother of Emperor Alexander III and paternal grandmother of Nicholas II and Grand Duke Dmitri Pavlovich of Russia, one of the murderers of Rasputin. She was a daughter of Ludwig II, Grand Duke of Hesse, and Princess Wilhelmine of Baden. Marie was raised in austerity but was well educated by her mother, who took personal charge of her education but died when Marie was still very young. She was only fourteen years old when the Tsarevich Alexander Nikolaevich, later Tsar Alexander II of Russia, fell in love with her while he was traveling to Western Europe. She arrived in Russia in September 1840, converted to the Orthodox Church, took the title of Grand Duchess of Russia and traded the name Marie for Maria Alexandrovna. She married Alexander on 16 April 1841. The couple had eight children: two daughters and six sons. For fourteen years (1840–1855), she was Tsarevna, the wife of the heir of the Russian throne. She became the Russian Empress consort after the death of her father-in-law, Tsar Nicholas I. Maria Alexandrovna learned the Russian language quickly; she was pious and identified with her adopted country. She did not enjoy court life or the duties of representation as she was shy and of a withdrawn nature. As a consequence she was not popular. She took a more focused interest in charitable activities after the death of her mother-in-law the Dowager Empress Alexandra Feodorovna in 1860. Maria Alexandrovna was particularly active in the field of female education, establishing Russia's first all female schools. She organized the Russian Red Cross and expanded its activities during the Russo-Turkish War of 1877–78. She was deeply affected by the death of her eldest son the Tsarevich Nicholas Alexandrovich in 1865. By then, her fragile constitution was undermined by her numerous pregnancies and by tuberculosis which afflicted her since 1863. To avoid the harsh Russian winters, she spent long sojourns in the Crimea and in southern Europe. During many summers she visited her family in Jugenheim, where she had spent her childhood. Her marriage to Tsar Alexander II started as a love match and it was happy for some years, but Alexander II had many affairs and in 1866 he fell in love with Catherine Dolgorukova and had four children with his mistress. Maria Alexandrovna was treated with respect by her philanderer husband and she was much loved by her surviving children. After a long illness, she died in 1880. The Mariinsky Theatre in Saint Petersburg, the city of Mariinsk in Kemerovo Oblast, and the city of Mariehamn in Åland are named after her.

Wikipedia Title: Abdul-Vahed Niyazov
' of Russia.

Wikipedia Title: Aleksey Belevsky-Zhukovsky
Count Alexei Alexeevich Belevsky- Zhukovsky( 26 November 1871, Salzburg – c. 1931 Caucasus) was the son of Grand Duke Alexei Alexandrovich of Russia and Alexandra Vasilievna, Baroness Seggiano. He was also, being the son of Grand Duke Alexei, a grandson of Alexander II of Russia.

Wikipedia Title: Grand Duke Andrei Vladimirovich of Russia
Grand Duke Andrei Vladimirovich of Russia ((14 May 1879 – 30 October 1956) was a son of Grand Duke Vladimir Alexandrovich of Russia, a grandson of Emperor Alexander II and a first cousin of Nicholas II, Russia's last Tsar. In 1900, he began an affair with the famous ballerina Mathilde Kschessinska, becoming the third grand duke to fall for her. Grand Duke Andrei followed a military career and graduated from the Alexandrovskaya Military Law academy in 1905. He occupied different military positions during the reign of Tsar Nicholas II, but with no particular distinction. He became senator in 1911 and was appointed Major General in the Russian Army in 1915. He took part in World War I, but was away from real combat spending most of the conflict at Russia's headquarters or in idle time in Saint Petersburg. In February 1917, shortly before the fall of the Russian monarchy, Grand Duke Andrei left Saint Petersburg to join his mother in Kislovodsk. He remained in the Caucasus for the next three years. After the October Revolution he was briefly arrested along with his brother, Grand Duke Boris, but they escaped. He departed revolutionary Russia in March 1920, being the last grand duke to leave for exile. In 1921, he married his longtime mistress Mathilde Kschessinska and recognized her son as his.
</doc>
<Question>: What is the cause of death of Grand Duke Alexei Alexandrovich Of Russia's mother?
Let's think step by step.
<Answer>:
```json
{{
    "thought": "The mother of Grand Duke Alexei Alexandrovich of Russia is Maria Alexandrovna. Maria Alexandrovna died from tuberculosis. So the answer is tuberculosis.",
    "answer": "tuberculosis"
}}
```

Now based on the given doc, answer the question after <Question>.
<doc>
{context}
</doc>
Previous reasoning: {generation_so_far}
<Question>: {question}
Let's think step by step.
<Answer>:
""".strip()