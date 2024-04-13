"""All prompts used for DFC prompting."""

DECONTEXTILISATION_PROMPT = """Decompose and decontextualise a document into independently meaningful sentences. This process will make each sentence stand alone that can be verified independently.

Input: Mary is a five-year old girl. She likes playing piano. She doesn't like cookies. 
Output: 
Mary is a five-year old girl.
Mary likes playing piano.
Mary doesn't like cookies. 

Input: Google began as an online search firm, but it now offers more than 50 Internet services and products, from e-mail and online document creation to software for mobile phones and tablet computers. In addition, its 2012 acquisition of Motorola Mobility put it in the position to sell hardware in the form of mobile phones. 
Ouput: 
Google began as an online search firm.
Google now offers more than 50 Internet services and products.
Google offers from e-mail and online document creation to software for mobile phones and tablet computers.
Google 2012 acquisition of Motorola Mobility put it in the position to sell hardware in the form of mobile phones.

Input: """


CHECK_WORTHINESS_LABEL_ONLY_PROMPT = """Identify whether this claim is an opinion or factual, and whether it is checkworthy or not in the context of fact-checking. Just return two labels without explanation.
I think Apple is a good company.
opinon, not checkworthy
Preslav is a professor in MBZUAI.
factual, checkworthy
Friends is a great TV series.
opinion, not checkworthy
The Stanford Prison Experiment was conducted in the basement of Encina Hall.
factual, checkworthy
"""


ENTITY_EXTRACTION_PROMPT = """Extract all entities of a claim.
Input: Google now offers more than 50 Internet services and products.
Output: Google, Internet services, product
Input: Donald John Trump is an American politician, media personality, and businessman.
Output: Donald John Trump, American politician, media personality, businessman
Input: """       


QGEN_PROMPT = """Give a list of queries using for searching related information for a claim. 
Input: Google now offers more than 50 Internet services and products. 
Output: What does Google offers now? 
How many service and product does Google offer? 
Google, more than 50 Internet services, products 
Input: Donald John Trump is an American politician, media personality, and businessman.
Output: Who is Donald John Trump?
Give information of Donald John Trump. 
Donald John Trump, American politician 
Donald John Trump, media personality 
Donald John Trump, businessman
Input: """


STANCE_DETECTION_PROMPT = """Determine whether the evidence support the claim or not. Choose label from [support, partial support, refute, other] and explain why. 
Support means we can entail the claim by the evidence.
Partial support means: part of the information presented in the claim appear in the evidence.
Refute means that the evidence mention the same event as the claim, but a clear opposite fact. It should be highlighed that under refute, the evidence mentions the fact in the claim, they are closely relevant, but opposite meaning or stance.
Other means the evidence does not mention anything about the fact described in the claim, such that it neither supports nor refutes the claim.

Claim: Elon Musk is the founder, CEO and chief engineer of SpaceX.
Evidence: Elon Musk is the owner and CEO of Twitter, and he is also the founder, CEO and chief engineer of SpaceX.
Stance: support, statement 'he is also the founder, CEO and chief engineer of SpaceX' in evidence above supports the claim.

Claim: Elon Musk is the owner and CEO of Twitter, and he is also the founder, CEO and chief engineer of SpaceX.
Evidence: Elon Musk is the founder, CEO and chief engineer of SpaceX.
Stance: partial support.

Claim: Steve Jobs is the founder, CEO and chief engineer of SpaceX.
Evidence: Elon Musk is the owner and CEO of Twitter, and he is also the founder, CEO and chief engineer of SpaceX.
Stance: refute.

Claim: Elon Musk is a professor in The Stanford University.
Evidence: Elon Musk is the owner and CEO of Twitter, and he is also the founder, CEO and chief engineer of SpaceX.
Stance: other, according to the evidence, I cannot judge whether the claim is true or not, not enough information, the evidence neither supports nor refutes.

Claim: On January 6, 2021, a mob of supporters of former President Donald Trump stormed the U.S. Capitol in an attempt to overturn the 2020 presidential election.
Evidence: On January 6, 2021, following the defeat of U.S. President Donald Trump in the 2020 presidential election, a mob of his supporters attacked the United States Capitol Building in Washington, D.C. The mob sought to keep Trump in power by preventing a joint session of Congress from counting the electoral college votes to formalize the victory of President-elect Joe Biden.
Stance: support.

Claim: The 2021 Capitol Hill riots resulted in the deaths of five people, including a Capitol police officer. 
Evidence: Five people died either shortly before, during, or following the riot: one was shot by Capitol Police, another died of a drug overdose, and three died of natural causes.
Stance: partial support, the evidence supports that fact that five deaths, but not sure whether they include a Capitol police officer or not.

Claim: More than 300 people have been charged with crimes related to the riots. 
Evidence: As of November 10, 2022, over 940 people had been charged in the Capitol breach. 
Stance: refute, evidence and claim are describing the same thing, the number of people who was charged is over 940, while more than 300 in the claim, so the evidence refutes the claim.

Claim: More than 300 people have been charged with crimes related to the riots. 
Evidence: The laptop computer taken from Pelosi's office was taken by 22-year-old Capitol rioter Riley Williams. Williams was arrested and indicted on eight counts, including theft of government property, obstructing an official proceeding, and assaulting or resisting police.
Stance: other, the evidence demonstrates something relevent to the fact in the claim, but it does not support or refute any information of it. 

Claim: {}
Evidence: {}
Stance: """


EDITOR_PROMPT = """Fix the claim according to the evidence.

Claim: Your nose switches back and forth between nostrils. When you sleep, you switch about every 45 minutes. This is to prevent a buildup of mucus. It’s called the nasal cycle.
Evidence: Although we don’t usually notice it, during the nasal cycle one nostril becomes congested and thus contributes less to airflow, while the other becomes decongested. On average, the congestion pattern switches about every 2 hours, according to a small 2016 study published in the journal PLOS One.
This suggests 45 minutes switch time in your statement is wrong.
Fix: Your nose switches back and forth between nostrils. When you sleep, you switch about every 2 hours. This is to prevent a buildup of mucus. It’s called the nasal cycle.

Claim: In the battles of Lexington and Concord, the British side was led by General Thomas Hall.
Evidence: Interesting Facts about the Battles of Lexington and Concord. The British were led by Lieutenant Colonel Francis Smith. There were 700 British regulars.
This suggests General Thomas Hall in your statement is wrong.
Fix: In the battles of Lexington and Concord, the British side was led by Lieutenant Colonel Francis Smith.

Claim: The Stanford Prison Experiment was conducted in the basement of Encina Hall, Stanford’s psychology building.
Evidence: Carried out August 15-21, 1971 in the basement of Jordan Hall, the Stanford Prison Experiment set out to examine the psychological effects of authority and powerlessness in a prison environment.
This suggests Encina Hall in your statement is wrong.
Fix: The Stanford Prison Experiment was conducted in the basement of Jordan Hall, Stanford’s psychology building.

Claim: The Havel-Hakimi algorithm is an algorithm for converting the adjacency matrix of a graph into its adjacency list. It is named after Vaclav Havel and Samih Hakimi.
Evidence: The Havel-Hakimi algorithm constructs a special solution if a simple graph for the given degree sequence exists, or proves that one cannot find a positive answer. This construction is based on a recursive algorithm. The algorithm was published by Havel (1955), and later by Hakimi (1962).
This suggests the Havel-Hakimi algorithm’s functionality in your statement is wrong.
Fix: The Havel-Hakimi algorithm constructs a special solution if a simple graph for the given degree sequence exists, or proves that one cannot find a positive answer. It is named after Vaclav Havel and Samih Hakimi.

Claim: "Time of My Life" is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing. The song was produced by Phil Ramone.
Evidence: On September 8, 2010, the original demo of this song, along with a remix by producer Michael Lloyd , was released as digital files in an effort to raise money for the Patrick Swayze Pancreas Cancer Resarch Foundation at Stanford University.
This suggests "Time of My Life" producer name in your statement is wrong.
Fix: "Time of My Life" is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing. The song was produced by Michael Lloyd.

Claim: Phoenix Market City Pune is located on 21 acres of prime property in Pune. It is spread across four levels with approximately 1.4 million square feet of built-up space. The mall is owned and operated by Phoenix Mills Limited.
Evidence: Phoenix Market City was opened in January 2013 and has the distinction of being the largest mall in the city of Pune, with the area of 3.4 million square feet. It is located in the Viman Nagar area of Pune.
This suggests the 1.4 million square feet of built-up space in your statment is wrong.
Fix: Phoenix Market City Pune is located on 21 acres of prime property in Pune. It is spread across four levels with approximately 3.4 million square feet of built-up space. The mall is owned and operated by Phoenix Mills Limited.

Claim: {claim}
Evidence: {evidence}
This suggests
""".strip()
