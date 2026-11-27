document: Software Engineering
chunks: Chunk size is 5 and usually each chunk contains about 5 sentences
step limit: 9
model: openai/gpt-oss-20b

Student Chatbot evaluation tests: 11/15 passed

1. What is Agile model?
PASS

2. What are four Agile values?
FAIL
Missing: [‘documentation’]
Got: The four Agile values are individuals and interactions over processes and tools, working software, customer collaboration, and responding to change over following a plan.

3. What is the Waterfall model?
PASS

4. What is Software Engineering?
PASS

5. Why is Software Engineering important?
FAIL
Missing: [‘time’]
Got: Software Engineering is important because it imposes discipline on software development, ensures quality software, and helps build complex systems reliably and efficiently.

6. What are the two roles of software?
PASS

7. What are the characteristics of software?
FAIL
Missing: [‘wear and tear’]
Got: Software is intangible, complex, evolves over time, and requires high dependability. It needs continuous updates to remain useful.

8. What is Legacy Software?
PASS

9. What are Software Myths?
PASS

10. What are layers of Software Engineering
PASS

11. What is a Software Process Framework?
FAIL
Missing: [‘umbrella activities’]
Got: A Software Process Framework is an organized approach combining processes, methods, and tools to guide software development. It includes framework activities and task sets.

12. What are the five activities of the Generic Process Model?
PASS

13. What is subnetting?
PASS

14. What is Reactjs?
PASS

15. What is binary search tree?
PASS

Ways to improve:
Increase/experiment with number of chunks
Alternative phrases for keywords in tests
Improve system prompt to be more specific for each question
