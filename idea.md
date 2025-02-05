I want to make a python program that performs "deep research" on a given query. Essentially, it's like Google but goes deeper and uses an AI model to answer the questions.

The loop is like this:
1. User inputs a query
2. The program uses an AI model (like google gemini for quick and free) to geenerate a list of 10-20 search queries that are related to the query.
3. The program then uses a search engine like google to search for the queries.
4. The search results are returned to another AI model which ranks them based on the relevance to the query.
5. Another AI call is made where it can choose to go deeper on a certain result
6. These results (that the ai chooses) are then scraped from the websites related to the search results
7. Any links on those webpages will then be submitted for analysis from the AI etc etc.
8. repeat until the AI is satisfied with the answer.
9. Generate a report with the results and the process used to generate the answer.
10. Wait for user input again