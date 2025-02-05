**Title:**  Deep Research Agent - AI-Powered In-Depth Information Retrieval and Report Generation

**Goal:** Develop a Python program that emulates an expert researcher, conducting in-depth research on a user-provided query, leveraging AI models and web search to generate a comprehensive report.

**Core Functionality:**

The program operates in a cyclical manner, iteratively refining its understanding of the query and expanding its knowledge base until a satisfactory answer is achieved.

**Detailed Workflow (Iterative Loop):**

1. **Query Input:**
    *   The program begins by accepting a user-defined query (text input).
    *   Consider error handling for invalid or empty inputs.

2. **Sub-Query Generation (AI Model 1):**
    *   **Prompt for AI Model 1:**
        *   "Given the user's query: '{user_query}', generate a list of 10-20 specific and diverse sub-queries that would be helpful in thoroughly researching this topic. These sub-queries should explore different facets of the main query, aiming to cover a wide range of relevant information. Focus on questions that are likely to have answers available on the web, and that require some research."
    *   **Implementation:**
        *   Utilize an AI model (e.g., Google Gemini) through its API.
        *   Parse the AI's output to extract a clean list of sub-queries.
        *   Consider adding logic to remove duplicate or highly similar sub-queries.

3. **Web Search:**
    *   **Prompt for Search Engine:**
        *   For each sub-query generated, construct a well-formed search query for a chosen search engine (e.g., Google).
    *   **Implementation:**
        *   Use a search engine API (e.g., Google Custom Search JSON API) or a web scraping library like `requests` and `Beautiful Soup` to perform the searches.
        *   Retrieve the top N search results (e.g., top 5-10) for each sub-query.
        *   Store the following information for each search result:
            *   Title
            *   URL
            *   Snippet (short description)
            *   (Optional) Source domain/website

4. **Relevance Ranking (AI Model 2):**
    *   **Prompt for AI Model 2:**
        *   "You are an expert research assistant. I will provide you with a user's main query and a list of search results. Rank these search results based on their relevance to the user's query. Consider the title, snippet, and potentially the source domain. Provide a score between 0 and 1 for each result, where 1 is highly relevant and 0 is not relevant at all. Organize your response as follows: URL: [URL], Score: [Score]"
        *   **Input to AI Model 2:**
            *   User's original query.
            *   A list of search results (title, URL, snippet) from a single sub-query (you can either provide the sub-query to the model or just the search results).
        *   **Output from AI Model 2:**
            *   A list of the same search results, but now with a relevance score assigned to each.
    *   **Implementation:**
        *   Use an AI model (potentially the same as in step 2, or a different one specialized in ranking/evaluation).
        *   Parse the AI's output to extract the relevance scores.

5. **Deep Dive Decision (AI Model 3):**
    *   **Prompt for AI Model 3:**
        *   "Based on the user's query '{user_query}', the sub-query '{sub_query}', and the following ranked search results: {list_of_results_with_scores}, determine which URLs (if any) warrant further investigation. Prioritize URLs with high relevance scores and those that likely contain in-depth information. Select up to 3 URLs to explore further. Output the selected URLs, one per line."
    *   **Input to AI Model 3:**
        *   User's original query
        *   The current sub-query
        *   The list of ranked search results (from step 4) for the current sub-query.
    *   **Output from AI Model 3:**
        *   A list of URLs selected for deep diving.
    *   **Implementation:**
        *   Potentially combine the ranking scores with logic that evaluates the potential "depth" of information a URL might contain (e.g., looking at domain authority, length of snippets, etc. - this part might need manual refinement).
        *   Consider setting a threshold to prevent deep diving into irrelevant pages.

6. **Web Scraping and Content Extraction:**
    *   **Implementation:**
        *   For each URL selected for deep diving, use web scraping libraries (e.g., `requests`, `Beautiful Soup`, `Scrapy`) to retrieve the full HTML content of the web page.
        *   Extract the relevant text content from the HTML (e.g., paragraphs, headings, lists).
        *   Filter out irrelevant content like ads, navigation menus, etc.
        *   Store the extracted text content, associating it with the original URL.

7. **Recursive Link Analysis:**
    *   **Implementation:**
        *   Within the scraped content of each deep-dived page, identify and extract all internal and external hyperlinks.
        *   For each extracted link:
            *   **Prompt for AI Model 4** (Link Prioritization - can be combined with model 3):
                *   "Given the user's query '{user_query}' and the context of the page from which it was extracted '{scraped_page_snippet}', evaluate the relevance of this link: {link_url}. Does this link potentially lead to valuable information related to the query? Respond with a score between 0 and 1, where 1 is highly relevant and 0 is not relevant."
            *   Filter links based on the relevance score (e.g., only consider links above a certain threshold).
            *   Add the highly relevant links to a queue for potential future deep diving, making the process recursive.

8. **Iteration and Termination:**
    *   **Stopping Criteria (AI Model 5 - the "Satisfied" Model):**
        *   **Prompt for AI Model 5:**
            *   "You are a research expert assessing the completeness of research performed on the query: '{user_query}'. You have access to the following information gathered so far: {summary_of_gathered_information}. Has enough research been conducted to provide a comprehensive and insightful answer to the query? If yes, respond with 'STOP'. If not, respond with 'CONTINUE'."
            *   **Input:**
                *   Original user query
                *   A concise summary of the information gathered so far (this summary needs to be generated - see next point).
        *   **Implementation:**
            *   The system should maintain a summary of the gathered information. This could be a list of key findings, a running text summary, or a combination of both. An AI model could be used to periodically condense the extracted content into a summary.
            *   Periodically (e.g., after each round of deep diving or after a certain number of iterations), query AI Model 5 to determine if the research is sufficient.
        *   **Alternative Stopping Criteria:**
            *   Maximum iteration depth reached.
            *   Maximum number of URLs scraped.
            *   Time limit exceeded.

9. **Report Generation (AI Model 6):**
    *   **Prompt for AI Model 6:**
        *   "Generate a comprehensive research report on the following query: '{user_query}'. Use the provided information gathered during the research process: {all_gathered_information}. Organize the report in a clear and logical manner, using headings, subheadings, and bullet points where appropriate. Include a summary of the key findings, and provide citations or links to the sources of information. The report should be detailed, well-structured, and easy to understand. Also, include a section detailing the process taken to generate the answer, including the subqueries generated, the urls chosen for further analysis, etc."
    *   **Input:**
        *   Original user query.
        *   All the information gathered during the research process (extracted text content, summaries, URLs).
    *   **Output:**
        *   A well-formatted research report in a human-readable format (e.g., plain text, Markdown, HTML).
    *   **Implementation:**
        *   Use an AI model capable of generating long-form text and structuring it effectively.
        *   Include proper citations or links to the source material.
        *   Consider adding a section that outlines the research methodology used (sub-queries generated, URLs explored, etc.).

10. **User Interaction:**
    *   After the report is generated, prompt the user for a new query or an option to exit the program.

**Enhancements and Considerations:**

*   **Caching:** Implement caching for web pages and AI responses to reduce API calls and improve speed.
*   **Error Handling:** Robust error handling for API calls, web scraping, and AI responses is crucial.
*   **Rate Limiting:** Respect API rate limits and website robots.txt rules to avoid being blocked.
*   **User Interface:** Consider a more user-friendly interface (e.g., command-line interface with progress indicators, a web-based interface).
*   **Fact-Checking:** Integrate a mechanism for fact-checking claims made in the extracted content (potentially another AI model or a dedicated fact-checking API).
*   **Bias Detection:**  Be mindful of potential biases in the AI models and the sources being scraped. Implement techniques to mitigate bias if possible.
*   **Modularity:**  Design the code in a modular way to make it easier to maintain, update, and expand.
*   **Context Window:** Be aware of the context window limitations of the chosen AI models. Consider using techniques like chunking or summarization to handle large amounts of text.
*   **Different Models:** Experiment with different AI models for different tasks to find the best combination for optimal performance.

**Technology Stack:**

*   **Programming Language:** Python
*   **AI Models:** Google Gemini API, potentially other models through their respective APIs (OpenAI, etc.)
*   **Search Engine:** Google Custom Search JSON API or similar
*   **Web Scraping:** `requests`, `Beautiful Soup`, `Scrapy`
*   **Data Storage:** Potentially use a database (e.g., SQLite, PostgreSQL) or store data in structured files (e.g., JSON, CSV).
*   **Other Libraries:** `time` (for delays), `random` (for varying delays), `os` (for file system operations), `re` (for regular expressions).