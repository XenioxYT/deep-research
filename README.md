# Deep Research Agent

An AI-powered research assistant that performs in-depth information retrieval and generates comprehensive reports on any given query.

## Features

- Generates diverse sub-queries to explore different aspects of the main research topic
- Performs web searches using Google Custom Search API
- Ranks search results based on relevance using AI
- Extracts and processes content from web pages
- Generates detailed research reports with proper citations

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root with the following API keys:
```
GOOGLE_AI_KEY=your_google_ai_api_key
GOOGLE_SEARCH_KEY=your_google_custom_search_api_key
GOOGLE_SEARCH_ENGINE_ID=your_google_custom_search_engine_id
```

To obtain the required API keys:
- Google AI (Gemini) API key: Visit https://makersuite.google.com/app/apikey
- Google Custom Search API key and Search Engine ID: Visit https://programmablesearchengine.google.com/

## Usage

Run the script:
```bash
python deep_research_agent.py
```

When prompted, enter your research query. The agent will:
1. Generate relevant sub-queries
2. Search the web for information
3. Extract and analyze content
4. Generate a comprehensive report

## Example

```python
from deep_research_agent import DeepResearchAgent

agent = DeepResearchAgent()
report = agent.research("What are the latest developments in quantum computing?")
print(report)
```

## Limitations

- Requires valid API keys for Google services
- Web scraping may be blocked by some websites
- Content extraction quality depends on website structure
- API rate limits may apply

## Contributing

Feel free to submit issues and enhancement requests! 