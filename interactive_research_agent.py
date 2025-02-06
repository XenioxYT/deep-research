from deep_research_agent import DeepResearchAgent
import asyncio
import os
import json
from typing import Dict, Set, Optional

class InteractiveResearchAgent:
    def __init__(self):
        self.previous_report: Optional[str] = None
        self.high_quality_urls: Dict[str, dict] = {}  # URLs with score > 0.85
        self.agent = None

    async def initialize(self):
        """Initialize the underlying research agent."""
        self.agent = DeepResearchAgent()
        await self.agent.__aenter__()
        return self

    async def cleanup(self):
        """Cleanup resources."""
        if self.agent:
            await self.agent.__aexit__(None, None, None)

    async def __aenter__(self):
        return await self.initialize()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

    def store_high_quality_context(self):
        """Store URLs and their data that had a score > 0.85."""
        self.high_quality_urls = {
            url: data for url, data in self.agent.high_ranking_urls.items()
            if data['score'] > 0.85
        }

    def read_previous_report(self, report_path: str) -> str:
        """Read the content of the previous report."""
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading previous report: {e}")
            return ""

    def get_user_choice(self) -> str:
        """Get user's choice for follow-up or new chat."""
        while True:
            choice = input("\nWould you like to:\n1. Ask a follow-up question\n2. Start a new chat\nEnter 1 or 2: ").strip()
            if choice in ['1', '2']:
                return choice
            print("Invalid choice. Please enter 1 or 2.")

    async def run_interactive_session(self):
        """Run the interactive research session."""
        try:
            while True:
                # Get the initial/new query
                query = input("\nEnter your research query: ").strip()
                if not query:
                    continue

                # Run the research
                if self.previous_report and self.high_quality_urls:
                    # Add context to the agent for follow-up questions
                    self.agent.high_ranking_urls = self.high_quality_urls
                    print("\nUsing context from previous research...")

                # Run the research and get the report path
                result = await self.agent.research(query)
                print("\nResearch Result:")
                print(result)

                # Extract the report path from the result
                report_path = None
                if "Report has been generated and saved to:" in result:
                    report_path = result.split(": ")[-1].strip()

                if report_path and os.path.exists(report_path):
                    # Store high-quality context for potential follow-up
                    self.store_high_quality_context()
                    # Store the report content
                    self.previous_report = self.read_previous_report(report_path)

                    # Ask user for next action
                    choice = self.get_user_choice()
                    
                    if choice == '2':  # New chat
                        # Reset context
                        self.previous_report = None
                        self.high_quality_urls = {}
                        self.agent.reset_state()
                    # If choice is 1 (follow-up), we continue with the loop keeping the context
                else:
                    print("\nError: Report file not found or research failed.")
                    # Reset context on error
                    self.previous_report = None
                    self.high_quality_urls = {}
                    self.agent.reset_state()

        except KeyboardInterrupt:
            print("\nExiting interactive session...")
        except Exception as e:
            print(f"\nError in interactive session: {e}")

def main():
    """Run the interactive research agent."""
    async def run():
        async with InteractiveResearchAgent() as agent:
            await agent.run_interactive_session()

    asyncio.run(run())

if __name__ == "__main__":
    main() 