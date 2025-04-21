from jupyter_ai.personas.base_persona import PersonaDefaults
from .agno_persona import AgnoPersona

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from .fd import FinancialDatasetsTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.reasoning import ReasoningTools
from agno.memory.v2.memory import Memory
from agno.memory.v2.db.sqlite import SqliteMemoryDb


memory_db = SqliteMemoryDb(table_name="memory", db_file=".memory.db")


class DebugPersona(AgnoPersona):


    def create_agent(self):
        return Agent(
            name="Financial Analyst",
            model=OpenAIChat(id="gpt-4.1"),
            memory=Memory(db=memory_db),
            enable_user_memories=True,
            enable_agentic_memory=True,
            # add_history_to_messages=True,
            # num_history_runs=3,
            tools=[
                YFinanceTools(
                    # stock_fundamentals=True,
                    # income_statements=True,
                    # key_financial_ratios=True,
                    analyst_recommendations=True,
                    company_news=True,
                    # technical_indicators=True,
                    # historical_prices=True
                ),
                FinancialDatasetsTools(
                    enable_company_info = True,
                    enable_financial_metrics = True,
                    enable_financial_statements = True,
                    enable_prices = True,
                    enable_news = True,
                    enable_sec_filings = False
                ),
                ReasoningTools(
                    think=True,
                    analyze=True,
                    add_instructions=True
                )
            ],
            description="You are a financial data specialist that helps analyze financial information for stocks.",
            instructions=[
                "When given a financial query:",
                "1. Use appropriate Financial datasets methods based on the query type",
                "2. Format financial data clearly and highlight key metrics",
                "3. For financial statements, compare important metrics with previous periods when relevant",
                "4. Calculate growth rates and trends when appropriate",
                "5. Handle errors gracefully and provide meaningful feedback",
                "6. Don't use tables to show results, instead use bullet points with narrative text.",
                "7. Give data to support all of your statements.",
                "8. Track your citations and references and include those in the final report.",
                "9. Use appropriate emojis in your responses to help the user understand what is going on."

            ],
            markdown=True,
            show_tool_calls=True,
            add_datetime_to_instructions=True,
            stream=True
        )

    @property
    def defaults(self):
        return PersonaDefaults(
            name="Analyst",
            avatar_path="/api/ai/static/jupyternaut.svg",
            description="A trader persona.",
            system_prompt="...",
        )
