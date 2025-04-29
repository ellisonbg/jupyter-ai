from textwrap import dedent

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


class SchedulerPersona(AgnoPersona):
    """
    The debug persona, the main persona provided by Jupyter AI.
    """

    def create_agent(self):
        return Agent(
            name="Work Scheduler",
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
                    enable_prices = True
                ),
                ReasoningTools(
                    think=True,
                    analyze=True,
                    add_instructions=True
                )
            ],
            description=dedent("""\
                You workload scheduler that helps the analyst and trader by scheduling work
                (trades or analyst reports) to run in the future or on a periodic schedule.
            """),
            instructions=[
                "When you are asked to schedule a trade or analyst report:",
                "1. First find out what you are being asked to schedule (trade or analyst report)",
                "2. Next find out when you are being asked to schedule that work. Convert all times to EST.",
                "3. You may need to look at the previous message to find some of the details.",
                "4. Next, summarize the what and when of the work you will schedule.",
                "5. Finally, give the user confirmation that the work is scheduled.",
                "8. Use appropriate emojis in your responses to help the user understand what is going on."
            ],
            markdown=True,
            show_tool_calls=True,
            add_datetime_to_instructions=True,
            stream=True
        )

    @property
    def defaults(self):
        return PersonaDefaults(
            name="Scheduler",
            avatar_path="/api/ai/static/jupyternaut.svg",
            description="Schedule workloads for finance.",
            system_prompt="...",
        )
