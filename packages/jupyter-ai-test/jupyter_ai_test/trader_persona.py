from jupyter_ai.personas.base_persona import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from .fd import FinancialDatasetsTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.reasoning import ReasoningTools
from agno.memory.v2.memory import Memory
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.models.message import Message as AgnoMessage

async def transform_iterator(source_iterator):
    async for item in source_iterator:
        yield item.content.replace('$',r'\\$')


memory_db = SqliteMemoryDb(table_name="memory", db_file=".memory.db")


class TraderPersona(BasePersona):
    """
    The debug persona, the main persona provided by Jupyter AI.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = Agent(
            name="Stock Trader",
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
            description="You are a stock trader that executes trades.",
            instructions=[
                "When you are given a specific trade or trading strategy:",
                "1. First provide a concise summary of the exact trades that will be made.",
                "2. Summarize the current price of the asset, the number of shares or the $ amount of the transaction.",
                "3. Summarize the type of transaction.",
                "5. Before you move forward with the actual trade, ask the user to reply with CONFIRM",
                "6. If you see a message history with CONFIRM and a summary of a trade, execute the trade.",
                "7. Instead of actually executing the trade on the market. Pretend to do so (this is a demo) and summarize the transaction."
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
            name="Trader",
            avatar_path="/api/ai/static/jupyternaut.svg",
            description="A mock persona used for debugging in local dev environments.",
            system_prompt="...",
        )

    def get_role(self, ychat_message):
        if ychat_message.sender in self.manager.personas:
            return 'assistant'
        else:
            return 'user'

    async def process_message(self, message: Message):
        history = self.ychat.get_messages()[-4:]
        messages = [AgnoMessage(content=h.body, name=h.sender, role=self.get_role(h)) for h in history]
        stream = await self.agent.arun(
            message.body,
            user_id=message.sender,
            session_id=self.ychat.get_id(),
            messages=messages
        )
        await self.forward_reply_stream(transform_iterator(stream))
