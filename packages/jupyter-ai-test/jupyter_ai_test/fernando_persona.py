from jupyter_ai.personas.base_persona import PersonaDefaults
from .agno_persona import AgnoPersona

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.reasoning import ReasoningTools
from agno.memory.v2.memory import Memory
from agno.memory.v2.db.sqlite import SqliteMemoryDb


memory_db = SqliteMemoryDb(table_name="memory", db_file=".memory.db")


class FernandoPersona(AgnoPersona):


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
                ReasoningTools(
                    think=True,
                    analyze=True,
                    add_instructions=True
                )
            ],
            description="You are Fernando Pérez. You created IPython and are a co-creator of Project Jupyter. You are from Colombia originally. You have a Ph.D. in Physics of CU Boulder and are a professor in the UC Berkeley Stats department.",
            instructions=[
                "Reply as Fernando Perez would.",
                "Keep your replies as short as possible and human sounding."
            ],
            markdown=True,
            add_datetime_to_instructions=True,
            stream=True
        )

    @property
    def defaults(self):
        return PersonaDefaults(
            name="Fernando Pérez",
            avatar_path="/api/ai/static/jupyternaut.svg",
            description="A human named Fernando Pérez",
            system_prompt="...",
        )
