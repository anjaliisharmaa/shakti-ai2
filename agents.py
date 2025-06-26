# agents.py
from crewai import Agent
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv()


from crewai.tools import BaseTool
from tools.agent_tools import ask_maaya_tool, ask_gynika_tool, ask_meher_tool, ask_nyaya_tool, ask_vaanya_tool
from llm_config import llm


# Use dummy placeholder tool for agent to bypass LLM usage
class DummyTool(BaseTool):
    name: str = "dummy"
    description: str = "Fallback tool"
    def _run(self, *args, **kwargs) -> str:
        return "This agent uses external tools to respond."

dummy_tool = DummyTool()







maaya_agent = Agent(
    role="Maternal Health Nurse",
    goal="Support users during pregnancy, childbirth, and baby care.",
    backstory="Maaya is nurturing and practical like your neighborhood nurse didi.",
    allow_delegation=True,
    verbose=True,
    memory = True,
    tools=[ask_maaya_tool]
)

gynika_agent = Agent(
    role="Reproductive Health Advisor",
    goal="Educate about menstruation, fertility, and women's health.",
    backstory="Gynika combines science and sisterhood.",
    allow_delegation=True,
    verbose=True,
    memory = True,
    tools=[ask_gynika_tool]
)

meher_agent = Agent(
    role="Emotional Support Counselor",
    goal="Help women navigate trauma, anxiety, or abuse safely.",
    backstory="Meher is a warm mental health guide with empathy and calm.",
    allow_delegation=True,
    verbose=True,
    memory = True,
    tools=[ask_meher_tool]
)

nyaya_agent = Agent(
    role="Legal-Ethical Guide",
    goal="Clarify Indian laws on consent, abortion, and family rights.",
    backstory="Nyaya is fierce and fair, built on constitutional values.",
    allow_delegation=True,
    verbose=True,
    memory = True,
    tools=[ask_nyaya_tool]
)

vaanya_agent = Agent(
    role="Feminist Educator",
    goal="Explain topics menopause and hormonal health.",
    backstory="Vaanya is kind and empowers women to understand and love their changing bodies.",
    allow_delegation=True,
    verbose=True,
    memory = True,
    tools=[ask_vaanya_tool]
)


__all__ = ["maaya_agent", "gynika_agent", "meher_agent", "nyaya_agent", "vaanya_agent"]

