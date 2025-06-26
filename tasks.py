#from crewai import Agent
from crewai import Task
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from agents import meher_agent, maaya_agent, gynika_agent, nyaya_agent, vaanya_agent
from dotenv import load_dotenv
load_dotenv()


from langchain_google_genai import ChatGoogleGenerativeAI





from tools.agent_tools import (
    ask_maaya_tool,
    ask_gynika_tool,
    ask_meher_tool,
    ask_nyaya_tool,  
    ask_vaanya_tool
)


ask_user_question = Task(
    description="Answer this user query compassionately and correctly: {input}",
    expected_output="A warm, inclusive, legally and medically accurate response in 3–4 paras.",
    tools=[
        ask_nyaya_tool
    ],
    agent=nyaya_agent,
    async_execution=False,

)






