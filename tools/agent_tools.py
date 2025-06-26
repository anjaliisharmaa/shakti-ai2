import os
import sys
from typing import Type
from pydantic import BaseModel
from crewai.tools import BaseTool
import traceback

# Add root to path to import agent cores
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# agent_tools.py
from tools.llm_utils import get_llm_response

from agentss.maaya.maaya_core import user_input as maaya_input
from agentss.gynika.gynika_core import user_input as gynika_input
from agentss.meher.meher_core import user_input as meher_input
from agentss.nyaya.nyaya_core import user_input as nyaya_input
from agentss.vaanya.vaanya_core import user_input as vaanya_input

class QueryInput(BaseModel):
    input: str



class AskMaayaTool(BaseTool):
    name: str = "ask_maaya_tool"
    description: str = "Ask Maaya about pregnancy and baby care"
    args_schema: Type[BaseModel] = QueryInput

    def _run(self, input: str) -> str:
        print(f"[TOOL LOG] Calling Maaya with input: {input}")
        os.environ["LITELLM_PROVIDER"] = "disabled"
        return maaya_input(input)

class AskGynikaTool(BaseTool):
    name: str = "ask_gynika_tool"
    description: str = "Ask Gynika about reproductive health"
    args_schema: Type[BaseModel] = QueryInput

    def _run(self, input: str) -> str:
        print(f"[TOOL LOG] Calling Gynika with input: {input}")
        os.environ["LITELLM_PROVIDER"] = "disabled"
        return gynika_input(input)

class AskMeherTool(BaseTool):
    name: str = "ask_meher_tool"
    description: str = "Ask Meher for mental health or abuse support"
    args_schema: Type[BaseModel] = QueryInput

    def _run(self, input: str) -> str:
        print(f"[TOOL LOG] Calling Meher with input: {input}")
        os.environ["LITELLM_PROVIDER"] = "disabled"
        return meher_input(input)

from tools.llm_utils import get_llm_response  # Your LiteLLM wrapper

class AskNyayaTool(BaseTool):
    name: str = "ask_nyaya_tool"
    description: str = "Ask Nyaya about legal and ethical issues in healthcare"
    args_schema: Type[BaseModel] = QueryInput

    def _run(self, input: str) -> str:
        print("⚠️ [AskNyayaTool] Starting tool with query:", input)
        try:
            response = get_llm_response([
                {"role": "user", "content": input}
            ])  # Pass as messages list
            print("🎯 [AskNyayaTool] Got result:", response)
            return response
        except Exception as e:
            print("❌ [AskNyayaTool] Error:", str(e))
            traceback.print_exc()
            return f"Nyaya failed with error: {str(e)}"


class AskVaanyaTool(BaseTool):
    name: str = "ask_vaanya_tool"
    description: str = "Ask Vaanya about menopause and hormonal changes"
    args_schema: Type[BaseModel] = QueryInput

    def _run(self, input: str) -> str:
        print(f"[TOOL LOG] Calling Vaanya with input: {input}")
        os.environ["LITELLM_PROVIDER"] = "disabled"
        return vaanya_input(input)


# export tools to use in tasks.py or crew_setup.py
ask_maaya_tool = AskMaayaTool()
ask_gynika_tool = AskGynikaTool()
ask_meher_tool = AskMeherTool()
ask_nyaya_tool = AskNyayaTool()
ask_vaanya_tool = AskVaanyaTool()
