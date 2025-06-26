# crew_setup.py
from crewai import Crew
from agents import meher_agent, maaya_agent, gynika_agent, nyaya_agent, vaanya_agent
from tasks import ask_user_question
import os

# Disable telemetry
os.environ["OTEL_SDK_DISABLED"] = "true"

# Create the crew with required agents and tasks
crew = Crew(
    agents=[meher_agent, maaya_agent, gynika_agent, nyaya_agent, vaanya_agent],
    tasks=[ask_user_question],
    verbose=True,
    telemetry=False
)

def ask_shakti_ai(query):
    try:
        result = crew.kickoff(inputs={"input": query})
        return result
    except Exception as e:
        return f"Error: {str(e)}"
    


if __name__ == "__main__":
    query = "How can a pregnant woman register for Janani Suraksha Yojana?"
    print("💬 Sending query to SHAKTI AI...")
    result = ask_shakti_ai(query)
    print("✅ Final Answer:\n", result)
