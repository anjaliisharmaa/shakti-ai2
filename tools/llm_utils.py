import os
from dotenv import load_dotenv
from litellm import completion

load_dotenv()


def get_llm_response(messages):
    print("[DEBUG] get_llm_response called with messages:", messages)
    try:
        response = completion(
            model="gemini/gemini-1.5-flash",
            api_key=os.getenv("GOOGLE_API_KEY"),
            provider="google",
            messages=messages
        )
        print("[DEBUG] Gemini response:", response)
        return response.choices[0].message.content
    except Exception as e:
        import traceback
        print("[ERROR] Exception in get_llm_response:", str(e))
        traceback.print_exc()
        return f"LLM call failed: {str(e)}"
