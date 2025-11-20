# router.py
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DEFAULT_ROUTER_MODEL = "gpt-4o-mini"

def classify_query(query: str) -> str:
    """
    Return: "SCIENCE" or "GENERAL"
    """
    prompt = [
        {
            "role": "system",
            "content":
            "You are a classifier. Your job is to classify a user query into one of two labels: "
            "'SCIENCE' (if the question asks for scientific knowledge, facts, explanation, biology, physics, astronomy, chemistry, math) "
            "or 'GENERAL' (if it is emotional talk, social conversation, greetings, casual messages, feelings, humor, etc.). "
            "Respond with only 'SCIENCE' or 'GENERAL'."
        },
        {
            "role": "user",
            "content": query
        }
    ]

    resp = client.chat.completions.create(
        model=DEFAULT_ROUTER_MODEL,
        messages=prompt,
        temperature=0.0
    )
    result = resp.choices[0].message.content.strip().upper()

    # fallback
    if result not in ["SCIENCE", "GENERAL"]:
        return "SCIENCE"

    return result
