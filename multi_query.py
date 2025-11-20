# multi_query.py
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DEFAULT_MULTI_MODEL = "gpt-4o-mini"

def generate_multi_queries(query: str, n=3):
    """
    return: [q1, q2, q3]
    """
    prompt = [
        {
            "role": "system",
            "content": (
                "You generate multiple semantically different variations of the user's question. "
                "Keep meaning but vary keywords, structure, and scope. "
                "Output exactly N variations, each on a new line."
            )
        },
        {
            "role": "user",
            "content": f"Original query: {query}\nN={n}"
        }
    ]

    resp = client.chat.completions.create(
        model=DEFAULT_MULTI_MODEL,
        messages=prompt,
        temperature=0.3
    )

    raw = resp.choices[0].message.content.strip()
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    return lines[:n]
