"""
First test (ChatNVIDIA version): send ONE question to ONE model via
NVIDIA AI Endpoints, print the raw SQL. No loop, no scoring, no cleaning.

Setup:
  pip install langchain-nvidia-ai-endpoints python-dotenv
  Configure the first model's API-key variable from model_nvidia.py in .env.
"""

import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from model_nvidia import MODELS

load_dotenv()

_, MODEL_ID, API_KEY_ENV = MODELS[0]
API_KEY = os.getenv(API_KEY_ENV)
if not API_KEY:
    raise SystemExit(f"Missing {API_KEY_ENV} - add it to your .env file")

# 1. Read the schema from disk
with open("schema.sql", encoding="utf-8") as f:
    schema = f.read().strip()

# 2. The one question we're testing
question = "Who scored the most runs in 2024?"

# 3. Build the prompt (schema + question)
system_msg = (
    "You are a text-to-SQL generator. "
    "Given a database schema and a question, return a single SQL query that answers it. "
    "Use SQLite syntax. Return only the SQL query."
)
user_msg = f"Schema:\n{schema}\n\nQuestion: {question}\n\nSQL:"

# 4. Connect via NVIDIA AI Endpoints
llm = ChatNVIDIA(
    model=MODEL_ID,
    api_key=API_KEY,
    temperature=0,
    max_completion_tokens=300,
)

# 5. Send it and print what comes back
messages = [
    SystemMessage(content=system_msg),
    HumanMessage(content=user_msg),
]
response = llm.invoke(messages)

raw_sql = response.content

print("=" * 60)
print("QUESTION:", question)
print("=" * 60)
print("RAW MODEL OUTPUT:")
print(raw_sql)
print("=" * 60)