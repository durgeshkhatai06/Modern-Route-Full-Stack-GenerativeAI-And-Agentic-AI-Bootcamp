import os
from pathlib import Path

os.environ.setdefault("MEM0_DIR", str(Path(__file__).resolve().parents[2] / ".mem0"))

from mem0 import MemoryClient
from dotenv import load_dotenv

load_dotenv()
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

mem0_api_key = os.environ.get("MEM0_API_KEY")
if not mem0_api_key:
    raise ValueError(
        "MEM0_API_KEY is not set. Add it to your environment or create a .env file in "
        "`crewai/crewai-advanced/.env` or `crewai/.env`."
    )

client = MemoryClient(api_key=mem0_api_key)

messages = [
    {"role": "user", "content": "Hi, I'm Alex. I'm a vegetarian and I'm allergic to nuts."},
    {"role": "assistant", "content": "Hello Alex! I've noted that you're a vegetarian and have a nut allergy. I'll keep this in mind for any food-related recommendations or discussions."}
]

client.add(messages, user_id="alex")

query = "What can I cook for dinner tonight?"
result = client.search(query, filters={"user_id": "alex"})
print(result)
