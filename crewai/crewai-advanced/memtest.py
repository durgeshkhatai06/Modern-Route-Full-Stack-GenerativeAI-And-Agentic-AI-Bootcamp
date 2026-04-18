import os  # Imports the os module so environment variables can be read and set.
from pathlib import Path  # Imports Path so filesystem paths can be built cleanly.

# Store Mem0's local files inside the repo instead of the user's home directory.
os.environ.setdefault("MEM0_DIR", str(Path(__file__).resolve().parents[2] / ".mem0"))  # Sets MEM0_DIR only if it is not already defined.

from mem0 import MemoryClient  # Imports MemoryClient so the script can add and search memories.
from dotenv import load_dotenv  # Imports dotenv support so API keys can be loaded from .env files.

load_dotenv()  # Loads environment variables from a default .env file if one exists.
load_dotenv(Path(__file__).resolve().parent / ".env")  # Loads environment variables from crewai/crewai-advanced/.env.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")  # Loads environment variables from crewai/.env.

mem0_api_key = os.environ.get("MEM0_API_KEY")  # Reads the Mem0 API key from the environment.
if not mem0_api_key:  # Checks whether the required API key is missing.
    raise ValueError(  # Raises a clear error if the key was not found.
        "MEM0_API_KEY is not set. Add it to your environment or create a .env file in "
        "`crewai/crewai-advanced/.env` or `crewai/.env`."
    )  # Ends the error message block.

client = MemoryClient(api_key=mem0_api_key)  # Creates a Mem0 client using the loaded API key.

messages = [  # Creates the conversation messages that will be stored as memory.
    {"role": "user", "content": "Hi, I'm Alex. I'm a vegetarian and I'm allergic to nuts."},  # Adds the user's preference and allergy information.
    {"role": "assistant", "content": "Hello Alex! I've noted that you're a vegetarian and have a nut allergy. I'll keep this in mind for any food-related recommendations or discussions."},  # Adds the assistant reply that confirms the memory.
]  # Finishes the list of memory messages.

client.add(messages, user_id="alex")  # Stores the conversation messages under the user id alex.

query = "What can I cook for dinner tonight?"  # Defines the question that will be used to search the stored memory.
result = client.search(query, filters={"user_id": "alex"})  # Searches memories for alex using the current Mem0 filter format.
print(result)  # Prints the search result so it can be seen in the terminal.
