from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv
import os
load_dotenv()

llm = LLM(model="gpt-4o", api_key=os.environ["OPENAI_API_KEY"])

# Create a multimodal agent for detailed analysis
