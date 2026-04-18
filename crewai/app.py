import os  # Imports Python's built-in os module so we can read and set environment variables.
import re  # Imports regular expressions so we can read retry timing from rate-limit messages.
import time  # Imports time so we can pause briefly before retrying a rate-limited request.
from pathlib import Path  # Imports Path so we can build filesystem paths in a clean cross-platform way.

# Keep CrewAI's local storage inside the repo so the script doesn't depend on writable AppData permissions.
os.environ.setdefault("LOCALAPPDATA", str(Path(__file__).resolve().parents[1] / ".localappdata"))  # Sets a local AppData override only if it is not already defined.

from crewai import Crew, Agent, Task, LLM  # Imports the main CrewAI classes used to build agents, tasks, crews, and the language model.
from crewai_tools import SerperDevTool  # Imports the Serper search tool so agents can search the web.
from dotenv import load_dotenv  # Imports dotenv support so values from a .env file can be loaded automatically.

load_dotenv()  # Loads environment variables from a .env file into this process if the file exists.

# Set your Serper key here if you don't want to use a .env file.
os.environ.setdefault("SERPER_API_KEY", "your_serper_api_key_here")  # Sets a fallback Serper API key when one is not already present.

topic = "AI in Healthcare"  # Defines the topic that the research and writing workflow will focus on.

llm = LLM(  # Creates the language model configuration that both agents will share.
    model=os.environ.get("CREWAI_MODEL", "llama-3.3-70b-versatile"),  # Uses CREWAI_MODEL if provided, otherwise defaults to a Groq model name.
    api_key=os.environ.get("GROQ_API_KEY"),  # Reads the Groq API key from the environment for authentication.
    base_url=os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),  # Uses Groq's OpenAI-compatible endpoint unless a custom one is provided.
    temperature=float(os.environ.get("CREWAI_TEMPERATURE", "0.1")),  # Uses a very low temperature to keep outputs focused and reduce unnecessary token usage.
    max_tokens=int(os.environ.get("CREWAI_MAX_TOKENS", "500")),  # Caps each response more aggressively so the crew stays under Groq's token-per-minute limit more easily.
)  # Finishes the shared LLM configuration.

# Tool
search_tool = SerperDevTool(n=1)  # Creates a search tool that returns fewer search results to reduce context size and token usage.

# Agent 1 - Research Agent
senior_research_analyst = Agent(  # Creates the first agent responsible for research and fact gathering.
    role="Senior Research Analyst",  # Gives the agent its job title and identity.
    goal=f"Research, analyze, and synthesize comprehensive information on {topic} from reliable web sources",  # Defines the main objective for this agent.
    backstory="You're an expert research analyst with advanced web research skills. "  # Starts the backstory that shapes how the agent behaves.
    "You excel at finding, analyzing, and synthesizing information from "  # Explains that the agent is strong at gathering and combining information.
    "across the internet using search tools. You're skilled at "  # Describes the agent's web-search ability.
    "distinguishing reliable sources from unreliable ones, "  # Tells the agent to judge source quality carefully.
    "fact-checking, cross-referencing information, and "  # Instructs the agent to verify claims across sources.
    "identifying key patterns and insights. You provide "  # Tells the agent to look for trends and useful findings.
    "well-organized research briefs with proper citations "  # Requires structured output with citations.
    "and source verification. Your analysis includes both "  # Clarifies that the report should be factual and validated.
    "raw data and interpreted insights, making complex "  # Encourages both evidence and explanation.
    "information accessible and actionable.",  # Finishes the backstory sentence.
    verbose=True,  # Enables detailed logging so agent steps are shown in the console.
    allow_delegation=False,  # Prevents this agent from creating or handing work off to other agents.
    tools=[search_tool],  # Gives the research agent access to the Serper search tool.
    llm=llm,  # Assigns the shared language model configuration to this agent.
)  # Finishes the research agent definition.

# Agent 2 - Content Creator
content_writer = Agent(  # Creates the second agent responsible for turning research into a blog post.
    role="Content Writer",  # Gives the second agent its writing-focused role.
    goal="Transform research findings into engaging blog posts while maintaining accuracy",  # Defines the writing agent's objective.
    backstory="You're a skilled content writer specialized in creating "  # Starts the writer agent's backstory.
    "engaging, accessible content from technical research. "  # Explains that the writer turns technical content into readable content.
    "You work closely with the Senior Research Analyst and excel at maintaining the perfect "  # Tells the writer to collaborate with the research agent's output.
    "balance between informative and entertaining writing, "  # Encourages a balance of clarity and engagement.
    "while ensuring all facts and citations from the research "  # Instructs the agent to preserve factual correctness.
    "are properly incorporated. You have a talent for making "  # Describes how the agent should write.
    "complex topics approachable without oversimplifying them.",  # Finishes the backstory with the expected writing style.
    verbose=True,  # Enables detailed logs for this agent too.
    allow_delegation=False,  # Prevents this agent from delegating tasks elsewhere.
    llm=llm,  # Assigns the same shared language model configuration to the writer.
)  # Finishes the content writer definition.

# TASK 1
research_tasks = Task(  # Creates the first task that asks the research agent to investigate the topic.
    description=(  # Starts the detailed instructions for the research task.
        """
            1. Conduct comprehensive research on {topic} including:
                - 2 recent developments or news items
                - 2 key industry trends or innovations
                - 1 expert opinion or analysis
                - 1 relevant fact or statistic
            2. Evaluate source credibility and fact-check all information
            3. Organize findings into a concise structured research brief
            4. Include all relevant citations and sources
            5. Keep the total response under 250 words
        """
    ),  # Ends the multi-line task description.
    expected_output="""A concise research report containing:
            - Executive summary of key findings
            - Comprehensive analysis of current trends and developments
            - List of verified facts and statistics
            - All citations and links to original sources
            - Clear categorization of main themes and patterns
            Please format with clear sections and bullet points for easy reference and keep it under 250 words.""",  # Defines what a successful research output should look like.
    agent=senior_research_analyst,  # Assigns this task to the research agent.
)  # Finishes the research task definition.

# Content Writer Task
# Task 2 Content Writing
writing_task = Task(  # Creates the second task that asks the writer to turn research into a blog post.
    description=(  # Starts the detailed instructions for the writing task.
        """
            Using the research brief provided, create an engaging blog post that:
            1. Transforms technical information into accessible content
            2. Maintains all factual accuracy and citations from the research
            3. Includes:
                - Attention-grabbing introduction
                - Well-structured body sections with clear headings
                - Compelling conclusion
            4. Preserves all source citations in [Source: URL] format
            5. Includes a References section at the end
            6. Keep the total blog post under 400 words
        """
    ),  # Ends the multi-line writing task description.
    expected_output="""A polished blog post in markdown format that:
            - Engages readers while maintaining accuracy
            - Contains properly structured sections
            - Includes Inline citations hyperlinked to the original source url
            - Presents information in an accessible yet informative way
            - Follows proper markdown formatting, use H1 for the title and H3 for the sub-sections
            - Stays under 400 words""",  # Describes the required format and quality of the final article.
    agent=content_writer,  # Assigns this task to the content writer agent.
)  # Finishes the writing task definition.

crew = Crew(  # Creates the CrewAI workflow that combines both agents and both tasks.
    agents=[senior_research_analyst, content_writer],  # Registers the two agents that will participate in the workflow.
    tasks=[research_tasks, writing_task],  # Registers the tasks in the order they should be completed.
    verbose=True,  # Enables detailed execution logs for the whole crew.
)  # Finishes the crew definition.


def kickoff_with_retry(active_crew: Crew, inputs: dict, retries: int = 3):  # Defines a helper that retries a few times after a rate-limit wait.
    for attempt in range(retries + 1):  # Loops through the initial attempt plus any allowed retries.
        try:  # Starts a protected block so CrewAI errors can be handled safely.
            return active_crew.kickoff(inputs=inputs)  # Runs the crew and returns the result on success.
        except Exception as exc:  # Handles any exception raised while the crew is running.
            error_text = str(exc)  # Converts the exception to text so we can inspect the message.
            wait_match = re.search(r"try again in ([0-9.]+)s", error_text, re.IGNORECASE)  # Looks for Groq's suggested retry delay.
            if "rate limit" in error_text.lower() and wait_match and attempt < retries:  # Checks whether this is a retryable rate-limit error.
                wait_seconds = float(wait_match.group(1)) + 1.0  # Adds a small buffer to the suggested wait time.
                print(f"Rate limit hit. Waiting {wait_seconds:.1f} seconds before retrying...")  # Tells the user what the script is doing.
                time.sleep(wait_seconds)  # Pauses before retrying so the next request is more likely to succeed.
                continue  # Tries the request again after the wait.
            raise  # Re-raises non-retryable errors or failures after the final attempt.


result = kickoff_with_retry(crew, {"topic": topic})  # Starts the crew with a one-time retry for rate-limit errors.

print(result)  # Prints the final combined result to the terminal.
