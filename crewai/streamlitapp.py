import os
import re
import time
from pathlib import Path

import streamlit as st
from crewai import Agent, Crew, LLM, Task
from crewai_tools import SerperDevTool
from dotenv import load_dotenv


os.environ.setdefault(
    "LOCALAPPDATA",
    str(Path(__file__).resolve().parents[1] / ".localappdata"),
)

load_dotenv()


def build_crew(topic: str, groq_api_key: str, serper_api_key: str) -> Crew:
    os.environ["GROQ_API_KEY"] = groq_api_key
    os.environ["SERPER_API_KEY"] = serper_api_key

    llm = LLM(
        model=os.environ.get("CREWAI_MODEL", "llama-3.3-70b-versatile"),
        api_key=os.environ["GROQ_API_KEY"],
        base_url=os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        temperature=float(os.environ.get("CREWAI_TEMPERATURE", "0.1")),
        max_tokens=int(os.environ.get("CREWAI_MAX_TOKENS", "500")),
    )

    search_tool = SerperDevTool(n=1)

    senior_research_analyst = Agent(
        role="Senior Research Analyst",
        goal=f"Research, analyze, and synthesize comprehensive information on {topic} from reliable web sources",
        backstory=(
            "You're an expert research analyst with advanced web research skills. "
            "You excel at finding, analyzing, and synthesizing information from "
            "across the internet using search tools. You're skilled at "
            "distinguishing reliable sources from unreliable ones, "
            "fact-checking, cross-referencing information, and "
            "identifying key patterns and insights. You provide "
            "well-organized research briefs with proper citations "
            "and source verification. Your analysis includes both "
            "raw data and interpreted insights, making complex "
            "information accessible and actionable."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
    )

    content_writer = Agent(
        role="Content Writer",
        goal="Transform research findings into engaging blog posts while maintaining accuracy",
        backstory=(
            "You're a skilled content writer specialized in creating "
            "engaging, accessible content from technical research. "
            "You work closely with the Senior Research Analyst and excel at maintaining the perfect "
            "balance between informative and entertaining writing, "
            "while ensuring all facts and citations from the research "
            "are properly incorporated. You have a talent for making "
            "complex topics approachable without oversimplifying them."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    research_task = Task(
        description="""
            1. Conduct comprehensive research on {topic} including:
                - 2 recent developments or news items
                - 2 key industry trends or innovations
                - 1 expert opinion or analysis
                - 1 relevant fact or statistic
            2. Evaluate source credibility and fact-check all information
            3. Organize findings into a concise structured research brief
            4. Include all relevant citations and sources
            5. Keep the total response under 250 words
        """,
        expected_output="""A concise research report containing:
            - Executive summary of key findings
            - Comprehensive analysis of current trends and developments
            - List of verified facts and statistics
            - All citations and links to original sources
            - Clear categorization of main themes and patterns
            Please format with clear sections and bullet points for easy reference and keep it under 250 words.""",
        agent=senior_research_analyst,
    )

    writing_task = Task(
        description="""
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
        """,
        expected_output="""A polished blog post in markdown format that:
            - Engages readers while maintaining accuracy
            - Contains properly structured sections
            - Includes Inline citations hyperlinked to the original source url
            - Presents information in an accessible yet informative way
            - Follows proper markdown formatting, use H1 for the title and H3 for the sub-sections
            - Stays under 400 words""",
        agent=content_writer,
    )

    return Crew(
        agents=[senior_research_analyst, content_writer],
        tasks=[research_task, writing_task],
        verbose=True,
    )


def kickoff_with_retry(active_crew: Crew, inputs: dict, retries: int = 3):
    for attempt in range(retries + 1):
        try:
            return active_crew.kickoff(inputs=inputs)
        except Exception as exc:
            error_text = str(exc)
            wait_match = re.search(r"try again in ([0-9.]+)s", error_text, re.IGNORECASE)
            if "rate limit" in error_text.lower() and wait_match and attempt < retries:
                wait_seconds = float(wait_match.group(1)) + 1.0
                st.warning(
                    f"Groq rate limit hit. Waiting {wait_seconds:.1f} seconds before retry {attempt + 1} of {retries}."
                )
                time.sleep(wait_seconds)
                continue
            raise


st.set_page_config(page_title="CrewAI Blog Generator", page_icon="AI", layout="wide")
st.title("CrewAI Blog Generator")
st.write("Generate a researched blog post from a topic using CrewAI, Groq, and Serper.")

with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input(
        "GROQ_API_KEY",
        value=os.environ.get("GROQ_API_KEY", ""),
        type="password",
    )
    serper_api_key = st.text_input(
        "SERPER_API_KEY",
        value=os.environ.get("SERPER_API_KEY", ""),
        type="password",
    )
    model_name = st.text_input(
        "Model",
        value=os.environ.get("CREWAI_MODEL", "llama-3.3-70b-versatile"),
    )
    max_tokens = st.number_input(
        "Max tokens per model call",
        min_value=200,
        max_value=4000,
        value=int(os.environ.get("CREWAI_MAX_TOKENS", "500")),
        step=50,
    )
    os.environ["CREWAI_MODEL"] = model_name
    os.environ["CREWAI_MAX_TOKENS"] = str(max_tokens)

topic = st.text_input("Topic", value="AI in Healthcare")

st.caption("Recommended for Groq on-demand tier: keep max tokens around 400-600 and use short topics.")

if st.button("Generate Blog Post", type="primary"):
    if not groq_api_key:
        st.error("Please provide a GROQ_API_KEY.")
    elif not serper_api_key:
        st.error("Please provide a SERPER_API_KEY.")
    else:
        try:
            with st.spinner("CrewAI is researching and writing..."):
                crew = build_crew(topic, groq_api_key, serper_api_key)
                result = kickoff_with_retry(crew, {"topic": topic})

            st.success("Blog post generated successfully.")
            st.subheader("Result")
            st.markdown(str(result))
        except Exception as exc:
            st.error(f"Failed to run CrewAI: {exc}")
