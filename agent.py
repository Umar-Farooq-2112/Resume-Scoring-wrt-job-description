from langgraph.graph import StateGraph, START ,END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from typing import TypedDict, Optional
import fitz  # PyMuPDF
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.environ.get("API_KEY")
# ---- LangGraph State ----
class AgentState(TypedDict):
    resume_text: str
    jd_text: str
    parsed_resume: str
    analyzed_jd: str
    comparison: str
    score: str
    suggestions: str

# ---- Gemini Model ----
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

# ---- Tools ----

# @tool
def parse_resume_tool(resume_text: str) -> str:
    return llm.invoke(f"""Extract structured details from the resume:
- Name
- Skills
- Education
- Experience
- Certifications
- Projects
Resume:
{resume_text}""").content

# @tool
def analyze_jd_tool(jd_text: str) -> str:
    return llm.invoke(f"""Analyze this job description and extract:
- Required skills
- Experience
- Industry
- Key responsibilities
JD:
{jd_text}""").content

# @tool
def compare_tool(parsed_resume: str, analyzed_jd: str) -> str:
    return llm.invoke(f"""Compare this resume with the job description.
Resume:
{parsed_resume}
Job Description:
{analyzed_jd}

Return a comparison in terms of skill match, experience match, and overall fit.
""").content

# @tool
def score_tool(comparison: str) -> str:
    return llm.invoke(f"""Rate this candidate from 0 to 10 based on the following comparison:
{comparison}

Format:
Score: X/10
Reason: ...
""").content

# @tool
def suggestions_tool(resume_text: str, jd_text: str) -> str:
    return llm.invoke(f"""Suggest 3 ways to improve the resume to better match this job:
Resume:
{resume_text}

Job Description:
{jd_text}
""").content

# ---- LangGraph Nodes ----
def parse_resume_node(state: AgentState):
    parsed = parse_resume_tool(state['resume_text'])
    return {"parsed_resume": parsed}

def analyze_jd_node(state: AgentState):
    analyzed = analyze_jd_tool(state['jd_text'])
    return {"analyzed_jd": analyzed}

def compare_node(state: AgentState):
    comparison = compare_tool(
        parsed_resume=state['parsed_resume'],
        analyzed_jd=state['analyzed_jd']
    )
    return {"comparison": comparison}

def score_node(state: AgentState):
    score = score_tool(state['comparison'])
    return {"score": score}

def suggestions_node(state: AgentState):
    suggestions = suggestions_tool(
        resume_text=state['resume_text'],
        jd_text=state['jd_text']
    )
    return {"suggestions": suggestions}

# ---- LangGraph DAG ----
builder = StateGraph(AgentState)

builder.add_node("parse_resume", parse_resume_node)
builder.add_node("analyze_jd", analyze_jd_node)
builder.add_node("compare", compare_node)
builder.add_node("score_node", score_node)  # Renamed from 'score'
builder.add_node("suggestions_node", suggestions_node)

builder.add_edge(START, "parse_resume")
# builder.set_entry_point("parse_resume")
builder.add_edge("parse_resume", "analyze_jd")
builder.add_edge("analyze_jd", "compare")
builder.add_edge("compare", "score_node")  # Updated edge
builder.add_edge("score_node", "suggestions_node")  # Updated edge
builder.add_edge("suggestions_node", END)

graph = builder.compile()