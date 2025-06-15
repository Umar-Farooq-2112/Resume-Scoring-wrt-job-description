from langgraph.graph import StateGraph, START ,END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from typing import TypedDict
import os
from dotenv import load_dotenv
from pydantic import BaseModel



load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.environ.get("API_KEY")



class Project(BaseModel):
    name: str
    description: str

class Experience(BaseModel):
    company: str
    role: str
    duration: str
    description: str

class Education(BaseModel):
    institution: str
    degree: str
    duration: str

class Certification(BaseModel):
    name: str
    issuing_organization: str

class Resume(BaseModel):
    name: str
    skills: list[str]
    education: list[Education]
    experience: list[Experience]
    certifications: list[Certification]
    projects: list[Project]

class JDAnalysis(BaseModel):
    required_skills: list[str]
    experience: str
    industry: str
    key_responsibilities: list[str]

class ComparisonResult(BaseModel):
    skill_match: str
    experience_match: str
    overall_fit: str

class ScoreResult(BaseModel):
    score: int
    reason: str



class AgentState(TypedDict):
    resume_text: str
    jd_text: str
    parsed_resume: Resume
    analyzed_jd: str
    comparison: str
    score: str
    suggestions: str

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)


def clean_json_response(response: str) -> str:
    """Remove leading 'json', code block markers, and whitespace/newlines from LLM output for safe JSON parsing."""
    import re
    response = response.strip()
    # Remove code block markers (``` and variants)
    response = re.sub(r'^```[a-zA-Z]*', '', response)
    response = re.sub(r'```$', '', response)
    # Remove leading 'json' (case-insensitive) and any whitespace/newlines after
    response = re.sub(r'^(json\s*)', '', response, flags=re.IGNORECASE)
    response = response.strip()
    return response

def try_parse_json(cleaned: str):
    import json
    # Try normal parsing first
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # Try replacing single quotes with double quotes
    try:
        fixed = cleaned.replace("'", '"')
        return json.loads(fixed)
    except Exception:
        pass
    # Try removing trailing commas
    import re
    try:
        no_trailing_commas = re.sub(r',\s*([}\]])', r'\1', cleaned)
        return json.loads(no_trailing_commas)
    except Exception:
        pass
    return None

@tool
def parse_resume_tool(resume_text: str) -> Resume | dict:
    """Extract structured details from the resume text such as Name, Skills, Education, Experience, Certifications, and Projects, and return as Resume model."""
    response = llm.invoke(f"""Extract structured details from the resume below and output as JSON in this format:
{{
  'name': str,
  'skills': list[str],
  'education': list[{{'institution': str, 'degree': str, 'duration': str}}],
  'experience': list[{{'company': str, 'role': str, 'duration': str, 'description': str}}],
  'certifications': list[{{'name': str, 'issuing_organization': str}}],
  'projects': list[{{'name': str, 'description': str}}]
}}
Resume:
{resume_text}
""").content
    import json
    try:
        cleaned = clean_json_response(response)
        data = json.loads(cleaned.replace("'", '"'))
        return Resume(**data)
    except Exception as e:
        return {"error": f"Failed to parse Resume: {e}", "raw": response}

@tool
def analyze_jd_tool(jd_text: str) -> JDAnalysis | dict:
    """Analyze the job description and extract required skills, experience, industry, and key responsibilities as JDAnalysis model."""
    response = llm.invoke(f"""Analyze this job description and output as JSON in this format:
{{
  'required_skills': list[str],
  'experience': str,
  'industry': str,
  'key_responsibilities': list[str]
}}
JD:
{jd_text}
""").content
    import json
    try:
        cleaned = clean_json_response(response)
        data = json.loads(cleaned.replace("'", '"'))
        return JDAnalysis(**data)
    except Exception as e:
        return {"error": f"Failed to parse JDAnalysis: {e}", "raw": response}

@tool
def compare_tool(parsed_resume: Resume | dict, analyzed_jd: JDAnalysis | dict) -> ComparisonResult | dict:
    """Compare the parsed resume with the analyzed job description and return a ComparisonResult model."""
    response = llm.invoke(f"""Compare this resume with the job description and output as JSON in this format:
{{
  'skill_match': str,
  'experience_match': str,
  'overall_fit': str
}}
Resume:
{parsed_resume}
Job Description:
{analyzed_jd}
""").content
    cleaned = clean_json_response(response)
    data = try_parse_json(cleaned)
    if data is not None:
        try:
            return ComparisonResult(**data)
        except Exception as e:
            return {"error": f"Failed to parse ComparisonResult: {e}", "raw": response}
    return {"error": "Failed to parse ComparisonResult: Could not decode JSON", "raw": response}

@tool
def score_tool(comparison: ComparisonResult | dict) -> ScoreResult | dict:
    """Rate the candidate from 0 to 10 based on the comparison and return as ScoreResult model."""
    response = llm.invoke(f"""Rate this candidate from 0 to 10 based on the following comparison and output as JSON:
{{
  'score': int,
  'reason': str
}}
Comparison:
{comparison}
""").content
    cleaned = clean_json_response(response)
    data = try_parse_json(cleaned)
    if data is not None:
        try:
            return ScoreResult(**data)
        except Exception as e:
            return {"error": f"Failed to parse ScoreResult: {e}", "raw": response}
    return {"error": "Failed to parse ScoreResult: Could not decode JSON", "raw": response}

@tool
def suggestions_tool(resume_text: str, jd_text: str) -> list[str] | dict:
    """Suggest 3 ways to improve the resume to better match the job description, output as a list of suggestions."""
    response = llm.invoke(f"""Suggest 3 ways to improve the resume to better match this job. Output as JSON list of suggestions:
['suggestion1', 'suggestion2', 'suggestion3']
Resume:
{resume_text}
Job Description:
{jd_text}
""").content
    cleaned = clean_json_response(response)
    data = try_parse_json(cleaned)
    if data is not None:
        if isinstance(data, list):
            return data
        return {"error": "Suggestions not a list", "raw": response}
    return {"error": "Failed to parse suggestions: Could not decode JSON", "raw": response}

def parse_resume_node(state: AgentState):
    parsed = parse_resume_tool.invoke({"resume_text": state['resume_text']})
    return {"parsed_resume": parsed}

def analyze_jd_node(state: AgentState):
    analyzed = analyze_jd_tool.invoke({"jd_text": state['jd_text']})
    return {"analyzed_jd": analyzed}

def compare_node(state: AgentState):
    comparison = compare_tool.invoke({
        "parsed_resume": state['parsed_resume'],
        "analyzed_jd": state['analyzed_jd']
    })
    return {"comparison": comparison}

def score_node(state: AgentState):
    score = score_tool.invoke({"comparison": state['comparison']})
    return {"score": score}

def suggestions_node(state: AgentState):
    suggestions = suggestions_tool.invoke({
        "resume_text": state['resume_text'],
        "jd_text": state['jd_text']
    })
    return {"suggestions": suggestions}

builder = StateGraph(AgentState)

builder.add_node("parse_resume", parse_resume_node)
builder.add_node("analyze_jd", analyze_jd_node)
builder.add_node("compare", compare_node)
builder.add_node("score_node", score_node) 
builder.add_node("suggestions_node", suggestions_node)

builder.add_edge(START, "parse_resume")
# builder.set_entry_point("parse_resume")
builder.add_edge("parse_resume", "analyze_jd")
builder.add_edge("analyze_jd", "compare")
builder.add_edge("compare", "score_node") 
builder.add_edge("score_node", "suggestions_node") 
builder.add_edge("suggestions_node", END)

graph = builder.compile()