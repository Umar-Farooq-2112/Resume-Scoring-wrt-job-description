# Agentic AI Resume Scorer

This project is an AI-powered resume scoring and analysis tool built with LangGraph, LangChain, and Google Gemini. It allows users to upload their resume (PDF) and a job description, then provides:
- Structured resume parsing
- Job description analysis
- Resume-to-JD comparison
- AI fit score
- Actionable improvement suggestions

## Features
- **Streamlit Web App**: User-friendly interface for uploading resumes and pasting job descriptions.
- **Agentic Workflow**: Uses a multi-step agentic pipeline for robust, explainable results.
- **Google Gemini LLM**: All analysis and scoring is powered by Gemini via LangChain.
- **Pydantic Validation**: Ensures all outputs are structured and validated.

---

## Project Structure

```
Agentic AI based Resume Scorer/
├── Simple Linear Agentic Approach/
│   ├── app.py              # Streamlit app (original version)
│   └── agent.py            # Agent logic (original version)
├── Defined Linear Agentic/
│   ├── app.py              # Streamlit app (Pydantic/robust version)
│   └── agent.py            # Agent logic (Pydantic/robust version)
├── readme.md               # This file
├── .gitignore              # git ignore files
```

---

## Versions Explained

### 1. Original Version
- **Files**: `Simple Linear Agentic Approach\app.py`, `Simple Linear Agentic Approach\agent.py`
- **Description**: Implements the agentic workflow using LangGraph and Gemini. Outputs are parsed as plain strings. Simpler, but less robust to LLM formatting issues.
- **Use Case**: Good for quick prototyping or when LLM output is always well-formed.

### 2. Defined Linear Agentic Version (Recommended)
- **Files**: `Defined Linear Agentic/app.py`, `Defined Linear Agentic/agent.py`
- **Description**: Adds strict Pydantic models for all agent outputs. Robustly parses LLM output, handling code block markers, single quotes, and other common issues. All outputs are validated and structured.
- **Use Case**: Recommended for production or when you want reliable, schema-validated results.

---

## How to Run

1. **Install dependencies**:
   ```bash
   cd '.\Simple Linear Agentic Approach\'
   pip install -r requirements.txt
   cd '.\Defined Linear Agentic\'
   pip install -r requirements.txt
   ```
2. **Set your Google Gemini API key**:
   - Create two `.env` files in "\Simple Linear Agentic Approach" and \Defined Linear Agentic with:
     ```
     API_KEY=your_google_gemini_api_key
     ```
3. **Run the app**:
   - For the original version:
     ```bash
     cd '.\Simple Linear Agentic Approach\'
     streamlit run app.py
     ```
   - For the robust version:
     ```bash
     cd '.\Defined Linear Agentic\'
     streamlit run app.py
     ```

---

## Credits
- Built with [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain), and [Google Gemini](https://ai.google/discover/gemini/).
- Resume parsing and scoring logic by Umar Farooq.

---
