import streamlit as st
import fitz  # PyMuPDF
from agent import graph

def extract_text_from_pdf(uploaded_file):
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = "\n".join([page.get_text() for page in doc])
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def main():
    st.set_page_config(page_title="AI Resume Scorer", page_icon="🤖", layout="centered")
    st.title("📄 AI Resume Scorer (Agentic LangGraph + Gemini)")
    st.markdown("Upload your **resume** and paste the **job description** to get an AI-driven fit score, insights, and suggestions.")

    resume_file = st.file_uploader("Upload Resume (PDF format)", type=["pdf"])
    jd_text = st.text_area("Paste Job Description", height=250)

    if st.button("🔍 Evaluate Resume"):
        if not resume_file:
            st.warning("Please upload a resume.")
            return
        if not jd_text.strip():
            st.warning("Please paste the job description.")
            return

        # extract resume text
        with st.spinner("📄 Extracting text from resume..."):
            resume_text = extract_text_from_pdf(resume_file)
            if not resume_text.strip():
                st.error("Couldn't extract text from PDF.")
                return

        # prepare initial state
        state = {
            "resume_text": resume_text,
            "jd_text": jd_text
        }

        # Run LangGraph agent
        with st.spinner("🤖 Evaluating with AI Agent..."):
            result = graph.invoke(state)

        # --- Display Results ---
        def pretty_display(val):
            import json
            if hasattr(val, 'dict'):
                val = val.model_dump()
            if isinstance(val, dict):
                if 'error' in val:
                    return f"**Error:** {val['error']}\n\n<details><summary>Raw Output</summary>\n<pre>{val.get('raw','')}</pre></details>"
                return f"```json\n{json.dumps(val, indent=2, ensure_ascii=False)}\n```"
            if isinstance(val, list):
                return '\n'.join(f"- {x}" for x in val)
            return str(val)

        st.subheader("📄 Parsed Resume Info")
        st.markdown(pretty_display(result.get("parsed_resume", "_No data_")), unsafe_allow_html=True)

        st.subheader("📝 Job Description Analysis")
        st.markdown(pretty_display(result.get("analyzed_jd", "_No data_")), unsafe_allow_html=True)

        st.subheader("📊 Resume-JD Comparison")
        st.markdown(pretty_display(result.get("comparison", "_No data_")), unsafe_allow_html=True)

        st.subheader("✅ AI Fit Score")
        st.markdown(pretty_display(result.get("score", "_No score returned_")), unsafe_allow_html=True)

        st.subheader("💡 Improvement Suggestions")
        st.markdown(pretty_display(result.get("suggestions", "_No suggestions_")), unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Powered by LangGraph 🧠 + Gemini 🌟")

# -------- Run App --------
if __name__ == "__main__":
    main()
