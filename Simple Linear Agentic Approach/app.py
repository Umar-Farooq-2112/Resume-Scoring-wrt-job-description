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
        st.subheader("📄 Parsed Resume Info")
        st.markdown(result.get("parsed_resume", "_No data_"))

        st.subheader("📝 Job Description Analysis")
        st.markdown(result.get("analyzed_jd", "_No data_"))

        st.subheader("📊 Resume-JD Comparison")
        st.markdown(result.get("comparison", "_No data_"))

        st.subheader("✅ AI Fit Score")
        st.markdown(result.get("score", "_No score returned_"))

        st.subheader("💡 Improvement Suggestions")
        st.markdown(result.get("suggestions", "_No suggestions_"))

    st.markdown("---")
    st.caption("Powered by LangGraph 🧠 + Gemini 🌟")

# -------- Run App --------
if __name__ == "__main__":
    main()
