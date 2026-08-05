import streamlit as st
import json
import os
import re
from openai import OpenAI
import docx
from pypdf import PdfReader
import pandas as pd
from PIL import Image

# ==============================================================================
# 1. HELPER FUNCTIONS & REGEX LATEX FIX
# ==============================================================================
DATA_FILE = "study_data.json"

def fix_latex_formatting(text: str) -> str:
    """
    Cleans up raw LaTeX equations sent by LLMs that use [ ... ] or ( ... ) brackets
    and converts them to standard KaTeX delimiters ($$ and $) for Streamlit rendering.
    """
    if not text:
        return ""
    
    # 1. Convert block math [ \formula ] or \[ \formula \] into $$ \formula $$
    text = re.sub(r'(?:\\\[|\[)\s*(\\.*?)\s*(?:\\\]|\])', r'$$\1$$', text, flags=re.DOTALL)
    
    # 2. Convert inline math ( \symbol ) or \( \symbol \) into $ \symbol $
    text = re.sub(r'(?:\\\Custom\(|\()\s*(\\.*?)\s*(?:\\\Custom\Component|\))', r'$\1$', text)
    
    # 3. Handle edge cases where math functions are missing delimiters entirely inside standalone lines
    text = re.sub(r'(?m)^\\(frac|sqrt|left|mathrm|mathbf|boldsymbol)\{.*\}$', r'$$\g<0>$$', text)
    
    return text

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "threads": {"Default Session": []},
        "evaluations": {"mcq": [], "essay": []}
    }

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

def parse_file(uploaded_file):
    file_type = uploaded_file.name.split('.')[-1].lower()
    text = ""
    try:
        if file_type == 'pdf':
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                text += page.extract_text() or ""
        elif file_type == 'docx':
            doc = docx.Document(uploaded_file)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif file_type in ['csv', 'xlsx']:
            if file_type == 'csv':
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            text = df.to_string()
        elif file_type in ['png', 'jpg', 'jpeg']:
            text = f"[Image uploaded: {uploaded_file.name}]"
    except Exception as e:
        text = f"[Error reading file: {str(e)}]"
    return text

# ==============================================================================
# 2. APP INITIALIZATION & STATE SETUP
# ==============================================================================
st.set_page_config(page_title="Agentic Study Platform", layout="wide", page_icon="📚")

if "db" not in st.session_state:
    st.session_state.db = load_data()

if "current_thread" not in st.session_state:
    st.session_state.current_thread = list(st.session_state.db["threads"].keys())[0]

# System prompt forcing LaTeX compliance
SYSTEM_PROMPT = """You are an expert AI Study Assistant.
When outputting mathematical expressions, physics derivations, chemical formulas, or statistical notation:
1. ALWAYS format inline math using single dollar signs: $...$
2. ALWAYS format block equations using double dollar signs: $$...$$
3. NEVER output raw brackets like [ \formula ] or ( \symbol ) for LaTeX rendering.
Provide clear, structured, and complete academic explanations.
"""

# OpenRouter Available Free Models
AVAILABLE_MODELS = {
    "DeepSeek R1 (Free)": "deepseek/deepseek-r1:free",
    "Meta Llama 3.3 70B (Free)": "meta-llama/llama-3.3-70b-instruct:free",
    "Mistral Small 24B (Free)": "mistralai/mistral-small-24b-instruct-2501:free",
    "Google Gemini 2.0 Flash (Free)": "google/gemini-2.0-flash-exp:free"
}

# ==============================================================================
# 3. SIDEBAR NAVIGATION & CONFIG
# ==============================================================================
with st.sidebar:
    st.title("📚 Study Agent")
    
    api_key = st.text_input("OpenRouter API Key", type="password", help="Enter your OpenRouter key here")
    selected_model_label = st.selectbox("Select LLM Model", list(AVAILABLE_MODELS.keys()))
    selected_model = AVAILABLE_MODELS[selected_model_label]
    
    st.markdown("---")
    st.subheader("💬 Chat Threads")
    
    # Create new thread
    new_thread_name = st.text_input("New Thread Name")
    if st.button("➕ Create Thread") and new_thread_name:
        if new_thread_name not in st.session_state.db["threads"]:
            st.session_state.db["threads"][new_thread_name] = []
            st.session_state.current_thread = new_thread_name
            save_data(st.session_state.db)
            st.rerun()

    # Select existing thread
    threads_list = list(st.session_state.db["threads"].keys())
    st.session_state.current_thread = st.selectbox(
        "Select Active Thread",
        threads_list,
        index=threads_list.index(st.session_state.current_thread) if st.session_state.current_thread in threads_list else 0
    )
    
    st.markdown("---")
    st.subheader("📁 Upload Reference Files")
    uploaded_files = st.file_uploader("Attach study materials (PDF, DOCX, CSV, XLSX)", accept_multiple_files=True)

# ==============================================================================
# 4. MAIN WORKSPACE SETUP (TABS)
# ==============================================================================
tab_chat, tab_mcq, tab_essay = st.tabs(["💬 Chat & Workspace", "📝 MCQ Evaluation", "📄 Essay Evaluation"])

# Lazy client instantiation helper
def get_openrouter_client():
    if not api_key:
        st.warning("Please enter your OpenRouter API Key in the sidebar to generate responses.")
        return None
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

# ------------------------------------------------------------------------------
# TAB 1: CHAT WORKSPACE
# ------------------------------------------------------------------------------
with tab_chat:
    st.header(f"Session: {st.session_state.current_thread}")
    
    # Process attached files into context string
    file_context = ""
    if uploaded_files:
        for f in uploaded_files:
            file_context += f"\n\n--- Content from {f.name} ---\n" + parse_file(f)

    # Display thread message history
    current_messages = st.session_state.db["threads"][st.session_state.current_thread]
    for msg in current_messages:
        with st.chat_message(msg["role"]):
            # Pass saved responses through LaTeX regex renderer
            st.markdown(fix_latex_formatting(msg["content"]))

    # Input field for user query
    if user_query := st.chat_input("Ask a question, request a formula derivation, or submit a problem..."):
        # Append user message
        st.session_state.db["threads"][st.session_state.current_thread].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        client = get_openrouter_client()
        if client:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                # Build context payload
                api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                for m in st.session_state.db["threads"][st.session_state.current_thread]:
                    api_messages.append({"role": m["role"], "content": m["content"]})
                
                if file_context:
                    api_messages.append({"role": "system", "content": f"User reference materials:\n{file_context}"})

                try:
                    completion = client.chat.completions.create(
                        model=selected_model,
                        messages=api_messages,
                        stream=True,
                        timeout=30.0
                    )
                    
                    for chunk in completion:
                        if chunk.choices and chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            # Real-time stream passing through LaTeX renderer
                            message_placeholder.markdown(fix_latex_formatting(full_response) + "▌")
                    
                    cleaned_final_response = fix_latex_formatting(full_response)
                    message_placeholder.markdown(cleaned_final_response)
                    
                    # Save assistant response to state
                    st.session_state.db["threads"][st.session_state.current_thread].append({
                        "role": "assistant", 
                        "content": cleaned_final_response
                    })
                    save_data(st.session_state.db)

                except Exception as e:
                    st.error(f"API Error: {str(e)}")

# ------------------------------------------------------------------------------
# TAB 2: MCQ EVALUATION WORKSPACE
# ------------------------------------------------------------------------------
with tab_mcq:
    st.header("MCQ Generator & Evaluator")
    mcq_topic = st.text_input("Enter Topic for MCQ Quiz", "Linear Regression & Mathematical Statistics")
    
    if st.button("Generate Practice MCQ"):
        client = get_openrouter_client()
        if client:
            prompt = f"Create a 3-question multiple choice quiz on '{mcq_topic}'. Include questions, 4 options each, correct answers, and LaTeX explanations for math expressions."
            res = client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
            )
            quiz_content = fix_latex_formatting(res.choices[0].message.content)
            st.markdown(quiz_content)
            
            st.session_state.db["evaluations"]["mcq"].append({"topic": mcq_topic, "quiz": quiz_content})
            save_data(st.session_state.db)

# ------------------------------------------------------------------------------
# TAB 3: ESSAY EVALUATION WORKSPACE
# ------------------------------------------------------------------------------
with tab_essay:
    st.header("Essay & Problem Evaluator")
    essay_text = st.text_area("Paste your solution, response, or essay here", height=200)
    rubric = st.text_input("Evaluation Focus / Criteria", "Mathematical Accuracy, Clarity, Logic")
    
    if st.button("Evaluate Submission"):
        if essay_text:
            client = get_openrouter_client()
            if client:
                prompt = f"Evaluate the following text against these criteria: '{rubric}'. Provide score, feedback, and LaTeX formula corrections if relevant.\n\nSubmission:\n{essay_text}"
                res = client.chat.completions.create(
                    model=selected_model,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
                )
                evaluation_res = fix_latex_formatting(res.choices[0].message.content)
                st.markdown(evaluation_res)
                
                st.session_state.db["evaluations"]["essay"].append({"input": essay_text, "evaluation": evaluation_res})
                save_data(st.session_state.db)
