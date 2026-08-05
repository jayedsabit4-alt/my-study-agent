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
# 1. LATEX FORMATTING FIX & CORE UTILITIES
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
    
    # 3. Handle standalone lines starting with LaTeX math commands missing delimiters
    text = re.sub(r'(?m)^\\(frac|sqrt|left|mathrm|mathbf|boldsymbol)\{.*\}$', r'$$\g<0>$$', text)
    
    return text

def load_data():
    """
    Loads persistent study data with fallback structures to prevent state crashes.
    """
    default_structure = {
        "threads": {"General Chat": []},
        "notes": {},
        "mistakes": [],
        "evaluations": {"mcq": [], "essay": []}
    }
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    return default_structure
                # Ensure all modular keys exist
                if "threads" not in data or not data["threads"]:
                    data["threads"] = {"General Chat": []}
                if "notes" not in data:
                    data["notes"] = {}
                if "mistakes" not in data:
                    data["mistakes"] = []
                if "evaluations" not in data:
                    data["evaluations"] = {"mcq": [], "essay": []}
                return data
        except Exception:
            return default_structure
    return default_structure

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
# 2. PAGE CONFIGURATION & STATE MANAGEMENT
# ==============================================================================
st.set_page_config(page_title="Notebook Workspace Platform", layout="wide", page_icon="📖")

if "db" not in st.session_state:
    st.session_state.db = load_data()

# Safe thread assignment
threads_dict = st.session_state.db.get("threads", {"General Chat": []})
if "current_thread" not in st.session_state or st.session_state.current_thread not in threads_dict:
    st.session_state.current_thread = list(threads_dict.keys())[0]

# Enhanced System Prompt enforcing mathematical syntax standards
SYSTEM_PROMPT = """You are an expert academic study platform assistant.
When outputting mathematical expressions, physics derivations, chemical formulas, or statistical notation:
1. ALWAYS format inline math using single dollar signs: $...$
2. ALWAYS format block equations using double dollar signs: $$...$$
3. NEVER output raw brackets like [ \\formula ] or ( \\symbol ) for LaTeX rendering.
Provide rigorous, structured, and complete mathematical derivations and technical explanations.
"""

AVAILABLE_MODELS = {
    "DeepSeek R1 (Free)": "deepseek/deepseek-r1:free",
    "Meta Llama 3.3 70B (Free)": "meta-llama/llama-3.3-70b-instruct:free",
    "Mistral Small 24B (Free)": "mistralai/mistral-small-24b-instruct-2501:free",
    "Google Gemini 2.0 Flash (Free)": "google/gemini-2.0-flash-exp:free"
}

# ==============================================================================
# 3. SIDEBAR NAVIGATION
# ==============================================================================
with st.sidebar:
    st.title("📖 Study Platform")
    
    api_key = st.text_input("OpenRouter API Key", type="password")
    selected_model_label = st.selectbox("LLM Architecture Model", list(AVAILABLE_MODELS.keys()))
    selected_model = AVAILABLE_MODELS[selected_model_label]
    
    st.markdown("---")
    navigation_mode = st.radio("Navigation View", ["💬 General Chat", "📓 Notebook Workspace", "⚠️ Mistake Tracker", "📝 Practice Generator"])
    
    st.markdown("---")
    st.subheader("Threads & Workspaces")
    
    new_thread_name = st.text_input("New Session Name")
    if st.button("➕ Create Session") and new_thread_name:
        if new_thread_name not in st.session_state.db["threads"]:
            st.session_state.db["threads"][new_thread_name] = []
            st.session_state.current_thread = new_thread_name
            save_data(st.session_state.db)
            st.rerun()

    threads_list = list(st.session_state.db["threads"].keys())
    current_index = threads_list.index(st.session_state.current_thread) if st.session_state.current_thread in threads_list else 0
    st.session_state.current_thread = st.selectbox(
        "Active Thread",
        threads_list,
        index=current_index
    )

    st.markdown("---")
    uploaded_files = st.file_uploader("Multimodal Context (PDF, DOCX, CSV)", accept_multiple_files=True)

def get_openrouter_client():
    if not api_key:
        st.warning("Please enter your OpenRouter API Key in the sidebar.")
        return None
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

# Context parsing from uploaded files
file_context = ""
if uploaded_files:
    for f in uploaded_files:
        file_context += f"\n\n--- Content from {f.name} ---\n" + parse_file(f)

# ==============================================================================
# 4. MODULAR VIEWS
# ==============================================================================

# ------------------------------------------------------------------------------
# VIEW 1: GENERAL CHAT
# ------------------------------------------------------------------------------
if navigation_mode == "💬 General Chat":
    st.header(f"Session Chat: {st.session_state.current_thread}")
    
    current_messages = st.session_state.db["threads"].get(st.session_state.current_thread, [])
    for msg in current_messages:
        with st.chat_message(msg["role"]):
            st.markdown(fix_latex_formatting(msg["content"]))

    if user_query := st.chat_input("Ask a question, formula derivation, or technical query..."):
        st.session_state.db["threads"][st.session_state.current_thread].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        client = get_openrouter_client()
        if client:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                for m in st.session_state.db["threads"][st.session_state.current_thread]:
                    api_messages.append({"role": m["role"], "content": m["content"]})
                
                if file_context:
                    api_messages.append({"role": "system", "content": f"Reference Materials:\n{file_context}"})

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
                            message_placeholder.markdown(fix_latex_formatting(full_response) + "▌")
                    
                    cleaned_final_response = fix_latex_formatting(full_response)
                    message_placeholder.markdown(cleaned_final_response)
                    
                    st.session_state.db["threads"][st.session_state.current_thread].append({
                        "role": "assistant", 
                        "content": cleaned_final_response
                    })
                    save_data(st.session_state.db)

                except Exception as e:
                    st.error(f"API Error: {str(e)}")

# ------------------------------------------------------------------------------
# VIEW 2: NOTEBOOK WORKSPACE
# ------------------------------------------------------------------------------
elif navigation_mode == "📓 Notebook Workspace":
    st.header("Persistent Study Notebook & Notes")
    
    col_notes, col_preview = st.columns([1, 1])
    
    with col_notes:
        st.subheader("Drafting Canvas")
        note_title = st.text_input("Note Topic", value="General Linear Models & Regression")
        existing_content = st.session_state.db["notes"].get(note_title, "")
        note_body = st.text_area("Write/Edit Note (LaTeX Supported)", value=existing_content, height=400)
        
        if st.button("💾 Save Note"):
            st.session_state.db["notes"][note_title] = note_body
            save_data(st.session_state.db)
            st.success("Note persistent state saved!")

    with col_preview:
        st.subheader("Rendered Preview")
        st.markdown(f"### {note_title}")
        st.markdown(fix_latex_formatting(note_body))

# ------------------------------------------------------------------------------
# VIEW 3: MISTAKE TRACKER
# ------------------------------------------------------------------------------
elif navigation_mode == "⚠️ Mistake Tracker":
    st.header("Mistake & Revision Logging System")
    
    with st.form("add_mistake"):
        m_topic = st.text_input("Subject / Topic", "Bayes' Theorem / Conditional Probability")
        m_desc = st.text_area("Mistake Description or Problem Statement")
        m_correction = st.text_area("Correct Solution & Formula Derivation")
        submit_m = st.form_submit_button("Log Mistake")
        
        if submit_m and m_desc:
            st.session_state.db["mistakes"].append({
                "topic": m_topic,
                "problem": m_desc,
                "correction": m_correction
            })
            save_data(st.session_state.db)
            st.success("Logged to mistake register!")

    st.markdown("---")
    st.subheader("Logged Mistake Register")
    for idx, item in enumerate(st.session_state.db["mistakes"]):
        with st.expander(f"#{idx+1} | {item['topic']}"):
            st.markdown("**Problem:**")
            st.write(item["problem"])
            st.markdown("**Correct Derivation / Concept:**")
            st.markdown(fix_latex_formatting(item["correction"]))

# ------------------------------------------------------------------------------
# VIEW 4: PRACTICE GENERATOR
# ------------------------------------------------------------------------------
elif navigation_mode == "📝 Practice Generator":
    st.header("Practice & Evaluation Workspace")
    
    tab_gen, tab_eval = st.tabs(["📝 Generate Questions", "📄 Solution Evaluator"])
    
    with tab_gen:
        eval_topic = st.text_input("Target Topic for Evaluation", "Maxwell–Boltzmann Speed Distribution")
        if st.button("Generate Practice Questions"):
            client = get_openrouter_client()
            if client:
                prompt = f"Create a comprehensive study quiz with solutions on '{eval_topic}'. Format all formulas cleanly using double dollar signs for equations and single dollar signs for inline variables."
                res = client.chat.completions.create(
                    model=selected_model,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
                )
                quiz = fix_latex_formatting(res.choices[0].message.content)
                st.markdown(quiz)
                st.session_state.db["evaluations"]["mcq"].append({"topic": eval_topic, "quiz": quiz})
                save_data(st.session_state.db)
                
    with tab_eval:
        user_solution = st.text_area("Paste your technical solution for AI review", height=200)
        rubric_criteria = st.text_input("Criteria", "Mathematical Rigor, Formula Accuracy")
        if st.button("Run Evaluation"):
            if user_solution:
                client = get_openrouter_client()
                if client:
                    prompt = f"Evaluate this solution against '{rubric_criteria}'. Provide score and corrections:\n\n{user_solution}"
                    res = client.chat.completions.create(
                        model=selected_model,
                        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
                    )
                    eval_out = fix_latex_formatting(res.choices[0].message.content)
                    st.markdown(eval_out)
                    st.session_state.db["evaluations"]["essay"].append({"input": user_solution, "evaluation": eval_out})
                    save_data(st.session_state.db)
