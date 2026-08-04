import streamlit as st
import json
import os
import time
from datetime import datetime
import google.genai as genai
from google.genai import types
from docx import Document

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Agentic AI Study Dashboard", page_icon="🎓", layout="wide")

DATA_FILE = "study_data.json"

# --- PERSISTENT DATA HELPERS ---
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"subjects": {}}

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# --- INITIALIZE DATABASE IN SESSION STATE ---
if "db" not in st.session_state:
    st.session_state.db = load_data()
    st.session_state.db = load_data()

# --- DOCX EXPORT GENERATOR ---
def generate_docx(subject, data):
    doc = Document()
    doc.add_heading(f"Revision Guide: {subject}", level=0)
    
    mcq_mistakes = data["subjects"].get(subject, {}).get("mcq_mistakes", [])
    writing_mistakes = data["subjects"].get(subject, {}).get("writing_mistakes", [])
    
    doc.add_heading("MCQ Conceptual Weak Spots", level=1)
    if not mcq_mistakes:
        doc.add_paragraph("No recorded MCQ mistakes yet.")
    for item in mcq_mistakes:
        p = doc.add_paragraph(style='List Bullet')
        p.add_run(f"[{item['date']}] Concept: ").bold = True
        p.add_run(f"{item['concept']}\n")
        p.add_run(f"Takeaway: {item['takeaway']}")

    doc.add_heading("Focus Writing Structural & Grammar Pitfalls", level=1)
    if not writing_mistakes:
        doc.add_paragraph("No recorded writing mistakes yet.")
    for item in writing_mistakes:
        p = doc.add_paragraph(style='List Bullet')
        p.add_run(f"[{item['date']}] Area: ").bold = True
        p.add_run(f"{item['area']}\n")
        p.add_run(f"Correction/Strategy: {item['correction']}")
        
    filename = f"{subject}_Revision_Guide.docx"
    doc.save(filename)
    return filename

# --- SIDEBAR & GLOBAL SETTINGS ---
st.sidebar.title("🎓 Study Agent Control")

api_key = st.sidebar.text_input("Gemini API Key", type="password")
if not api_key:
    st.info("👈 Please input your Google Gemini API Key in the sidebar to start.")
    st.stop()

# Initialize Gemini Client
client = genai.Client(api_key=api_key)

# Language Selection
language = st.sidebar.selectbox("Preferred Output Language / ভাষা", ["English", "Bengali (বাংলা)"])

# Subject Management
subjects = list(st.session_state.db["subjects"].keys())
if not subjects:
    subjects = ["General"]
    st.session_state.db["subjects"]["General"] = {"chat": [], "mcq_mistakes": [], "writing_mistakes": []}
    save_data(st.session_state.db)

selected_subject = st.sidebar.selectbox("Select Subject / Section", subjects)

new_subj = st.sidebar.text_input("Create New Subject")
if st.sidebar.button("Add Subject") and new_subj:
    if new_subj not in st.session_state.db["subjects"]:
        st.session_state.db["subjects"][new_subj] = {"chat": [], "mcq_mistakes": [], "writing_mistakes": []}
        save_data(st.session_state.db)
        st.rerun()

st.title(f"📚 {selected_subject} - Study Dashboard")

# --- NAVIGATION TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 General Chat", 
    "📝 MCQ Exam Generator", 
    "✍️ Focus Writing Evaluator", 
    "📖 Revision Notes & Export"
])

# ==========================================
# TAB 1: GENERAL CHAT
# ==========================================
with tab1:
    st.subheader("General Chat & Document Q&A")
    
    # Subject-specific chat history
    chat_history = st.session_state.db["subjects"][selected_subject].get("chat", [])
    
    for msg in chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    user_query = st.chat_input("Ask anything about this subject...")
    if user_query:
        # Save user message
        chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        sys_instruction = f"You are an expert tutor in {selected_subject}. Respond in {language}. Use clear, structured formatting and LaTeX ($ or $$) for math formulas."
        
        with st.chat_message("assistant"):
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=user_query,
                config=types.GenerateContentConfig(system_instruction=sys_instruction)
            )
            st.markdown(response.text)
            chat_history.append({"role": "assistant", "content": response.text})
            st.session_state.db["subjects"][selected_subject]["chat"] = chat_history
            save_data(st.session_state.db)

# ==========================================
# TAB 2: MCQ EXAM GENERATOR
# ==========================================
with tab2:
    st.subheader("Custom MCQ Exam Generator")
    
    col1, col2 = st.columns(2)
    with col1:
        num_mcqs = st.number_input("Number of Questions", min_value=1, max_value=20, value=5)
        difficulty = st.selectbox("Difficulty Level", ["Easy", "Medium", "Hard", "Advanced Exam Level"])
    with col2:
        timer_minutes = st.number_input("Practice Timer (Minutes)", min_value=1, max_value=120, value=10)
        mcq_instructions = st.text_area("Custom Instructions (e.g. Focus on Chapter 3 formulas, clinical scenarios, etc.)", "")

    if st.button("Generate Quiz"):
        prompt = f"""
        Generate {num_mcqs} multiple-choice questions for the subject '{selected_subject}'.
        Difficulty: {difficulty}.
        Additional Instructions: {mcq_instructions}.
        Language: {language}.
        
        Return ONLY valid JSON matching this schema:
        [
          {{
            "id": 1,
            "question": "Question text here",
            "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
            "correct": "A) ...",
            "explanation": "Detailed explanation here"
          }}
        ]
        """
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        st.session_state.current_quiz = json.loads(response.text)
        st.session_state.quiz_start_time = time.time()
        st.session_state.timer_duration = timer_minutes * 60

    if "current_quiz" in st.session_state:
        # Timer display
        elapsed = time.time() - st.session_state.quiz_start_time
        remaining = st.session_state.timer_duration - elapsed
        
        if remaining > 0:
            st.warning(f"⏱️ Time Remaining: {int(remaining // 60)}m {int(remaining % 60)}s")
        else:
            st.error("⏰ Time is up! Submit your answers for evaluation.")
            
        user_answers = {}
        with st.form("quiz_form"):
            for q in st.session_state.current_quiz:
                st.markdown(f"**Q{q['id']}: {q['question']}**")
                user_answers[q['id']] = st.radio(f"Select answer for Q{q['id']}", q['options'], key=f"q_{q['id']}")
                st.write("---")
            
            submit_quiz = st.form_submit_button("Submit Exam")
            
        if submit_quiz:
            score = 0
            new_mistakes = []
            
            for q in st.session_state.current_quiz:
                selected = user_answers[q['id']]
                if selected == q['correct']:
                    score += 1
                    st.success(f"Q{q['id']}: Correct! {q['explanation']}")
                else:
                    st.error(f"Q{q['id']}: Incorrect. Selected: {selected} | Correct: {q['correct']}")
                    st.info(f"Explanation: {q['explanation']}")
                    
                    # Log mistake
                    new_mistakes.append({
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "concept": q['question'],
                        "takeaway": q['explanation']
                    })
            
            st.balloons()
            st.metric("Final Score", f"{score} / {len(st.session_state.current_quiz)}")
            
            # Save mistakes to database
            existing_mistakes = st.session_state.db["subjects"][selected_subject].get("mcq_mistakes", [])
            existing_mistakes.extend(new_mistakes)
            st.session_state.db["subjects"][selected_subject]["mcq_mistakes"] = existing_mistakes
            save_data(st.session_state.db)
            st.success("Weak points automatically added to your Revision Guide!")

# ==========================================
# TAB 3: FOCUS WRITING EVALUATOR
# ==========================================
with tab3:
    st.subheader("Focus Writing Diagnostic")
    
    target_style = st.text_area("Benchmark / Demo Style Sample (Paste standard reference text)", height=100)
    topic_prompt = st.text_input("Writing Topic / Prompt")
    user_essay = st.text_area("Your Written Essay / Response", height=200)
    custom_writing_instructions = st.text_area("Custom Grading Criteria (e.g., Strict academic tone, active voice, line-by-line grammar check)", "")
    
    if st.button("Evaluate Essay") and user_essay:
        prompt = f"""
        Evaluate this essay for subject '{selected_subject}'.
        Topic: {topic_prompt}
        Benchmark Style: {target_style}
        Essay: {user_essay}
        Custom Criteria: {custom_writing_instructions}
        Output Language: {language}
        
        Provide response as JSON:
        {{
          "style_score": "85%",
          "grammar_corrections": ["Line/Phrase -> Corrected Version: Reason"],
          "vocab_upgrades": ["Original Word -> Advanced Synonym"],
          "recurring_weakness_area": "Brief name of main writing weakness",
          "actionable_strategy": "Detailed strategy to fix this weakness"
        }}
        """
        
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        result = json.loads(response.text)
        
        st.metric("Style Match Score", result["style_score"])
        
        st.markdown("### 🔍 Grammar & Syntax Line Corrections")
        for corr in result["grammar_corrections"]:
            st.write(f"- {corr}")
            
        st.markdown("### 💡 Vocabulary Upgrades")
        for v in result["vocab_upgrades"]:
            st.write(f"- {v}")
            
        # Log writing mistake automatically
        writing_mistakes = st.session_state.db["subjects"][selected_subject].get("writing_mistakes", [])
        writing_mistakes.append({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "area": result["recurring_weakness_area"],
            "correction": result["actionable_strategy"]
        })
        st.session_state.db["subjects"][selected_subject]["writing_mistakes"] = writing_mistakes
        save_data(st.session_state.db)
        st.success("Writing feedback saved to your Revision Log!")

# ==========================================
# TAB 4: REVISION NOTES & WORD EXPORT
# ==========================================
with tab4:
    st.subheader(f"Unified Revision Log: {selected_subject}")
    
    subj_data = st.session_state.db["subjects"].get(selected_subject, {})
    mcq_logs = subj_data.get("mcq_mistakes", [])
    writing_logs = subj_data.get("writing_mistakes", [])
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("### 📝 MCQ Weak Spots")
        if not mcq_logs:
            st.write("No MCQ mistakes logged yet.")
        for idx, m in enumerate(mcq_logs):
            with st.expander(f"[{m['date']}] {m['concept'][:40]}..."):
                st.write(f"**Concept:** {m['concept']}")
                st.write(f"**Takeaway:** {m['takeaway']}")
                if st.button(f"Delete Entry #{idx+1}", key=f"del_mcq_{idx}"):
                    mcq_logs.pop(idx)
                    st.session_state.db["subjects"][selected_subject]["mcq_mistakes"] = mcq_logs
                    save_data(st.session_state.db)
                    st.rerun()

    with col_b:
        st.markdown("### ✍️ Writing Pitfalls")
        if not writing_logs:
            st.write("No writing mistakes logged yet.")
        for idx, w in enumerate(writing_logs):
            with st.expander(f"[{w['date']}] {w['area']}"):
                st.write(f"**Area:** {w['area']}")
                st.write(f"**Strategy:** {w['correction']}")
                if st.button(f"Delete Entry #{idx+1}", key=f"del_w_{idx}"):
                    writing_logs.pop(idx)
                    st.session_state.db["subjects"][selected_subject]["writing_mistakes"] = writing_logs
                    save_data(st.session_state.db)
                    st.rerun()
                    
    st.write("---")
    if st.button("Generate & Download Word (.docx) Revision Guide"):
        file_path = generate_docx(selected_subject, st.session_state.db)
        with open(file_path, "rb") as fp:
            st.download_button(
                label="📥 Click Here to Download .docx",
                data=fp,
                file_name=file_path,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )# Code goes here.
