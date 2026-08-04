import streamlit as st
import json
import os
import time
from datetime import datetime
import google.genai as genai
from google.genai import types
from docx import Document

st.set_page_config(page_title="Agentic Study Platform", page_icon="🎓", layout="wide")

DATA_FILE = "study_data.json"

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "chats": {"Default Chat": []},
        "mcq_subjects": {
            "General Math": {"sources": [], "chat": [], "mistakes": []}
        },
        "written_subjects": {
            "Academic Essay": {"sources": [], "chat": [], "mistakes": []}
        }
    }

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if "db" not in st.session_state:
    st.session_state.db = load_data()

if "active_mode" not in st.session_state:
    st.session_state.active_mode = "general_chat"

if "active_chat" not in st.session_state:
    st.session_state.active_chat = "Default Chat"

def generate_docx(subject_name, mistakes_list, section_type="MCQ"):
    doc = Document()
    doc.add_heading(f"{section_type} Revision Guide: {subject_name}", level=0)
    
    if not mistakes_list:
        doc.add_paragraph("No weak spots logged yet.")
    for item in mistakes_list:
        p = doc.add_paragraph(style='List Bullet')
        p.add_run(f"[{item['date']}] ").bold = True
        p.add_run(f"{item.get('concept', item.get('area', 'Topic'))}\n")
        p.add_run(f"Takeaway: {item.get('takeaway', item.get('correction', 'N/A'))}")
        
    filename = f"{subject_name}_{section_type}_Revision.docx"
    doc.save(filename)
    return filename

# --- SIDEBAR: GEMINI GENERAL CHAT & NAVIGATION ---
with st.sidebar:
    st.title("🎓 Gemini + Notebook Studio")
    
    api_key = st.text_input("Gemini API Key", type="password")
    if not api_key:
        st.info("👈 Enter your Gemini API Key to activate features.")
        st.stop()

    client = genai.Client(api_key=api_key)
    language = st.selectbox("Preferred Output Language / ভাষা", ["English", "Bengali (বাংলা)"])

    st.divider()
    
    st.subheader("📌 Navigation")
    nav_choice = st.radio("Select Workspace View", ["💬 Gemini General Chat", "📚 Notebook Workspaces"])
    if nav_choice == "💬 Gemini General Chat":
        st.session_state.active_mode = "general_chat"
    else:
        st.session_state.active_mode = "notebook_studio"

    st.divider()

    if st.session_state.active_mode == "general_chat":
        col_c1, col_c2 = st.columns([3, 1])
        col_c1.subheader("💬 Chat Threads")
        if col_c2.button("➕", key="new_chat"):
            new_chat_name = f"Chat {datetime.now().strftime('%b %d %H:%M')}"
            st.session_state.db["chats"][new_chat_name] = []
            save_data(st.session_state.db)
            st.session_state.active_chat = new_chat_name
            st.rerun()

        for chat_name in list(st.session_state.db["chats"].keys()):
            if st.button(f"🗨️ {chat_name[:20]}", key=f"chat_{chat_name}"):
                st.session_state.active_chat = chat_name
                st.rerun()

# ==========================================
# VIEW 1: GEMINI GENERAL CHAT
# ==========================================
if st.session_state.active_mode == "general_chat":
    st.title(f"💬 {st.session_state.active_chat}")
    
    chat_history = st.session_state.db["chats"].get(st.session_state.active_chat, [])
    for msg in chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    uploaded_files = st.file_uploader(
        "Attach Files / Images to General Chat", 
        type=["png", "jpg", "jpeg", "pdf", "txt"], 
        accept_multiple_files=True,
        key=f"gen_upload_{st.session_state.active_chat}"
    )
    
    user_query = st.chat_input("Ask anything...")
    if user_query:
        chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        sys_inst = f"You are a general study assistant. Respond in {language}. Use LaTeX ($ or $$) for math formulas."
        contents = [user_query]
        
        if uploaded_files:
            for file in uploaded_files:
                uploaded_part = client.files.upload(file=file)
                contents.append(uploaded_part)
                
        with st.chat_message("assistant"):
            res = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=sys_inst)
            )
            st.markdown(res.text)
            chat_history.append({"role": "assistant", "content": res.text})
            st.session_state.db["chats"][st.session_state.active_chat] = chat_history
            save_data(st.session_state.db)

# ==========================================
# VIEW 2: NOTEBOOK WORKSPACES (MCQ & WRITTEN)
# ==========================================
elif st.session_state.active_mode == "notebook_studio":
    st.title("📚 Notebook Workspaces")
    
    workspace_type = st.radio("Select Section Workspace", ["📝 MCQ Workspace", "✍️ Focus Written Workspace"], horizontal=True)

    # ----------------------------------------------------
    # SECTION 1: MCQ WORKSPACE
    # ----------------------------------------------------
    if workspace_type == "📝 MCQ Workspace":
        st.subheader("📝 MCQ Subjects Studio")
        
        col_s1, col_s2 = st.columns([3, 1])
        mcq_subs = list(st.session_state.db["mcq_subjects"].keys())
        selected_mcq_sub = col_s1.selectbox("Select Subject", mcq_subs if mcq_subs else ["None"])
        
        new_mcq_sub = col_s2.text_input("Create New MCQ Subject")
        if col_s2.button("Add MCQ Subject") and new_mcq_sub:
            if new_mcq_sub not in st.session_state.db["mcq_subjects"]:
                st.session_state.db["mcq_subjects"][new_mcq_sub] = {"sources": [], "chat": [], "mistakes": []}
                save_data(st.session_state.db)
                st.success(f"Added {new_mcq_sub}")
                st.rerun()

        if selected_mcq_sub and selected_mcq_sub != "None":
            sub_data = st.session_state.db["mcq_subjects"][selected_mcq_sub]
            
            with st.expander("📁 Import / Manage Subject Source Documents", expanded=False):
                files = st.file_uploader("Upload files for this subject", accept_multiple_files=True, key=f"mcq_files_{selected_mcq_sub}")
                if st.button("Save Sources to Subject"):
                    if files:
                        for f in files:
                            if f.name not in sub_data["sources"]:
                                sub_data["sources"].append(f.name)
                        save_data(st.session_state.db)
                        st.success("Sources updated!")
                        st.rerun()
                st.write("**Current Sources:**", sub_data["sources"])

            m_tab1, m_tab2, m_tab3 = st.tabs(["🎯 Exam Generator", "💬 Subject Q&A Chat", "📖 Mistakes & Revision Log"])

            with m_tab1:
                c1, c2 = st.columns(2)
                with c1:
                    num_q = st.number_input("Number of Questions", 1, 20, 5)
                    diff = st.selectbox("Difficulty", ["Easy", "Medium", "Hard", "Advanced Exam Level"])
                with c2:
                    timer_m = st.number_input("Practice Timer (Minutes)", 1, 120, 10)
                    custom_inst = st.text_area("Custom Prompt / Focus Chapter", "")

                if st.button("Generate Practice Quiz"):
                    prompt = f"Generate {num_q} MCQs for '{selected_mcq_sub}'. Difficulty: {diff}. Instructions: {custom_inst}. Language: {language}. Return strictly JSON format: [{{'id':1,'question':'...','options':['A)...','B)...'],'correct':'A)...','explanation':'...'}}]"
                    res = client.models.generate_content(
                        model="gemini-3.5-flash",
                        contents=prompt,
                        config=types.GenerateContentConfig(response_mime_type="application/json")
                    )
                    st.session_state.quiz = json.loads(res.text)
                    st.session_state.quiz_start = time.time()
                    st.session_state.quiz_duration = timer_m * 60

                if "quiz" in st.session_state:
                    elapsed = time.time() - st.session_state.quiz_start
                    rem = st.session_state.quiz_duration - elapsed
                    if rem > 0:
                        st.warning(f"⏱️ Time Remaining: {int(rem // 60)}m {int(rem % 60)}s")
                    else:
                        st.error("⏰ Time Expired!")

                    user_ans = {}
                    with st.form("mcq_form"):
                        for q in st.session_state.quiz:
                            st.write(f"**Q{q['id']}: {q['question']}**")
                            user_ans[q['id']] = st.radio(f"Choose option Q{q['id']}", q['options'])
                        submit_m = st.form_submit_button("Submit Quiz")

                    if submit_m:
                        score = 0
                        new_m = []
                        for q in st.session_state.quiz:
                            if user_ans[q['id']] == q['correct']:
                                score += 1
                                st.success(f"Q{q['id']} Correct!")
                            else:
                                st.error(f"Q{q['id']} Incorrect. Answer: {q['correct']}")
                                new_m.append({"date": datetime.now().strftime("%Y-%m-%d"), "concept": q['question'], "takeaway": q['explanation']})
                        st.metric("Score", f"{score} / {len(st.session_state.quiz)}")
                        sub_data["mistakes"].extend(new_m)
                        save_data(st.session_state.db)

            with m_tab2:
                for chat in sub_data.get("chat", []):
                    with st.chat_message(chat["role"]):
                        st.markdown(chat["content"])
                q_in = st.chat_input(f"Ask about {selected_mcq_sub}...")
                if q_in:
                    sub_data["chat"].append({"role": "user", "content": q_in})
                    res = client.models.generate_content(
                        model="gemini-3.5-flash",
                        contents=q_in,
                        config=types.GenerateContentConfig(system_instruction=f"Answer for {selected_mcq_sub} in {language} based on sources: {sub_data['sources']}")
                    )
                    sub_data["chat"].append({"role": "assistant", "content": res.text})
                    save_data(st.session_state.db)
                    st.rerun()

            with m_tab3:
                st.write("### Recorded Mistake Entries")
                for idx, m in enumerate(sub_data.get("mistakes", [])):
                    col_m1, col_m2 = st.columns([5, 1])
                    col_m1.write(f"- **{m['concept']}**: {m['takeaway']}")
                    if col_m2.button("Delete", key=f"del_mcq_{selected_mcq_sub}_{idx}"):
                        sub_data["mistakes"].pop(idx)
                        save_data(st.session_state.db)
                        st.rerun()
                
                if st.button("Export MCQ Revision Guide (.docx)"):
                    fpath = generate_docx(selected_mcq_sub, sub_data["mistakes"], "MCQ")
                    with open(fpath, "rb") as fp:
                        st.download_button("📥 Download Document", fp, file_name=fpath)

    # ----------------------------------------------------
    # SECTION 2: FOCUS WRITTEN WORKSPACE
    # ----------------------------------------------------
    elif workspace_type == "✍️ Focus Written Workspace":
        st.subheader("✍️ Focus Written Subjects Studio")
        
        col_w1, col_w2 = st.columns([3, 1])
        w_subs = list(st.session_state.db["written_subjects"].keys())
        selected_w_sub = col_w1.selectbox("Select Subject", w_subs if w_subs else ["None"])
        
        new_w_sub = col_w2.text_input("Create New Written Subject")
        if col_w2.button("Add Written Subject") and new_w_sub:
            if new_w_sub not in st.session_state.db["written_subjects"]:
                st.session_state.db["written_subjects"][new_w_sub] = {"sources": [], "chat": [], "mistakes": []}
                save_data(st.session_state.db)
                st.success(f"Added {new_w_sub}")
                st.rerun()

        if selected_w_sub and selected_w_sub != "None":
            w_sub_data = st.session_state.db["written_subjects"][selected_w_sub]
            
            with st.expander("📁 Import / Manage Subject Source Documents", expanded=False):
                w_files = st.file_uploader("Upload files for this subject", accept_multiple_files=True, key=f"w_files_{selected_w_sub}")
                if st.button("Save Sources to Subject"):
                    if w_files:
                        for f in w_files:
                            if f.name not in w_sub_data["sources"]:
                                w_sub_data["sources"].append(f.name)
                        save_data(st.session_state.db)
                        st.success("Sources updated!")
                        st.rerun()
                st.write("**Current Sources:**", w_sub_data["sources"])

            w_t1, w_t2, w_t3 = st.tabs(["✍️ Evaluator & Benchmark", "💬 Subject Q&A Chat", "📖 Revision Log & Export"])

            with w_t1:
                target_benchmark = st.text_area("Benchmark Writing Sample")
                essay = st.text_area("Your Essay Input", height=150)
                if st.button("Evaluate Essay") and essay:
                    prompt = f"Evaluate essay for '{selected_w_sub}'. Benchmark: {target_benchmark}. Essay: {essay}. Output JSON: {{'score':'80%','weakness':'...','strategy':'...'}}"
                    res = client.models.generate_content(
                        model="gemini-3.5-flash",
                        contents=prompt,
                        config=types.GenerateContentConfig(response_mime_type="application/json")
                    )
                    eval_data = json.loads(res.text)
                    st.metric("Style Match", eval_data["score"])
                    w_sub_data["mistakes"].append({"date": datetime.now().strftime("%Y-%m-%d"), "area": eval_data["weakness"], "correction": eval_data["strategy"]})
                    save_data(st.session_state.db)
                    st.success("Writing feedback saved to log!")

            with w_t2:
                for chat in w_sub_data.get("chat", []):
                    with st.chat_message(chat["role"]):
                        st.markdown(chat["content"])
                wq_in = st.chat_input(f"Ask about writing in {selected_w_sub}...")
                if wq_in:
                    w_sub_data["chat"].append({"role": "user", "content": wq_in})
                    res = client.models.generate_content(
                        model="gemini-3.5-flash",
                        contents=wq_in,
                        config=types.GenerateContentConfig(system_instruction=f"Answer for {selected_w_sub} in {language} using sources: {w_sub_data['sources']}")
                    )
                    w_sub_data["chat"].append({"role": "assistant", "content": res.text})
                    save_data(st.session_state.db)
                    st.rerun()

            with w_t3:
                st.write("### Recorded Writing Structural & Grammar Pitfalls")
                for idx, w in enumerate(w_sub_data.get("mistakes", [])):
                    col_w1, col_w2 = st.columns([5, 1])
                    col_w1.write(f"- **{w['area']}**: {w['correction']}")
                    if col_w2.button("Delete", key=f"del_w_{selected_w_sub}_{idx}"):
                        w_sub_data["mistakes"].pop(idx)
                        save_data(st.session_state.db)
                        st.rerun()
                
                if st.button("Export Written Revision Guide (.docx)"):
                    fpath = generate_docx(selected_w_sub, w_sub_data["mistakes"], "Writing")
                    with open(fpath, "rb") as fp:
                        st.download_button("📥 Download Document", fp, file_name=fpath)
