import json
import os
import re
from datetime import datetime
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Agentic Study Platform", page_icon="🎓", layout="wide"
)

DATA_FILE = "study_data.json"

MODEL_OPTIONS = {
    "openrouter/free (Auto-Router - Multimodal & Fast)": "openrouter/free",
    "google/gemma-4-31b-it:free (Vision & Document OCR)": (
        "google/gemma-4-31b-it:free"
    ),
    "nvidia/nemotron-3-ultra-550b-a55b:free (Deep Reasoning)": (
        "nvidia/nemotron-3-ultra-550b-a55b:free"
    ),
    "openai/gpt-oss-20b:free (Lightweight Chat)": "openai/gpt-oss-20b:free",
}

MATH_SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "When rendering mathematical variables, formulas, or statistical notation, "
        "ALWAYS wrap them in standard LaTeX dollar signs. Use single dollar signs "
        "for inline math (e.g., $\\mu$, $\\bar{x}$, $\\theta$) and double dollar signs "
        "for standalone equations ($$...$$). NEVER use raw brackets like [ \\formula ] or "
        "( \\symbol ) for LaTeX equations."
    ),
}


# --- LATEX REGEX CLEANER ---
def fix_latex_formatting(text: str) -> str:
    """Cleans up raw LLM bracket outputs and converts them to standard KaTeX delimiters."""
    if not text:
        return ""
    text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\w)\[\s*(\\.*?)\s*\]", r"$$\1$$", text, flags=re.DOTALL)
    text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\w)\(\s*(\\.*?)\s*\)", r"$\1$", text, flags=re.DOTALL)
    text = re.sub(
        r"(?m)^\\(frac|sqrt|left|mathrm|mathbf|begin|end|boldsymbol)\{.*\}$",
        r"$$\g<0>$$",
        text,
    )
    return text


# --- SAFE JSON RESPONSE CLEANER ---
def clean_json_response(content: str) -> str:
    """Safely extracts raw JSON arrays or objects from markdown responses."""
    if not content:
        return ""
    cleaned = re.sub(r"^```(?:json)?", "", content.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
    match = re.search(r"([\[\{].*[\]\}])", cleaned, re.DOTALL)
    return match.group(1) if match else cleaned.strip()


# --- LAZY CLIENT INITIALIZATION ---
def get_openrouter_client(api_key):
    from openai import OpenAI

    return OpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=30.0
    )


# --- DATA PERSISTENCE ---
def load_data():
    default_structure = {
        "chats": {"Default Chat": []},
        "mcq_subjects": {
            "General Math": {
                "sources": [],
                "chat": [],
                "mistakes": [],
                "instructions": "",
            }
        },
        "written_subjects": {
            "Academic Essay": {
                "sources": [],
                "chat": [],
                "mistakes": [],
                "instructions": "Focus on clarity, analytical depth, and structural coherence.",
            }
        },
    }
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    return default_structure
                if "chats" not in data or not data["chats"]:
                    data["chats"] = {"Default Chat": []}
                if "mcq_subjects" not in data:
                    data["mcq_subjects"] = default_structure["mcq_subjects"]
                if "written_subjects" not in data:
                    data["written_subjects"] = default_structure["written_subjects"]
                return data
        except Exception:
            st.warning("⚠️ Warning: Could not read saved data. Initializing new state.")
    return default_structure


def save_data(data):
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Failed to save data: {e}")


# --- STATE INITIALIZATION ---
if "db" not in st.session_state:
    st.session_state.db = load_data()

if "active_mode" not in st.session_state:
    st.session_state.active_mode = "general_chat"

if "active_chat" not in st.session_state:
    existing_chats = list(st.session_state.db["chats"].keys())
    st.session_state.active_chat = (
        existing_chats[0] if existing_chats else "Default Chat"
    )


def process_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return ""
    file_type = uploaded_file.name.split(".")[-1].lower()
    extracted_text = f"\n--- Attached File: {uploaded_file.name} ---\n"
    try:
        if file_type in ["png", "jpg", "jpeg"]:
            extracted_text += f"[Image / Handwritten Document File: {uploaded_file.name}]"
        elif file_type == "pdf":
            from pypdf import PdfReader

            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                extracted_text += page.extract_text() or ""
        elif file_type == "docx":
            from docx import Document

            doc = Document(uploaded_file)
            extracted_text += "\n".join([p.text for p in doc.paragraphs])
        elif file_type in ["csv", "xlsx"]:
            df = (
                pd.read_csv(uploaded_file)
                if file_type == "csv"
                else pd.read_excel(uploaded_file)
            )
            extracted_text += df.to_string(index=False)
        else:
            extracted_text += uploaded_file.getvalue().decode("utf-8")
    except Exception as e:
        extracted_text += f"Error reading file: {str(e)}"
    return extracted_text


def generate_docx(subject_name, mistakes_list, section_type="MCQ"):
    from docx import Document

    doc = Document()
    doc.add_heading(f"{section_type} Revision Guide: {subject_name}", level=0)
    if not mistakes_list:
        doc.add_paragraph("No weak spots logged yet.")
    for item in mistakes_list:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(f"[{item.get('date', 'N/A')}] ").bold = True
        p.add_run(f"{item.get('concept', item.get('area', 'Topic'))}\n")
        p.add_run(
            f"Takeaway / Explanation: {item.get('takeaway', item.get('correction', 'N/A'))}"
        )
    filename = f"{subject_name}_{section_type}_Revision.docx"
    doc.save(filename)
    return filename


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.title("🎓 OpenRouter Studio")

    default_key = st.secrets.get(
        "OPENROUTER_API_KEY", st.session_state.get("saved_openrouter_key", "")
    )
    api_key = st.text_input(
        "OpenRouter API Key",
        value=default_key,
        type="password",
        help="Get a free key from openrouter.ai",
    )

    if api_key:
        st.session_state.saved_openrouter_key = api_key
    else:
        st.warning("⚠️ Enter your free OpenRouter API Key to start.")

    st.divider()

    nav_choice = st.radio(
        "📌 Navigation", ["💬 General Chat", "📚 Notebook Workspaces"]
    )
    st.session_state.active_mode = (
        "general_chat"
        if nav_choice == "💬 General Chat"
        else "notebook_studio"
    )

    st.divider()

    # --- CHAT THREAD MANAGER ---
    if st.session_state.active_mode == "general_chat":
        col_c1, col_c2 = st.columns([3, 1])
        col_c1.subheader("Threads")

        if col_c2.button("➕ New", key="btn_new_chat"):
            new_title = f"Chat {datetime.now().strftime('%H:%M:%S')}"
            st.session_state.db["chats"][new_title] = []
            save_data(st.session_state.db)
            st.session_state.active_chat = new_title
            st.rerun()

        for chat_name in list(st.session_state.db["chats"].keys()):
            col_btn, col_del = st.columns([4, 1])
            is_active = chat_name == st.session_state.active_chat
            label = f"👉 {chat_name[:12]}" if is_active else f"💬 {chat_name[:12]}"

            if col_btn.button(label, key=f"sel_{chat_name}"):
                st.session_state.active_chat = chat_name
                st.rerun()

            if len(st.session_state.db["chats"]) > 1:
                if col_del.button("🗑️", key=f"del_{chat_name}"):
                    del st.session_state.db["chats"][chat_name]
                    st.session_state.active_chat = list(
                        st.session_state.db["chats"].keys()
                    )[0]
                    save_data(st.session_state.db)
                    st.rerun()

        st.divider()
        rename_input = st.text_input("Rename Thread", st.session_state.active_chat)
        if (
            st.button("Update Title")
            and rename_input
            and rename_input != st.session_state.active_chat
        ):
            st.session_state.db["chats"][rename_input] = st.session_state.db[
                "chats"
            ].pop(st.session_state.active_chat)
            st.session_state.active_chat = rename_input
            save_data(st.session_state.db)
            st.rerun()


# ==========================================
# VIEW 1: GENERAL CHAT
# ==========================================
if st.session_state.active_mode == "general_chat":
    st.title(f"💬 {st.session_state.active_chat}")

    chat_history = st.session_state.db["chats"].get(
        st.session_state.active_chat, []
    )

    for msg in chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(fix_latex_formatting(msg["content"]))
            if msg["role"] == "assistant":
                with st.expander("📋 Copy Raw Text"):
                    st.code(msg["content"], language="markdown")

    st.markdown("---")

    col_ctrl1, col_ctrl2 = st.columns([1, 1])
    with col_ctrl1:
        selected_label = st.selectbox(
            "🌐 AI Model Architecture",
            options=list(MODEL_OPTIONS.keys()),
            index=0,
            key="chat_model_select",
        )
        selected_model_slug = MODEL_OPTIONS[selected_label]

    with col_ctrl2:
        attached_file = st.file_uploader(
            "Attach Context Material (PDF, DOCX, CSV, Image)",
            type=["png", "jpg", "jpeg", "pdf", "docx", "csv", "xlsx"],
            key="gen_chat_file",
        )

    user_query = st.chat_input("Ask anything, request a math derivation, or submit a problem...")

    if user_query:
        if not st.session_state.get("saved_openrouter_key"):
            st.error("Please enter an OpenRouter API key in the sidebar first!")
        else:
            client = get_openrouter_client(st.session_state.saved_openrouter_key)
            file_context = (
                process_uploaded_file(attached_file) if attached_file else ""
            )
            full_prompt = (
                f"{user_query}\n{file_context}" if file_context else user_query
            )

            chat_history.append({"role": "user", "content": full_prompt})
            st.session_state.db["chats"][st.session_state.active_chat] = (
                chat_history
            )
            save_data(st.session_state.db)

            with st.chat_message("user"):
                st.markdown(full_prompt)

            with st.chat_message("assistant"):
                with st.status("🧠 AI is thinking...", expanded=True) as status:
                    response_placeholder = st.empty()
                    full_response = ""

                    api_messages = [MATH_SYSTEM_PROMPT] + [
                        {"role": m["role"], "content": m["content"]}
                        for m in chat_history[-6:]
                    ]

                    try:
                        response = client.chat.completions.create(
                            model=selected_model_slug, messages=api_messages, stream=True
                        )

                        for chunk in response:
                            if hasattr(chunk, "choices") and chunk.choices:
                                choice = chunk.choices[0]
                                if hasattr(choice, "delta") and choice.delta:
                                    content = getattr(choice.delta, "content", None)
                                    if content:
                                        full_response += content
                                        response_placeholder.markdown(
                                            fix_latex_formatting(full_response) + " ▌"
                                        )

                        cleaned_final = fix_latex_formatting(full_response)
                        if not cleaned_final.strip():
                            cleaned_final = "*(Model returned empty response. Please try another model.)*"

                        response_placeholder.markdown(cleaned_final)
                        status.update(label="✅ Response Done!", state="complete")

                        chat_history.append({"role": "assistant", "content": cleaned_final})
                        st.session_state.db["chats"][st.session_state.active_chat] = (
                            chat_history
                        )
                        save_data(st.session_state.db)
                        st.rerun()

                    except Exception as e:
                        status.update(label="❌ Error occurred", state="error")
                        st.error(f"Execution Error: {str(e)}")

# ==========================================
# VIEW 2: NOTEBOOK WORKSPACES
# ==========================================
elif st.session_state.active_mode == "notebook_studio":
    st.title("📚 Notebook Workspaces")

    selected_label = st.selectbox(
        "🌐 Workspace Model (Vision & Document Capable)",
        options=list(MODEL_OPTIONS.keys()),
        index=0,
        key="notebook_model_select",
    )
    selected_model_slug = MODEL_OPTIONS[selected_label]

    workspace_type = st.radio(
        "Select Workspace Mode",
        ["📝 MCQ Workspace", "✍️ Focus Written Workspace"],
        horizontal=True,
    )

    # --------------------------------------------------------------------------
    # WORKSPACE 1: MCQ WORKSPACE
    # --------------------------------------------------------------------------
    if workspace_type == "📝 MCQ Workspace":
        with st.expander("⏱️ MCQ Practice Exam Timer", expanded=False):
            col_mcq_t1, col_mcq_t2 = st.columns([2, 1])
            mcq_mins = col_mcq_t1.number_input("Set Timer (Minutes)", min_value=1, max_value=180, value=15, step=5, key="mcq_timer_input")
            if col_mcq_t2.button("🚀 Start / Reset Timer", key="btn_mcq_timer"):
                st.session_state.mcq_timer_end = datetime.now().timestamp() + (mcq_mins * 60)
                st.success(f"Timer set for {mcq_mins} minutes!")

            if "mcq_timer_end" in st.session_state:
                remaining_sec = int(st.session_state.mcq_timer_end - datetime.now().timestamp())
                if remaining_sec > 0:
                    mins, secs = divmod(remaining_sec, 60)
                    st.info(f"⏳ **Time Remaining**: {mins:02d}m {secs:02d}s")
                else:
                    st.error("⏰ **Time is up! Submit your answers now.**")

        st.divider()

        col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
        mcq_subs = list(st.session_state.db["mcq_subjects"].keys())
        selected_mcq_sub = col_s1.selectbox(
            "Select Subject", mcq_subs if mcq_subs else ["None"]
        )

        new_mcq_sub = col_s2.text_input("New MCQ Subject")
        if col_s2.button("Add Subject", key="btn_add_mcq_sub") and new_mcq_sub:
            st.session_state.db["mcq_subjects"][new_mcq_sub] = {
                "sources": [],
                "chat": [],
                "mistakes": [],
                "instructions": "",
            }
            save_data(st.session_state.db)
            st.rerun()

        # Added Delete Subject functionality
        if selected_mcq_sub and selected_mcq_sub != "None":
            if col_s3.button("🗑️ Delete Subject", key="btn_del_mcq_sub"):
                del st.session_state.db["mcq_subjects"][selected_mcq_sub]
                save_data(st.session_state.db)
                st.success(f"Deleted subject '{selected_mcq_sub}'!")
                st.rerun()

            sub_data = st.session_state.db["mcq_subjects"][selected_mcq_sub]
            m_tab1, m_tab2 = st.tabs(["🎯 Quiz Generator & Practice", "📖 Mistakes Register & Notes"])

            with m_tab1:
                col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
                num_q = col_cfg1.number_input("Number of Questions", 1, 20, 5)
                diff = col_cfg2.selectbox("Difficulty", ["Medium", "Hard", "Advanced Exam"])
                lang_choice = col_cfg3.selectbox("Language Option", ["English", "Bangla (বাংলা)", "Bilingual (English + Bangla)"])

                custom_instructions = st.text_area(
                    "Chapter / Custom Prompt Instructions for MCQ Generation",
                    value=sub_data.get("instructions", "Focus on core chapter concepts and past weak points. Avoid repeat questions."),
                    height=90,
                    key="mcq_inst_input"
                )
                if st.button("💾 Save Instructions"):
                    sub_data["instructions"] = custom_instructions
                    save_data(st.session_state.db)
                    st.success("Instructions saved!")

                mcq_file = st.file_uploader(
                    "Attach Study Material (PDF, DOCX, CSV, Images, Handwritten Notes)",
                    type=["pdf", "docx", "csv", "xlsx", "png", "jpg", "jpeg"],
                    key="mcq_file_up",
                )

                if st.button("🚀 Generate Chapter Quiz"):
                    if not st.session_state.get("saved_openrouter_key"):
                        st.error("Please enter your API Key in the sidebar.")
                    else:
                        client = get_openrouter_client(
                            st.session_state.saved_openrouter_key
                        )
                        file_content = process_uploaded_file(mcq_file) if mcq_file else ""
                        prior_mistakes = [
                            f"- {m.get('concept', '')}: {m.get('takeaway', '')}"
                            for m in sub_data.get("mistakes", [])
                        ]
                        weakness_context = (
                            "\n".join(prior_mistakes) if prior_mistakes else "None recorded yet."
                        )

                        prompt = f"""Language Requirement: {lang_choice}
Subject: {selected_mcq_sub}
Difficulty Level: {diff}
Target Questions Count: {num_q}

CUSTOM INSTRUCTIONS / CHAPTER FOCUS:
{custom_instructions}

LOGGED PAST MISTAKES & WEAKNESSES:
{weakness_context}

ATTACHED SOURCE MATERIALS / NOTES:
{file_content}

GENERATE A QUIZ FOLLOWING THESE RULES:
1. Target the chapter concepts in source materials and past logged weaknesses.
2. DO NOT repeat exact duplicate questions from past mistakes; create NEW variations or deeper questions testing those concepts.
3. Use proper LaTeX notation ($...$ inline, $$...$$ block) for math.
4. Output STRICT raw JSON array format without markdown backticks:
[
  {{"id": 1, "question": "...", "options": ["A) ...", "B) ...", "C) ...", "D) ..."], "correct": "A) ...", "explanation": "..."}}
]
"""

                        with st.status("🧠 Generating Targeted Quiz...", expanded=True):
                            try:
                                res = client.chat.completions.create(
                                    model=selected_model_slug,
                                    messages=[
                                        MATH_SYSTEM_PROMPT,
                                        {"role": "user", "content": prompt},
                                    ],
                                )
                                raw_json = clean_json_response(res.choices[0].message.content)
                                st.session_state.quiz = json.loads(raw_json)
                            except Exception as qe:
                                st.error(f"Quiz Generation Error: {str(qe)}")

                if "quiz" in st.session_state:
                    st.divider()
                    user_ans = {}
                    with st.form("mcq_form"):
                        for q in st.session_state.quiz:
                            st.write(f"**Q{q['id']}: {fix_latex_formatting(q['question'])}**")
                            user_ans[q["id"]] = st.radio(
                                f"Select Option for Q{q['id']}", q["options"], key=f"q_{q['id']}"
                            )
                            st.markdown("---")
                        submit_m = st.form_submit_button("Submit Quiz Answers")

                    if submit_m:
                        score = 0
                        existing_concepts = [
                            m["concept"] for m in sub_data.get("mistakes", [])
                        ]

                        for q in st.session_state.quiz:
                            if user_ans[q["id"]] == q["correct"]:
                                score += 1
                                st.success(f"Q{q['id']} Correct! 🎉")
                            else:
                                st.error(f"Q{q['id']} Incorrect. Correct: {q['correct']}")
                                st.info(f"💡 Explanation: {fix_latex_formatting(q['explanation'])}")

                                if q["question"] not in existing_concepts:
                                    sub_data["mistakes"].append({
                                        "date": datetime.now().strftime("%Y-%m-%d"),
                                        "concept": q["question"],
                                        "takeaway": q["explanation"],
                                    })
                                else:
                                    for ex_m in sub_data["mistakes"]:
                                        if ex_m["concept"] == q["question"]:
                                            ex_m["date"] = datetime.now().strftime("%Y-%m-%d")

                        st.metric("Final Quiz Score", f"{score} / {len(st.session_state.quiz)}")
                        save_data(st.session_state.db)

            with m_tab2:
                st.subheader(f"Logged Mistakes & Revision Notes ({selected_mcq_sub})")

                with st.expander("➕ Manually Log Weak Spot / Concept"):
                    manual_concept = st.text_input("Concept / Question")
                    manual_takeaway = st.text_area("Correct Explanation / Key Takeaway")
                    if st.button("Save Weak Spot"):
                        if manual_concept:
                            sub_data["mistakes"].append({
                                "date": datetime.now().strftime("%Y-%m-%d"),
                                "concept": manual_concept,
                                "takeaway": manual_takeaway
                            })
                            save_data(st.session_state.db)
                            st.success("Logged into revision register!")
                            st.rerun()

                st.markdown("---")
                for idx, m in enumerate(sub_data.get("mistakes", [])):
                    with st.expander(f"#{idx+1} [{m.get('date', 'N/A')}] {m['concept'][:60]}..."):
                        st.markdown(f"**Question / Concept:**\n{fix_latex_formatting(m['concept'])}")
                        st.markdown(f"**Explanation / Revision Takeaway:**\n{fix_latex_formatting(m['takeaway'])}")

                st.divider()
                if st.button("Export MCQ Revision Guide (.docx)"):
                    fpath = generate_docx(
                        selected_mcq_sub, sub_data["mistakes"], "MCQ"
                    )
                    with open(fpath, "rb") as fp:
                        st.download_button("📥 Download Word Revision Guide", fp, file_name=fpath)

    # --------------------------------------------------------------------------
    # WORKSPACE 2: FOCUS WRITTEN WORKSPACE
    # --------------------------------------------------------------------------
    elif workspace_type == "✍️ Focus Written Workspace":
        with st.expander("⏱️ Written Exam Timer", expanded=False):
            col_w_t1, col_w_t2 = st.columns([2, 1])
            written_mins = col_w_t1.number_input("Set Timer (Minutes)", min_value=1, max_value=180, value=30, step=5, key="written_timer_input")
            if col_w_t2.button("🚀 Start / Reset Timer", key="btn_written_timer"):
                st.session_state.written_timer_end = datetime.now().timestamp() + (written_mins * 60)
                st.success(f"Timer set for {written_mins} minutes!")

            if "written_timer_end" in st.session_state:
                remaining_sec = int(st.session_state.written_timer_end - datetime.now().timestamp())
                if remaining_sec > 0:
                    mins, secs = divmod(remaining_sec, 60)
                    st.info(f"⏳ **Time Remaining**: {mins:02d}m {secs:02d}s")
                else:
                    st.error("⏰ **Time is up! Submit your answer now.**")

        st.divider()

        col_w1, col_w2, col_w3 = st.columns([2, 1, 1])
        w_subs = list(st.session_state.db["written_subjects"].keys())
        selected_w_sub = col_w1.selectbox(
            "Select Subject", w_subs if w_subs else ["None"]
        )

        new_w_sub = col_w2.text_input("New Written Subject")
        if col_w2.button("Add Subject", key="btn_add_w_sub") and new_w_sub:
            st.session_state.db["written_subjects"][new_w_sub] = {
                "sources": [],
                "chat": [],
                "mistakes": [],
                "instructions": "Focus on clarity, analytical depth, and structural coherence.",
            }
            save_data(st.session_state.db)
            st.rerun()

        # Added Delete Subject functionality for Written Workspace
        if selected_w_sub and selected_w_sub != "None":
            if col_w3.button("🗑️ Delete Subject", key="btn_del_w_sub"):
                del st.session_state.db["written_subjects"][selected_w_sub]
                save_data(st.session_state.db)
                st.success(f"Deleted subject '{selected_w_sub}'!")
                st.rerun()

            w_sub_data = st.session_state.db["written_subjects"][selected_w_sub]

            col_lang1, col_lang2 = st.columns([1, 2])
            lang_w_choice = col_lang1.selectbox("Evaluation Language", ["English", "Bangla (বাংলা)", "Bilingual (English + Bangla)"])

            custom_instructions = st.text_area(
                "Evaluation Rubric / Specific Criteria Instructions",
                value=w_sub_data.get("instructions", "Focus on clarity, analytical depth, logic, and formula accuracy."),
                height=90,
            )
            if st.button("💾 Save Rubric Criteria"):
                w_sub_data["instructions"] = custom_instructions
                save_data(st.session_state.db)
                st.success("Rubric criteria saved!")

            written_file = st.file_uploader(
                "Attach Context Material (PDF, DOCX, Images, Notes)",
                type=["pdf", "docx", "png", "jpg", "jpeg"],
                key="written_file_up",
            )
            essay_input = st.text_area("Write or Paste Solution / Essay Submission", height=220)

            if st.button("🔍 Evaluate Submission"):
                if not st.session_state.get("saved_openrouter_key"):
                    st.error("Please enter your API Key in the sidebar.")
                elif not essay_input.strip():
                    st.warning("Please enter your written solution or essay before running evaluation.")
                else:
                    client = get_openrouter_client(st.session_state.saved_openrouter_key)
                    file_text = (
                        process_uploaded_file(written_file) if written_file else ""
                    )
                    prior_mistakes = [
                        f"- {m.get('area', '')}: {m.get('correction', '')}"
                        for m in w_sub_data.get("mistakes", [])
                    ]
                    weakness_context = (
                        "\n".join(prior_mistakes) if prior_mistakes else "None recorded yet."
                    )

                    combined_text = (
                        f"Student Submission:\n{essay_input}\n\nAttached Material:\n{file_text}"
                    )

                    prompt = f"""Language Requirement: {lang_w_choice}
Subject: {selected_w_sub}
Evaluation Criteria: {custom_instructions}

PAST LOGGED MISTAKES FOR THIS STUDENT:
{weakness_context}

SUBMISSION & CONTENT TO EVALUATE:
{combined_text}

Provide detailed feedback, point out errors/mistakes, and output STRICT JSON format:
{{
  "score": "85%",
  "weakness": "Key area needing improvement...",
  "strategy": "Actionable takeaway to fix mistake..."
}}
"""

                    with st.status("🧠 Evaluating Written Solution...", expanded=True):
                        try:
                            res = client.chat.completions.create(
                                model=selected_model_slug,
                                messages=[
                                    MATH_SYSTEM_PROMPT,
                                    {"role": "user", "content": prompt},
                                ],
                            )
                            raw_json = clean_json_response(res.choices[0].message.content)
                            eval_data = json.loads(raw_json)

                            st.metric("Evaluation Score", eval_data["score"])
                            st.info(f"**Identified Weakness / Mistake**: {fix_latex_formatting(eval_data['weakness'])}")
                            st.success(f"**Improvement Strategy**: {fix_latex_formatting(eval_data['strategy'])}")

                            w_sub_data["mistakes"].append({
                                "date": datetime.now().strftime("%Y-%m-%d"),
                                "area": eval_data["weakness"],
                                "correction": eval_data["strategy"],
                            })
                            save_data(st.session_state.db)
                        except Exception as we:
                            st.error(f"Evaluation Error: {str(we)}")

            st.divider()
            st.subheader(f"Logged Written Feedback Register ({selected_w_sub})")
            for idx, item in enumerate(w_sub_data.get("mistakes", [])):
                with st.expander(f"#{idx+1} [{item.get('date', 'N/A')}] {item.get('area', 'Feedback')[:60]}..."):
                    st.markdown(f"**Weakness Area:**\n{fix_latex_formatting(item.get('area', ''))}")
                    st.markdown(f"**Correction Strategy:**\n{fix_latex_formatting(item.get('correction', ''))}")

            st.divider()
            if st.button("Export Written Revision Guide (.docx)"):
                fpath = generate_docx(
                    selected_w_sub, w_sub_data["mistakes"], "Written"
                )
                with open(fpath, "rb") as fp:
                    st.download_button("📥 Download Revision Guide", fp, file_name=fpath)
