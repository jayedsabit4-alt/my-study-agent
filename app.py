import base64
import io
import json
import os
import re
from datetime import datetime
import pandas as pd
from PIL import Image
import streamlit as st

st.set_page_config(
    page_title="Agentic Study Platform", page_icon="🎓", layout="wide"
)

DATA_FILE = "study_data.json"

MODEL_OPTIONS = {
    "google/gemma-4-31b-it:free (Vision & Document OCR)": (
        "google/gemma-4-31b-it:free"
    ),
    "openrouter/free (Auto-Router - Multimodal & Fast)": "openrouter/free",
    "nvidia/nemotron-3-ultra-550b-a55b:free (Deep Reasoning)": (
        "nvidia/nemotron-3-ultra-550b-a55b:free"
    ),
    "openai/gpt-oss-20b:free (Lightweight Chat)": "openai/gpt-oss-20b:free",
}

MATH_SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are an expert academic assistant. When rendering formatting and mathematical notation:\n"
        "1. ALWAYS place markdown headers (e.g., ### Section) on separate lines with BLANK LINES before and after.\n"
        "2. Put standalone display math equations inside double dollar signs on their OWN separate lines:\n"
        "$$\n"
        "f(x) = \\dots\n"
        "$$\n"
        "3. Wrap inline math variables/symbols in single dollar signs (e.g., $x = 5$).\n"
        "4. Render tables using valid Markdown table format with proper alignment rows (|---|---|\n"
        "5. NEVER output orphan structural tags like \\end{cases} without a matching \\begin{cases}."
    ),
}


# --- ADVANCED LATEX & MARKDOWN FORMAT SANITIZER ---
def fix_latex_formatting(text: str) -> str:
    """Cleans raw LLM outputs, separates squished headers, and fixes KaTeX & Table rendering bugs."""
    if not text:
        return ""

    # 1. Force newline spacing around markdown headers
    text = re.sub(r"(?<!\n)(###?\s+)", r"\n\n\1", text)
    text = re.sub(r"(\\end\{[a-zA-Z]+\})\s*(###?|---)", r"\1\n\n\2", text)

    # 2. Fix standalone horizontal rules WITHOUT breaking markdown tables (|---|---|)
    text = re.sub(r"(?<![|\w\n])\n?^\s*---\s*$(?![|\w])", r"\n\n---\n\n", text, flags=re.MULTILINE)

    # 3. Convert bracket notations [ ... ] and \( ... \) to standard dollar signs
    text = re.sub(r"\\\[\s*(.*?)\s*\\\]", r"\n$$\1$$\n", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\w)\[\s*(\\.*?)\s*\]", r"\n$$\1$$\n", text, flags=re.DOTALL)
    text = re.sub(r"\\\(\s*(.*?)\s*\\\)", r"$\1$", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\w)\(\s*(\\.*?)\s*\)", r"$\1$", text, flags=re.DOTALL)

    # 4. Fix nested or double $$ tags around alignment environments
    text = re.sub(
        r"\$\$\s*(\\begin\{(aligned|equation|gather|alignat|matrix|bmatrix|cases|array)\})",
        r"\n$$\n\1",
        text,
    )
    text = re.sub(
        r"(\\end\{(aligned|equation|gather|alignat|matrix|bmatrix|cases|array)\})\s*\$\$",
        r"\1\n$$\n",
        text,
    )

    # 5. Clean orphan \end{cases} tags missing their starting \begin{cases}
    if "\\end{cases}" in text and "\\begin{cases}" not in text:
        text = text.replace("\\end{cases}", "")

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
        base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=45.0
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


# --- FAST IMAGE COMPRESSION HELPER ---
def compress_image_to_b64(image_bytes, max_dim=1000, quality=75):
    """Resizes and compresses images to JPEG before base64 encoding."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.thumbnail((max_dim, max_dim))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"
    except Exception:
        b64_str = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"


# --- HIGH-QUALITY TABLE & TEXT PDF EXTRACTOR ---
def extract_file_data(uploaded_file):
    """Extracts structured text and markdown tables with multi-engine PDF parsing."""
    file_type = uploaded_file.name.split(".")[-1].lower()
    extracted_text = ""
    image_url = None

    try:
        if file_type in ["png", "jpg", "jpeg"]:
            image_url = compress_image_to_b64(uploaded_file.getvalue())
            extracted_text = f"[Image Document: {uploaded_file.name}]"

        elif file_type == "pdf":
            pdf_text = []
            
            # Primary Engine: Try pdfplumber for table and layout extraction
            try:
                import pdfplumber
                with pdfplumber.open(io.BytesIO(uploaded_file.getvalue())) as pdf:
                    for page_idx, page in enumerate(pdf.pages):
                        if page_idx >= 50:
                            pdf_text.append("\n[Truncated remaining pages for performance]")
                            break
                        
                        page_content = f"--- Page {page_idx + 1} ---\n"
                        
                        # 1. Extract tables as Markdown
                        tables = page.extract_tables()
                        md_tables = ""
                        if tables:
                            for tbl in tables:
                                clean_rows = [[str(cell or '').strip().replace('\n', ' ') for cell in row] for row in tbl if any(row)]
                                if len(clean_rows) > 1:
                                    header = clean_rows[0]
                                    md_tables += "\n| " + " | ".join(header) + " |\n"
                                    md_tables += "| " + " | ".join(["---"] * len(header)) + " |\n"
                                    for row in clean_rows[1:]:
                                        row_padded = row + [""] * (len(header) - len(row))
                                        md_tables += "| " + " | ".join(row_padded[:len(header)]) + " |\n"
                                    md_tables += "\n"

                        # 2. Extract standard text
                        raw_text = page.extract_text() or ""
                        if raw_text:
                            page_content += raw_text + "\n"
                        if md_tables:
                            page_content += "\n**Extracted Tables:**\n" + md_tables
                            
                        if page_content.strip():
                            pdf_text.append(page_content)

            except Exception:
                # Fallback Engine: pypdf if pdfplumber is unavailable or fails
                from pypdf import PdfReader
                pdf_stream = io.BytesIO(uploaded_file.getvalue())
                reader = PdfReader(pdf_stream, strict=False)

                for page_idx, page in enumerate(reader.pages):
                    if page_idx >= 50:
                        pdf_text.append("\n[Truncated remaining pages for performance]")
                        break
                    try:
                        txt = page.extract_text()
                        if txt:
                            pdf_text.append(f"--- Page {page_idx + 1} ---\n" + txt)
                    except Exception:
                        continue

            extracted_text = (
                "\n\n".join(pdf_text)
                if pdf_text
                else "[Scanned/Handwritten or unreadable PDF Content]"
            )

        elif file_type == "docx":
            from docx import Document

            doc = Document(io.BytesIO(uploaded_file.getvalue()))
            extracted_text = "\n".join([p.text for p in doc.paragraphs])

        elif file_type in ["csv", "xlsx"]:
            df = (
                pd.read_csv(uploaded_file)
                if file_type == "csv"
                else pd.read_excel(uploaded_file)
            )
            extracted_text = df.head(100).to_string(index=False)

        else:
            extracted_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")

    except Exception as e:
        extracted_text = f"Error extracting {uploaded_file.name}: {str(e)}"

    return {
        "name": uploaded_file.name,
        "type": file_type,
        "text": extracted_text,
        "image_url": image_url,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


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
            if msg.get("file_names"):
                st.caption(f"📎 Attached Context: {', '.join(msg['file_names'])}")
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
        attached_files = st.file_uploader(
            "Attach Context Sources (Multiple PDFs, DOCX, CSV, Images)",
            type=["png", "jpg", "jpeg", "pdf", "docx", "csv", "xlsx"],
            accept_multiple_files=True,
            key="gen_chat_files",
        )

    user_query = st.chat_input("Ask anything, request a math derivation, or submit a problem...")

    if user_query:
        if not st.session_state.get("saved_openrouter_key"):
            st.error("Please enter an OpenRouter API key in the sidebar first!")
        else:
            client = get_openrouter_client(st.session_state.saved_openrouter_key)

            temp_file_text = ""
            image_urls = []
            file_names_list = []
            if attached_files:
                for f in attached_files:
                    f_data = extract_file_data(f)
                    file_names_list.append(f_data['name'])
                    temp_file_text += f"\n\n--- Attached File Context: {f_data['name']} ---\n{f_data['text']}"
                    if f_data.get("image_url"):
                        image_urls.append(f_data["image_url"])

            # Store ONLY user query and file badge in display content
            user_msg_record = {
                "role": "user",
                "content": user_query,
                "context": temp_file_text,
                "file_names": file_names_list
            }

            chat_history.append(user_msg_record)
            st.session_state.db["chats"][st.session_state.active_chat] = chat_history
            save_data(st.session_state.db)

            with st.chat_message("user"):
                if file_names_list:
                    st.caption(f"📎 Attached Context: {', '.join(file_names_list)}")
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.status("🧠 AI is thinking...", expanded=True) as status:
                    response_placeholder = st.empty()
                    full_response = ""

                    # Build API context without cluttering the UI
                    api_messages = [MATH_SYSTEM_PROMPT]
                    for m in chat_history[-6:-1]:
                        m_text = f"{m['content']}\n\n{m.get('context', '')}".strip()
                        api_messages.append({"role": m["role"], "content": m_text})

                    latest_prompt = f"{user_query}\n\n{temp_file_text}".strip()
                    if image_urls:
                        user_content = [{"type": "text", "text": latest_prompt}]
                        for img_url in image_urls[:2]:
                            user_content.append({"type": "image_url", "image_url": {"url": img_url}})
                        api_messages.append({"role": "user", "content": user_content})
                    else:
                        api_messages.append({"role": "user", "content": latest_prompt})

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
                        st.session_state.db["chats"][st.session_state.active_chat] = chat_history
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
            "Select Subject / Notebook", mcq_subs if mcq_subs else ["None"]
        )

        new_mcq_sub = col_s2.text_input("New MCQ Notebook")
        if col_s2.button("Add Notebook", key="btn_add_mcq_sub") and new_mcq_sub:
            st.session_state.db["mcq_subjects"][new_mcq_sub] = {
                "sources": [],
                "chat": [],
                "mistakes": [],
                "instructions": "",
            }
            save_data(st.session_state.db)
            st.rerun()

        if selected_mcq_sub and selected_mcq_sub != "None":
            if col_s3.button("🗑️ Delete Notebook", key="btn_del_mcq_sub"):
                del st.session_state.db["mcq_subjects"][selected_mcq_sub]
                save_data(st.session_state.db)
                st.success(f"Deleted notebook '{selected_mcq_sub}'!")
                st.rerun()

            sub_data = st.session_state.db["mcq_subjects"][selected_mcq_sub]
            if "sources" not in sub_data:
                sub_data["sources"] = []

            # --- NOTEBOOK LM SOURCE MANAGER PANEL ---
            with st.expander(f"📚 Managed Persistent Sources for Notebook: '{selected_mcq_sub}' ({len(sub_data['sources'])} Saved)", expanded=True):
                st.markdown("Upload files here once. They will stay saved in this notebook across all refreshes.")
                new_mcq_files = st.file_uploader(
                    "Add Sources to Notebook (PDF, DOCX, CSV, Images)",
                    type=["pdf", "docx", "csv", "xlsx", "png", "jpg", "jpeg"],
                    accept_multiple_files=True,
                    key=f"mcq_sources_up_{selected_mcq_sub}",
                )

                if st.button("💾 Save Files to Notebook Sources", key=f"btn_save_mcq_sources_{selected_mcq_sub}"):
                    if new_mcq_files:
                        for nf in new_mcq_files:
                            extracted_obj = extract_file_data(nf)
                            sub_data["sources"].append(extracted_obj)
                        save_data(st.session_state.db)
                        st.success("Successfully saved sources to this notebook!")
                        st.rerun()

                if sub_data["sources"]:
                    st.write("---")
                    st.write("**Current Saved Sources in this Notebook:**")
                    for s_idx, src in enumerate(sub_data["sources"]):
                        sc1, sc2 = st.columns([5, 1])
                        sc1.markdown(f"📄 **{src['name']}** *(Added {src['date']})*")
                        if sc2.button("🗑️ Delete", key=f"del_mcq_src_{selected_mcq_sub}_{s_idx}"):
                            sub_data["sources"].pop(s_idx)
                            save_data(st.session_state.db)
                            st.rerun()

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

                if st.button("🚀 Generate Chapter Quiz"):
                    if not st.session_state.get("saved_openrouter_key"):
                        st.error("Please enter your API Key in the sidebar.")
                    else:
                        client = get_openrouter_client(
                            st.session_state.saved_openrouter_key
                        )

                        saved_sources_text = ""
                        image_urls = []
                        for src_item in sub_data.get("sources", []):
                            saved_sources_text += f"\n\n--- Source File: {src_item['name']} ---\n{src_item['text']}"
                            if src_item.get("image_url"):
                                image_urls.append(src_item["image_url"])

                        prior_mistakes = [
                            f"- {m.get('concept', '')}: {m.get('takeaway', '')}"
                            for m in sub_data.get("mistakes", [])
                        ]
                        weakness_context = (
                            "\n".join(prior_mistakes) if prior_mistakes else "None recorded yet."
                        )

                        prompt_text = f"""Language Requirement: {lang_choice}
Subject Notebook: {selected_mcq_sub}
Difficulty Level: {diff}
Target Questions Count: {num_q}

CUSTOM INSTRUCTIONS / CHAPTER FOCUS:
{custom_instructions}

LOGGED PAST MISTAKES & WEAKNESSES:
{weakness_context}

SAVED NOTEBOOK SOURCES & CONTEXT:
{saved_sources_text if saved_sources_text else "No persistent sources uploaded yet. Generate based on subject domain."}

GENERATE A QUIZ FOLLOWING THESE RULES:
1. Target the chapter concepts in saved source materials and past logged weaknesses.
2. If handwritten images or scanned notes are attached, perform OCR and generate questions directly from the text.
3. DO NOT repeat exact duplicate questions from past mistakes; create NEW variations or deeper questions testing those concepts.
4. Use proper LaTeX notation ($...$ inline, $$...$$ block) for math and valid Markdown tables.
5. Output STRICT raw JSON array format without markdown backticks:
[
  {{"id": 1, "question": "...", "options": ["A) ...", "B) ...", "C) ...", "D) ..."], "correct": "A) ...", "explanation": "..."}}
]
"""

                        if image_urls:
                            user_msg_content = [{"type": "text", "text": prompt_text}]
                            for img_url in image_urls[:2]:
                                user_msg_content.append({"type": "image_url", "image_url": {"url": img_url}})
                        else:
                            user_msg_content = prompt_text

                        with st.status("🧠 Generating Targeted Quiz from Notebook Sources...", expanded=True):
                            try:
                                res = client.chat.completions.create(
                                    model=selected_model_slug,
                                    messages=[
                                        MATH_SYSTEM_PROMPT,
                                        {"role": "user", "content": user_msg_content},
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
            "Select Subject / Notebook", w_subs if w_subs else ["None"]
        )

        new_w_sub = col_w2.text_input("New Written Notebook")
        if col_w2.button("Add Notebook", key="btn_add_w_sub") and new_w_sub:
            st.session_state.db["written_subjects"][new_w_sub] = {
                "sources": [],
                "chat": [],
                "mistakes": [],
                "instructions": "Focus on clarity, analytical depth, and structural coherence.",
            }
            save_data(st.session_state.db)
            st.rerun()

        if selected_w_sub and selected_w_sub != "None":
            if col_w3.button("🗑️ Delete Notebook", key="btn_del_w_sub"):
                del st.session_state.db["written_subjects"][selected_w_sub]
                save_data(st.session_state.db)
                st.success(f"Deleted notebook '{selected_w_sub}'!")
                st.rerun()

            w_sub_data = st.session_state.db["written_subjects"][selected_w_sub]
            if "sources" not in w_sub_data:
                w_sub_data["sources"] = []

            # --- NOTEBOOK LM SOURCE MANAGER PANEL ---
            with st.expander(f"📚 Managed Persistent Sources for Notebook: '{selected_w_sub}' ({len(w_sub_data['sources'])} Saved)", expanded=True):
                st.markdown("Upload files here once. They will stay saved in this notebook across all refreshes.")
                new_w_files = st.file_uploader(
                    "Add Sources to Notebook (PDF, DOCX, Images)",
                    type=["pdf", "docx", "png", "jpg", "jpeg"],
                    accept_multiple_files=True,
                    key=f"written_sources_up_{selected_w_sub}",
                )

                if st.button("💾 Save Files to Notebook Sources", key=f"btn_save_w_sources_{selected_w_sub}"):
                    if new_w_files:
                        for nf in new_w_files:
                            extracted_obj = extract_file_data(nf)
                            w_sub_data["sources"].append(extracted_obj)
                        save_data(st.session_state.db)
                        st.success("Successfully saved sources to this notebook!")
                        st.rerun()

                if w_sub_data["sources"]:
                    st.write("---")
                    st.write("**Current Saved Sources in this Notebook:**")
                    for s_idx, src in enumerate(w_sub_data["sources"]):
                        sc1, sc2 = st.columns([5, 1])
                        sc1.markdown(f"📄 **{src['name']}** *(Added {src['date']})*")
                        if sc2.button("🗑️ Delete", key=f"del_w_src_{selected_w_sub}_{s_idx}"):
                            w_sub_data["sources"].pop(s_idx)
                            save_data(st.session_state.db)
                            st.rerun()

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

            essay_input = st.text_area("Write or Paste Solution / Essay Submission", height=220)

            if st.button("🔍 Evaluate Submission"):
                if not st.session_state.get("saved_openrouter_key"):
                    st.error("Please enter your API Key in the sidebar.")
                elif not essay_input.strip() and not w_sub_data["sources"]:
                    st.warning("Please enter a written solution or ensure you have saved notebook sources.")
                else:
                    client = get_openrouter_client(st.session_state.saved_openrouter_key)

                    saved_sources_text = ""
                    image_urls = []
                    for src_item in w_sub_data.get("sources", []):
                        saved_sources_text += f"\n\n--- Source File: {src_item['name']} ---\n{src_item['text']}"
                        if src_item.get("image_url"):
                            image_urls.append(src_item["image_url"])

                    prior_mistakes = [
                        f"- {m.get('area', '')}: {m.get('correction', '')}"
                        for m in w_sub_data.get("mistakes", [])
                    ]
                    weakness_context = (
                        "\n".join(prior_mistakes) if prior_mistakes else "None recorded yet."
                    )

                    combined_text = (
                        f"Student Submission:\n{essay_input}\n\nSAVED NOTEBOOK SOURCES:\n{saved_sources_text}"
                    )

                    prompt_text = f"""Language Requirement: {lang_w_choice}
Subject Notebook: {selected_w_sub}
Evaluation Criteria: {custom_instructions}

PAST LOGGED MISTAKES FOR THIS STUDENT:
{weakness_context}

SUBMISSION & CONTEXT TO EVALUATE:
{combined_text}

Provide detailed feedback, point out errors/mistakes, and output STRICT JSON format:
{{
  "score": "85%",
  "weakness": "Key area needing improvement...",
  "strategy": "Actionable takeaway to fix mistake..."
}}
"""

                    if image_urls:
                        user_msg_content = [{"type": "text", "text": prompt_text}]
                        for img_url in image_urls[:2]:
                            user_msg_content.append({"type": "image_url", "image_url": {"url": img_url}})
                    else:
                        user_msg_content = prompt_text

                    with st.status("🧠 Evaluating Solution against Notebook Sources...", expanded=True):
                        try:
                            res = client.chat.completions.create(
                                model=selected_model_slug,
                                messages=[
                                    MATH_SYSTEM_PROMPT,
                                    {"role": "user", "content": user_msg_content},
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
