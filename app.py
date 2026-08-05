from datetime import datetime
import json
import os
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Agentic Study Platform", page_icon="🎓", layout="wide"
)

DATA_FILE = "study_data.json"

MODEL_OPTIONS = {
    "openrouter/free (Auto-Router - Fast & Free)": "openrouter/free",
    "openai/gpt-oss-20b:free (Lightweight Chat)": "openai/gpt-oss-20b:free",
    "google/gemma-4-31b-it:free (Document & Vision)": (
        "google/gemma-4-31b-it:free"
    ),
    "nvidia/nemotron-3-ultra-550b-a55b:free (Deep Reasoning)": (
        "nvidia/nemotron-3-ultra-550b-a55b:free"
    ),
}

MATH_SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "When rendering mathematical variables, formulas, or statistical"
        " notation, ALWAYS wrap them in standard LaTeX dollar signs. Use"
        " single dollar signs for inline math (e.g., $\\mu$, $\\bar{x}$,"
        " $\\theta$) and double dollar signs for standalone equations ($$...$$)."
    ),
}


# --- LAZY CLIENT INITIALIZATION (Prevents Startup Hang) ---
def get_openrouter_client(api_key):
  from openai import OpenAI

  return OpenAI(
      base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=20.0
  )


# --- DATA PERSISTENCE ---
def load_data():
  if os.path.exists(DATA_FILE):
    try:
      with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
    except Exception:
      st.warning("⚠️ Warning: Could not read saved data. Initializing new state.")
  return {
      "chats": {"Default Chat": []},
      "mcq_subjects": {
          "General Math": {"sources": [], "chat": [], "mistakes": []}
      },
      "written_subjects": {
          "Academic Essay": {"sources": [], "chat": [], "mistakes": []}
      },
  }


def save_data(data):
  try:
    with open(DATA_FILE, "w", encoding="utf-8") as f:
      json.dump(data, f, ensure_ascii=False, indent=2)
  except Exception as e:
    st.error(f"Failed to save data: {e}")


# --- STATE INITIALIZATION (No Top-Level Reruns) ---
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
      extracted_text += f"[Image File: {uploaded_file.name}]"
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
        f"Takeaway: {item.get('takeaway', item.get('correction', 'N/A'))}"
    )
  filename = f"{subject_name}_{section_type}_Revision.docx"
  doc.save(filename)
  return filename


# --- SIDEBAR NAV ---
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

  selected_label = st.selectbox(
      "🌐 AI Model",
      options=list(MODEL_OPTIONS.keys()),
      index=0,
      key="chat_model_select",
  )
  selected_model_slug = MODEL_OPTIONS[selected_label]

  chat_history = st.session_state.db["chats"].get(
      st.session_state.active_chat, []
  )

  for msg in chat_history:
    with st.chat_message(msg["role"]):
      st.markdown(msg["content"])
      if msg["role"] == "assistant":
        with st.expander("📋 Copy Raw Text"):
          st.code(msg["content"], language="markdown")

  attached_file = st.file_uploader(
      "Attach Context File",
      type=["png", "jpg", "jpeg", "pdf", "docx", "csv", "xlsx"],
      key="gen_chat_file",
  )

  user_query = st.chat_input("Ask anything...")

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
              if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                response_placeholder.markdown(full_response + " ▌")

            response_placeholder.markdown(full_response)
            status.update(label="✅ Response Done!", state="complete")

            chat_history.append({"role": "assistant", "content": full_response})
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

  workspace_type = st.radio(
      "Select Section",
      ["📝 MCQ Workspace", "✍️ Focus Written Workspace"],
      horizontal=True,
  )

  if workspace_type == "📝 MCQ Workspace":
    col_s1, col_s2 = st.columns([3, 1])
    mcq_subs = list(st.session_state.db["mcq_subjects"].keys())
    selected_mcq_sub = col_s1.selectbox(
        "Select Subject", mcq_subs if mcq_subs else ["None"]
    )

    new_mcq_sub = col_s2.text_input("New Subject")
    if col_s2.button("Add Subject") and new_mcq_sub:
      st.session_state.db["mcq_subjects"][new_mcq_sub] = {
          "sources": [],
          "chat": [],
          "mistakes": [],
      }
      save_data(st.session_state.db)
      st.rerun()

    if selected_mcq_sub and selected_mcq_sub != "None":
      sub_data = st.session_state.db["mcq_subjects"][selected_mcq_sub]
      m_tab1, m_tab2 = st.tabs(["🎯 Quiz Generator", "📖 Mistakes Log"])

      with m_tab1:
        num_q = st.number_input("Questions", 1, 10, 3)
        diff = st.selectbox("Difficulty", ["Medium", "Hard", "Advanced Exam"])
        mcq_file = st.file_uploader(
            "Attach Material (PDF, DOCX, CSV)",
            type=["pdf", "docx", "csv", "xlsx", "png", "jpg"],
            key="mcq_file_up",
        )

        if st.button("Generate Quiz"):
          if not st.session_state.get("saved_openrouter_key"):
            st.error("Please enter your API Key in the sidebar.")
          else:
            client = get_openrouter_client(
                st.session_state.saved_openrouter_key
            )
            file_content = process_uploaded_file(mcq_file) if mcq_file else ""
            prior_mistakes = [
                m.get("concept", "") for m in sub_data.get("mistakes", [])
            ]
            weakness_context = (
                f"Prior mistakes to test: {', '.join(prior_mistakes)}"
                if prior_mistakes
                else ""
            )

            prompt = (
                f"Generate {num_q} MCQs for '{selected_mcq_sub}'."
                f" Context: {file_content}. Difficulty: {diff}."
                f" {weakness_context} Return STRICT raw JSON list without markdown:"
                ' [{"id":1,"question":"...","options":["A)...","B)...","C)...","D)..."],"correct":"A)...","explanation":"..."}]'
            )

            with st.status("🧠 Generating Quiz...", expanded=True):
              try:
                res = client.chat.completions.create(
                    model="openrouter/free",
                    messages=[
                        MATH_SYSTEM_PROMPT,
                        {"role": "user", "content": prompt},
                    ],
                )
                raw_text = (
                    res.choices[0]
                    .message.content.replace("```json", "")
                    .replace("```", "")
                    .strip()
                )
                st.session_state.quiz = json.loads(raw_text)
              except Exception as qe:
                st.error(f"Quiz Generation Error: {str(qe)}")

        if "quiz" in st.session_state:
          user_ans = {}
          with st.form("mcq_form"):
            for q in st.session_state.quiz:
              st.write(f"**Q{q['id']}: {q['question']}**")
              user_ans[q["id"]] = st.radio(
                  f"Option for Q{q['id']}", q["options"]
              )
            submit_m = st.form_submit_button("Submit Answers")

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
                st.info(f"💡 Explanation: {q['explanation']}")

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

            st.metric("Score", f"{score} / {len(st.session_state.quiz)}")
            save_data(st.session_state.db)

      with m_tab2:
        for m in sub_data.get("mistakes", []):
          st.write(f"- **{m['concept']}**: {m['takeaway']}")

        if st.button("Export Revision Guide (.docx)"):
          fpath = generate_docx(
              selected_mcq_sub, sub_data["mistakes"], "MCQ"
          )
          with open(fpath, "rb") as fp:
            st.download_button("📥 Download Document", fp, file_name=fpath)

  elif workspace_type == "✍️ Focus Written Workspace":
    col_w1, col_w2 = st.columns([3, 1])
    w_subs = list(st.session_state.db["written_subjects"].keys())
    selected_w_sub = col_w1.selectbox(
        "Select Subject", w_subs if w_subs else ["None"]
    )

    new_w_sub = col_w2.text_input("New Subject ")
    if col_w2.button("Add ") and new_w_sub:
      st.session_state.db["written_subjects"][new_w_sub] = {
          "sources": [],
          "chat": [],
          "mistakes": [],
      }
      save_data(st.session_state.db)
      st.rerun()

    if selected_w_sub and selected_w_sub != "None":
      w_sub_data = st.session_state.db["written_subjects"][selected_w_sub]

      custom_instructions = st.text_area(
          "Evaluation Rubric",
          "Focus on clarity, analytical depth, and structural coherence.",
          height=80,
      )
      written_file = st.file_uploader(
          "Attach Reference File",
          type=["pdf", "docx", "png", "jpg"],
          key="written_file_up",
      )
      essay_input = st.text_area("Write / Paste Essay", height=180)

      if st.button("Evaluate Essay"):
        if not st.session_state.get("saved_openrouter_key"):
          st.error("Please enter your API Key in the sidebar.")
        else:
          client = get_openrouter_client(st.session_state.saved_openrouter_key)
          file_text = (
              process_uploaded_file(written_file) if written_file else ""
          )
          combined_text = (
              f"Essay Text: {essay_input}\nAttached Context: {file_text}"
          )

          prompt = (
              f"Evaluate essay for '{selected_w_sub}'. Rubric:"
              f" {custom_instructions}.\nContent: {combined_text}\nOutput"
              ' STRICT JSON:\n{"score":"85%","weakness":"...","strategy":"..."}'
          )

          with st.status("🧠 Evaluating Essay...", expanded=True):
            try:
              res = client.chat.completions.create(
                  model="openrouter/free",
                  messages=[
                      MATH_SYSTEM_PROMPT,
                      {"role": "user", "content": prompt},
                  ],
              )
              raw_text = (
                  res.choices[0]
                  .message.content.replace("```json", "")
                  .replace("```", "")
                  .strip()
              )
              eval_data = json.loads(raw_text)

              st.metric("Evaluation Score", eval_data["score"])
              st.info(f"**Weakness Area**: {eval_data['weakness']}")
              st.success(f"**Strategy / Takeaway**: {eval_data['strategy']}")

              w_sub_data["mistakes"].append({
                  "date": datetime.now().strftime("%Y-%m-%d"),
                  "area": eval_data["weakness"],
                  "correction": eval_data["strategy"],
              })
              save_data(st.session_state.db)
            except Exception as we:
              st.error(f"Evaluation Error: {str(we)}")

      st.divider()
      if st.button("Export Essay Revision Guide (.docx)"):
        fpath = generate_docx(
            selected_w_sub, w_sub_data["mistakes"], "Written"
        )
        with open(fpath, "rb") as fp:
          st.download_button("📥 Download Revision Guide", fp, file_name=fpath)
