import base64
from datetime import datetime
import json
import os
from io import BytesIO
from docx import Document
from openai import OpenAI
import pandas as pd
from PIL import Image
from pypdf import PdfReader
import streamlit as st

st.set_page_config(
    page_title="Agentic Study Platform", page_icon="🎓", layout="wide"
)

DATA_FILE = "study_data.json"

# --- CURATED FREE MODEL POOL WITH SPECIALTY TAGS ---
MODEL_OPTIONS = {
    "Auto-Router [Best Active Free Model]": "openrouter/free",
    "Nvidia Nemotron 3 Ultra [Best Reasoning & Explanations]": (
        "nvidia/nemotron-3-ultra-550b-a55b:free"
    ),
    "Poolside Laguna S 2.1 [Best Code Generator & Programming]": (
        "poolside/laguna-s-2.1:free"
    ),
    "Google Gemma 4 31B [Best Document & Vision Tasks]": (
        "google/gemma-4-31b-it:free"
    ),
    "OpenAI GPT-OSS 20B [Fast & Lightweight]": "openai/gpt-oss-20b:free",
    "InclusionAI Ling 3.0 Flash [Fast Agentic Tasks]": (
        "inclusionai/ling-3.0-flash:free"
    ),
}

# Math formatting system prompt
MATH_SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "When rendering mathematical variables, formulas, or statistical"
        " notation, ALWAYS wrap them in standard LaTeX dollar signs. Use"
        " single dollar signs for inline math (e.g., $\\mu$, $\\bar{x}$,"
        " $\\theta$, $N$, $f(x)$) and double dollar signs for standalone"
        " equations ($$...$$). Never output raw unescaped math commands like"
        " (\\mu) or (\\theta)."
    ),
}


# --- PERSISTENCE LOGIC ---
def load_data():
  if os.path.exists(DATA_FILE):
    try:
      with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
    except Exception:
      pass
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
  with open(DATA_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)


if "db" not in st.session_state:
  st.session_state.db = load_data()

if "active_mode" not in st.session_state:
  st.session_state.active_mode = "general_chat"

if "active_chat" not in st.session_state:
  chats = list(st.session_state.db["chats"].keys())
  st.session_state.active_chat = chats[0] if chats else "Default Chat"


# --- FILE PARSER HELPER ---
def process_uploaded_file(uploaded_file):
  if uploaded_file is None:
    return ""

  file_type = uploaded_file.name.split(".")[-1].lower()
  extracted_text = f"\n--- Attached File: {uploaded_file.name} ---\n"

  try:
    if file_type in ["png", "jpg", "jpeg"]:
      extracted_text += f"[Image attached: {uploaded_file.name}]"
    elif file_type == "pdf":
      reader = PdfReader(uploaded_file)
      for page in reader.pages:
        extracted_text += page.extract_text() or ""
    elif file_type == "docx":
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
    extracted_text += f"Error parsing file: {str(e)}"

  return extracted_text


def generate_docx(subject_name, mistakes_list, section_type="MCQ"):
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


# --- SIDEBAR NAVIGATION & MODEL SELECTOR ---
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
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
  else:
    st.info("👈 Enter your free OpenRouter API Key to start.")
    st.stop()

  st.divider()

  selected_label = st.selectbox(
      "🤖 Select AI Model",
      options=list(MODEL_OPTIONS.keys()),
      index=0,
  )
  selected_model_slug = MODEL_OPTIONS[selected_label]

  st.divider()

  st.subheader("📌 Navigation")
  nav_choice = st.radio(
      "Select View", ["💬 General Chat", "📚 Notebook Workspaces"]
  )
  st.session_state.active_mode = (
      "general_chat"
      if nav_choice == "💬 General Chat"
      else "notebook_studio"
  )

  st.divider()

  if st.session_state.active_mode == "general_chat":
    col_c1, col_c2 = st.columns([3, 1])
    col_c1.subheader("💬 Threads")

    if col_c2.button("➕", key="new_chat"):
      new_chat_name = f"Chat {datetime.now().strftime('%H:%M:%S')}"
      st.session_state.db["chats"][new_chat_name] = []
      save_data(st.session_state.db)
      st.session_state.active_chat = new_chat_name
      st.rerun()

    for chat_name in list(st.session_state.db["chats"].keys()):
      col_btn, col_del = st.columns([4, 1])
      is_active = chat_name == st.session_state.active_chat
      label = f"👉 {chat_name[:14]}" if is_active else f"💬 {chat_name[:14]}"

      if col_btn.button(label, key=f"select_{chat_name}"):
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
    st.subheader("✏️ Rename Active Thread")
    rename_input = st.text_input("Thread Title", st.session_state.active_chat)
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

  # Clean display without duplicate code copy blocks
  for msg in chat_history:
    with st.chat_message(msg["role"]):
      st.markdown(msg["content"])

  attached_file = st.file_uploader(
      "Attach File (Image, PDF, Docx, CSV, Excel)",
      type=["png", "jpg", "jpeg", "pdf", "docx", "csv", "xlsx"],
      key="gen_chat_file",
  )

  user_query = st.chat_input("Ask anything...")

  if user_query:
    file_context = process_uploaded_file(attached_file) if attached_file else ""
    full_prompt = f"{user_query}\n{file_context}" if file_context else user_query

    chat_history.append({"role": "user", "content": full_prompt})
    st.session_state.db["chats"][st.session_state.active_chat] = chat_history
    save_data(st.session_state.db)

    with st.chat_message("user"):
      st.markdown(full_prompt)

    with st.chat_message("assistant"):
      with st.status("🧠 AI is thinking & generating...", expanded=True) as status:
        response_placeholder = st.empty()
        full_response = ""

        api_messages = [MATH_SYSTEM_PROMPT] + [
            {"role": m["role"], "content": m["content"]}
            for m in chat_history[-6:]
        ]

        try:
          response = client.chat.completions.create(
              model=selected_model_slug,
              messages=api_messages,
              stream=True,
          )

          for chunk in response:
            if chunk.choices[0].delta.content:
              full_response += chunk.choices[0].delta.content
              response_placeholder.markdown(full_response + " ▌")

          response_placeholder.markdown(full_response)
          status.update(label="✅ Generation Complete!", state="complete")

          chat_history.append({"role": "assistant", "content": full_response})
          st.session_state.db["chats"][st.session_state.active_chat] = (
              chat_history
          )

          if len(chat_history) == 2 and st.session_state.active_chat.startswith(
              "Chat "
          ):
            title_res = client.chat.completions.create(
                model=selected_model_slug,
                messages=[{
                    "role": "user",
                    "content": (
                        "Generate a concise 3-4 word title for a chat based"
                        f" on this prompt: {user_query}"
                    ),
                }],
            )
            auto_title = (
                title_res.choices[0]
                .message.content.strip()
                .replace('"', "")[:25]
            )
            st.session_state.db["chats"][auto_title] = st.session_state.db[
                "chats"
            ].pop(st.session_state.active_chat)
            st.session_state.active_chat = auto_title

          save_data(st.session_state.db)
          st.rerun()

        except Exception as e:
          status.update(label="❌ Generation Failed", state="error")
          st.error(f"Error: {str(e)}")

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

  # ----------------------------------------------------
  # MCQ WORKSPACE
  # ----------------------------------------------------
  if workspace_type == "📝 MCQ Workspace":
    col_s1, col_s2 = st.columns([3, 1])
    mcq_subs = list(st.session_state.db["mcq_subjects"].keys())
    selected_mcq_sub = col_s1.selectbox(
        "Select MCQ Subject", mcq_subs if mcq_subs else ["None"]
    )

    new_mcq_sub = col_s2.text_input("New Subject")
    if col_s2.button("Add") and new_mcq_sub:
      st.session_state.db["mcq_subjects"][new_mcq_sub] = {
          "sources": [],
          "chat": [],
          "mistakes": [],
      }
      save_data(st.session_state.db)
      st.rerun()

    if selected_mcq_sub and selected_mcq_sub != "None":
      sub_data = st.session_state.db["mcq_subjects"][selected_mcq_sub]
      m_tab1, m_tab2 = st.tabs(["🎯 Exam Generator", "📖 Mistakes Log"])

      with m_tab1:
        num_q = st.number_input("Questions", 1, 10, 3)
        diff = st.selectbox("Difficulty", ["Medium", "Hard", "Advanced Exam"])
        mcq_file = st.file_uploader(
            "Attach Source Material for Quiz (Optional)",
            type=["pdf", "docx", "csv", "xlsx", "png", "jpg"],
            key="mcq_file_up",
        )

        if st.button("Generate Quiz"):
          file_content = process_uploaded_file(mcq_file) if mcq_file else ""
          prompt = (
              f"Generate {num_q} MCQs for subject '{selected_mcq_sub}' based on"
              f" this context: {file_content}. Difficulty: {diff}. Return"
              " STRICTLY a raw JSON list without markdown:\n"
              '[{"id":1,"question":"...","options":["A)...","B)...","C)...","D)..."],"correct":"A)...","explanation":"..."}]'
          )

          with st.status("🧠 Generating Quiz Questions...", expanded=True):
            try:
              res = client.chat.completions.create(
                  model=selected_model_slug,
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
            submit_m = st.form_submit_button("Submit Quiz")

          if submit_m:
            score = 0
            new_m = []
            for q in st.session_state.quiz:
              if user_ans[q["id"]] == q["correct"]:
                score += 1
                st.success(f"Q{q['id']} Correct!")
              else:
                st.error(f"Q{q['id']} Incorrect. Correct: {q['correct']}")
                new_m.append({
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "concept": q["question"],
                    "takeaway": q["explanation"],
                })
            st.metric("Score", f"{score} / {len(st.session_state.quiz)}")
            sub_data["mistakes"].extend(new_m)
            save_data(st.session_state.db)

      with m_tab2:
        for m in sub_data.get("mistakes", []):
          st.write(f"- **{m['concept']}**: {m['takeaway']}")

        if st.button("Export MCQ Revision Guide (.docx)"):
          fpath = generate_docx(
              selected_mcq_sub, sub_data["mistakes"], "MCQ"
          )
          with open(fpath, "rb") as fp:
            st.download_button("📥 Download Document", fp, file_name=fpath)

  # ----------------------------------------------------
  # WRITTEN WORKSPACE & TIMED INSTRUCTIONS
  # ----------------------------------------------------
  elif workspace_type == "✍️ Focus Written Workspace":
    col_w1, col_w2 = st.columns([3, 1])
    w_subs = list(st.session_state.db["written_subjects"].keys())
    selected_w_sub = col_w1.selectbox(
        "Select Written Subject", w_subs if w_subs else ["None"]
    )

    new_w_sub = col_w2.text_input("New Written Subject")
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

      st.subheader("⏱️ Timer & Custom Evaluation Rules")
      custom_instructions = st.text_area(
          "Custom Prompt / Rules for Evaluation",
          "Focus on clarity, analytical depth, structural coherence, and"
          " proper vocabulary.",
          height=80,
      )

      written_file = st.file_uploader(
          "Attach Reference File or Essay Document",
          type=["pdf", "docx", "png", "jpg"],
          key="written_file_up",
      )

      essay_input = st.text_area("Write / Paste your Essay here", height=180)

      if st.button("Evaluate Essay"):
        file_text = (
            process_uploaded_file(written_file) if written_file else ""
        )
        combined_text = (
            f"Essay Text: {essay_input}\nAttached Context: {file_text}"
        )

        prompt = (
            f"Evaluate essay for subject '{selected_w_sub}'. Rules:"
            f" {custom_instructions}.\nContent: {combined_text}\nOutput"
            ' STRICT JSON:\n{"score":"85%","weakness":"...","strategy":"..."}'
        )

        with st.status("🧠 AI is Evaluating Your Essay...", expanded=True):
          try:
            res = client.chat.completions.create(
                model=selected_model_slug,
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
