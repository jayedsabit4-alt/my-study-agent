import json
import os
import time
from datetime import datetime
from docx import Document
from openai import OpenAI
import streamlit as st
from tenacity import retry, stop_after_attempt, wait_exponential

st.set_page_config(
    page_title="Agentic Study Workspace", page_icon="🎓", layout="wide"
)

DATA_FILE = "study_data.json"

# --- GUARANTEED FREE OPENROUTER MODELS ---
MODEL_AUTO = "openrouter/free"  # Dynamic router (Always active)
MODEL_REASONING = "nvidia/nemotron-3-ultra-550b-a55b:free"  # Heavy reasoning
MODEL_GENERAL = "google/gemma-4-31b-it:free"  # General academic
MODEL_STRUCTURED = "openai/gpt-oss-20b:free"  # Fast structured output

FREE_MODELS = [MODEL_AUTO, MODEL_REASONING, MODEL_GENERAL, MODEL_STRUCTURED]

# --- PERSISTENCE HELPERS ---
def load_data():
  data = {}
  if os.path.exists(DATA_FILE):
    try:
      with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    except Exception:
      data = {}

  if "chats" not in data or not isinstance(data["chats"], dict):
    data["chats"] = {"Default Chat": []}
  if "mcq_subjects" not in data or not isinstance(data["mcq_subjects"], dict):
    data["mcq_subjects"] = {
        "General Math": {"sources": [], "chat": [], "mistakes": []}
    }
  if "written_subjects" not in data or not isinstance(
      data["written_subjects"], dict
  ):
    data["written_subjects"] = {
        "Academic Essay": {"sources": [], "chat": [], "mistakes": []}
    }

  empty_chats = [
      k
      for k, v in data["chats"].items()
      if len(v) == 0 and k != "Default Chat"
  ]
  for ec in empty_chats:
    del data["chats"][ec]

  return data


def save_data(data):
  with open(DATA_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)


if "db" not in st.session_state:
  st.session_state.db = load_data()

if "active_mode" not in st.session_state:
  st.session_state.active_mode = "general_chat"

if "active_chat" not in st.session_state:
  st.session_state.active_chat = (
      list(st.session_state.db["chats"].keys())[0]
      if st.session_state.db["chats"]
      else "Default Chat"
  )


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


# --- SIDEBAR: OPENROUTER AUTH & NAVIGATION ---
with st.sidebar:
  st.title("🎓 OpenRouter Study Studio")

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
      new_chat_name = f"Chat {datetime.now().strftime('%H:%M')}"
      st.session_state.db["chats"][new_chat_name] = []
      save_data(st.session_state.db)
      st.session_state.active_chat = new_chat_name
      st.rerun()

    for chat_name in list(st.session_state.db["chats"].keys()):
      col_btn, col_del = st.columns([4, 1])
      label = (
          f"👉 {chat_name[:12]}.."
          if chat_name == st.session_state.active_chat
          else f"💬 {chat_name[:12]}.."
      )
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

# ==========================================
# VIEW 1: GENERAL CHAT
# ==========================================
if st.session_state.active_mode == "general_chat":
  col_h1, col_h2 = st.columns([3, 1])
  col_h1.title(f"💬 {st.session_state.active_chat}")

  selected_model = col_h2.selectbox(
      "🤖 Model", FREE_MODELS, index=0, key="gen_model"
  )
  chat_history = st.session_state.db["chats"].get(
      st.session_state.active_chat, []
  )

  for msg in chat_history:
    with st.chat_message(msg["role"]):
      st.markdown(msg["content"])

  user_query = st.chat_input("Ask anything...")

  if user_query:
    chat_history.append({"role": "user", "content": user_query})
    st.session_state.db["chats"][st.session_state.active_chat] = chat_history
    save_data(st.session_state.db)

    with st.chat_message("user"):
      st.markdown(user_query)

    with st.chat_message("assistant"):
      response_placeholder = st.empty()
      full_response = ""

      # Prepare message stack for OpenAI format
      api_messages = [
          {"role": m["role"], "content": m["content"]}
          for m in chat_history[-6:]
      ]

      try:
        response = client.chat.completions.create(
            model=selected_model,
            messages=api_messages,
            stream=True,
        )

        for chunk in response:
          if chunk.choices[0].delta.content:
            full_response += chunk.choices[0].delta.content
            response_placeholder.markdown(full_response + " ▌")

        response_placeholder.markdown(full_response)
        chat_history.append({"role": "assistant", "content": full_response})
        st.session_state.db["chats"][st.session_state.active_chat] = (
            chat_history
        )
        save_data(st.session_state.db)
      except Exception as e:
        st.error(f"OpenRouter Error: {str(e)}")

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

        if st.button("Generate Quiz"):
          prompt = (
              f"Generate {num_q} MCQs for subject '{selected_mcq_sub}'."
              f" Difficulty: {diff}. Return STRICTLY a raw JSON list without"
              " markdown wrapping:\n"
              '[{"id":1,"question":"...","options":["A)...","B)...","C)...","D)..."],"correct":"A)...","explanation":"..."}]'
          )
          try:
            res = client.chat.completions.create(
                model=MODEL_STRUCTURED,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = (
                res.choices[0]
                .message.content.replace("```json", "")
                .replace("```", "")
                .strip()
            )
            st.session_state.quiz = json.loads(raw_text)
          except Exception as qe:
            st.error(f"Parsing error: {str(qe)}")

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
        for idx, m in enumerate(sub_data.get("mistakes", [])):
          st.write(f"- **{m['concept']}**: {m['takeaway']}")

        if st.button("Export Revision Guide (.docx)"):
          fpath = generate_docx(
              selected_mcq_sub, sub_data["mistakes"], "MCQ"
          )
          with open(fpath, "rb") as fp:
            st.download_button("📥 Download Document", fp, file_name=fpath)

  # ----------------------------------------------------
  # WRITTEN WORKSPACE
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

      essay = st.text_area("Your Essay Input", height=150)
      if st.button("Evaluate Essay") and essay:
        prompt = (
            f"Evaluate essay for '{selected_w_sub}'. Essay: {essay}. Output"
            " STRICT JSON:\n"
            '{"score":"85%","weakness":"...","strategy":"..."}'
        )
        try:
          res = client.chat.completions.create(
              model=MODEL_REASONING,
              messages=[{"role": "user", "content": prompt}],
          )
          raw_text = (
              res.choices[0]
              .message.content.replace("```json", "")
              .replace("```", "")
              .strip()
          )
          eval_data = json.loads(raw_text)
          st.metric("Evaluation Score", eval_data["score"])
          w_sub_data["mistakes"].append({
              "date": datetime.now().strftime("%Y-%m-%d"),
              "area": eval_data["weakness"],
              "correction": eval_data["strategy"],
          })
          save_data(st.session_state.db)
          st.success("Feedback logged to Revision Guide!")
        except Exception as we:
          st.error(f"Evaluation Error: {str(we)}")
