import json
import os
import time
from datetime import datetime
from docx import Document
import google.genai as genai
from google.genai import types
from google.genai.errors import APIError, ClientError
import streamlit as st

st.set_page_config(
    page_title="Agentic Study Platform", page_icon="🎓", layout="wide"
)

DATA_FILE = "study_data.json"

# --- STANDARD GEMINI MODEL STRINGS ---
CHAT_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash-lite",
]
NOTEBOOK_MODEL = "gemini-2.5-pro"  # Default model for source document analysis


# --- DEFENSIVE DATA LOADING ---
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


# --- INITIALIZE SESSION STATE ---
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

if "editing_idx" not in st.session_state:
  st.session_state.editing_idx = None


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


def generate_chat_title(client, model, user_prompt):
  """Generates a concise 3-5 word topic title based on first user prompt."""
  try:
    title_prompt = (
        "Summarize the following prompt into a brief, descriptive 3 to 5 word chat title. "
        "Do not use quotes or punctuation.\n\nPrompt: " + user_prompt
    )
    res = client.models.generate_content(
        model=model,
        contents=title_prompt,
    )
    clean_title = res.text.strip().replace('"', "").replace("'", "")
    return clean_title[:30] if clean_title else "New Study Chat"
  except Exception:
    return f"Chat {datetime.now().strftime('%b %d %H:%M')}"


# --- SIDEBAR: CONFIGURATION & NAVIGATION ---
with st.sidebar:
  st.title("🎓 Gemini + Notebook Studio")

  default_key = st.secrets.get(
      "GEMINI_API_KEY", st.session_state.get("saved_api_key", "")
  )
  api_key = st.text_input(
      "Gemini API Key",
      value=default_key,
      type="password",
      help="Set GEMINI_API_KEY in Streamlit Secrets for permanent access.",
  )

  if api_key:
    st.session_state.saved_api_key = api_key
    client = genai.Client(api_key=api_key)
  else:
    st.info("👈 Enter your Gemini API Key or configure st.secrets.")
    st.stop()

  st.divider()

  st.subheader("📌 Navigation")
  nav_choice = st.radio(
      "Select Workspace View",
      ["💬 Gemini General Chat", "📚 Notebook Workspaces"],
  )
  if nav_choice == "💬 Gemini General Chat":
    st.session_state.active_mode = "general_chat"
  else:
    st.session_state.active_mode = "notebook_studio"

  st.divider()

  if st.session_state.active_mode == "general_chat":
    col_c1, col_c2 = st.columns([3, 1])
    col_c1.subheader("💬 Threads")

    if col_c2.button("➕", key="new_chat"):
      current_chat_content = st.session_state.db["chats"].get(
          st.session_state.active_chat, []
      )
      if (
          len(current_chat_content) > 0
          or len(st.session_state.db["chats"]) == 0
      ):
        new_chat_name = f"New Chat {datetime.now().strftime('%H:%M')}"
        st.session_state.db["chats"][new_chat_name] = []
        save_data(st.session_state.db)
        st.session_state.active_chat = new_chat_name
        st.rerun()

    for chat_name in list(st.session_state.db["chats"].keys()):
      col_btn, col_ren, col_del = st.columns([3, 1, 1])

      label = (
          f"👉 {chat_name[:10]}.."
          if chat_name == st.session_state.active_chat
          else f"💬 {chat_name[:10]}.."
      )
      if col_btn.button(label, key=f"chat_select_{chat_name}"):
        st.session_state.active_chat = chat_name
        st.rerun()

      with col_ren.popover("✏️"):
        new_title_in = st.text_input(
            "Rename Chat", value=chat_name, key=f"rename_in_{chat_name}"
        )
        if st.button("Save", key=f"save_rename_{chat_name}"):
          if new_title_in and new_title_in != chat_name:
            st.session_state.db["chats"][new_title_in] = (
                st.session_state.db["chats"].pop(chat_name)
            )
            if st.session_state.active_chat == chat_name:
              st.session_state.active_chat = new_title_in
            save_data(st.session_state.db)
            st.rerun()

      if len(st.session_state.db["chats"]) > 1:
        if col_del.button("🗑️", key=f"del_chat_{chat_name}"):
          del st.session_state.db["chats"][chat_name]
          st.session_state.active_chat = list(
              st.session_state.db["chats"].keys()
          )[0]
          save_data(st.session_state.db)
          st.rerun()

# ==========================================
# VIEW 1: GEMINI GENERAL CHAT
# ==========================================
if st.session_state.active_mode == "general_chat":
  col_head1, col_head2 = st.columns([3, 1])
  col_head1.title(f"💬 {st.session_state.active_chat}")

  # Model selector placed inside the general chat interface
  selected_model = col_head2.selectbox(
      "🤖 Select Chat Model",
      CHAT_MODELS,
      index=0,
      key=f"model_select_{st.session_state.active_chat}",
  )

  chat_history = st.session_state.db["chats"].get(
      st.session_state.active_chat, []
  )

  for idx, msg in enumerate(chat_history):
    with st.chat_message(msg["role"]):
      st.markdown(msg["content"])

      col_a1, col_a2 = st.columns([10, 1])
      if msg["role"] == "user":
        if col_a2.button("✏️", key=f"edit_msg_{idx}"):
          st.session_state.editing_idx = idx
          st.rerun()
      else:
        with col_a2.popover("📋"):
          st.code(msg["content"], language=None)

  if st.session_state.editing_idx is not None:
    edit_idx = st.session_state.editing_idx
    with st.form("edit_message_form"):
      st.write("✏️ **Edit Prompt:**")
      edited_text = st.text_area(
          "Update message", chat_history[edit_idx]["content"]
      )
      col_e1, col_e2 = st.columns(2)
      submit_edit = col_e1.form_submit_button("Save & Resubmit")
      cancel_edit = col_e2.form_submit_button("Cancel")

      if submit_edit:
        chat_history = chat_history[:edit_idx]
        chat_history.append({"role": "user", "content": edited_text})
        st.session_state.db["chats"][st.session_state.active_chat] = (
            chat_history
        )
        st.session_state.editing_idx = None
        save_data(st.session_state.db)
        st.rerun()
      if cancel_edit:
        st.session_state.editing_idx = None
        st.rerun()

  with st.expander("📎 Attach Documents/Images to Next Prompt", expanded=False):
    uploaded_files = st.file_uploader(
        "Supported: PDF, PNG, JPG, TXT, PPTX, XLSX, DOCX",
        type=[
            "png",
            "jpg",
            "jpeg",
            "pdf",
            "txt",
            "ppt",
            "pptx",
            "xls",
            "xlsx",
            "doc",
            "docx",
        ],
        accept_multiple_files=True,
        key=f"gen_upload_{st.session_state.active_chat}",
    )

  user_query = st.chat_input("Ask anything...")

  should_generate = user_query or (
      len(chat_history) > 0 and chat_history[-1]["role"] == "user"
  )

  if user_query:
    if len(chat_history) == 0:
      auto_title = generate_chat_title(client, selected_model, user_query)
      old_chat_key = st.session_state.active_chat
      st.session_state.db["chats"][auto_title] = st.session_state.db[
          "chats"
      ].pop(old_chat_key, [])
      st.session_state.active_chat = auto_title
      chat_history = st.session_state.db["chats"][auto_title]

    chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
      st.markdown(user_query)

  if should_generate and (
      len(chat_history) > 0 and chat_history[-1]["role"] == "user"
  ):
    sys_inst = (
        "You are a general study assistant. Respond in the user's preferred"
        " language. Use LaTeX ($ or $$) for math formulas."
    )

    latest_prompt = chat_history[-1]["content"]
    contents = [latest_prompt]

    if uploaded_files:
      for file in uploaded_files:
        try:
          uploaded_part = client.files.upload(file=file)
          contents.append(uploaded_part)
        except Exception as fe:
          st.error(f"Error uploading {file.name}: {str(fe)}")

    with st.chat_message("assistant"):
      status_box = st.info(f"⏳ Generating with {selected_model}...")
      response_placeholder = st.empty()
      full_response = ""

      try:
        stream_response = client.models.generate_content_stream(
            model=selected_model,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=sys_inst),
        )

        for chunk in stream_response:
          if chunk.text:
            full_response += chunk.text
            response_placeholder.markdown(full_response + " ▌")

        response_placeholder.markdown(full_response)
        status_box.empty()

        chat_history.append({"role": "assistant", "content": full_response})
        st.session_state.db["chats"][st.session_state.active_chat] = (
            chat_history
        )
        save_data(st.session_state.db)
      except (ClientError, APIError) as e:
        status_box.empty()
        st.error(f"Gemini API Error: {str(e)}")

# ==========================================
# VIEW 2: NOTEBOOK WORKSPACES
# ==========================================
elif st.session_state.active_mode == "notebook_studio":
  st.title("📚 Notebook Workspaces")
  st.caption(f"Powered by ground-truth model: `{NOTEBOOK_MODEL}`")

  workspace_type = st.radio(
      "Select Section Workspace",
      ["📝 MCQ Workspace", "✍️ Focus Written Workspace"],
      horizontal=True,
  )

  # ----------------------------------------------------
  # SECTION 1: MCQ WORKSPACE
  # ----------------------------------------------------
  if workspace_type == "📝 MCQ Workspace":
    st.subheader("📝 MCQ Subjects Studio")

    col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
    mcq_subs = list(st.session_state.db["mcq_subjects"].keys())
    selected_mcq_sub = col_s1.selectbox(
        "Select MCQ Subject", mcq_subs if mcq_subs else ["None"]
    )

    new_mcq_sub = col_s2.text_input("Create MCQ Subject")
    if col_s2.button("Add Subject") and new_mcq_sub:
      if new_mcq_sub not in st.session_state.db["mcq_subjects"]:
        st.session_state.db["mcq_subjects"][new_mcq_sub] = {
            "sources": [],
            "chat": [],
            "mistakes": [],
        }
        save_data(st.session_state.db)
        st.success(f"Added {new_mcq_sub}")
        st.rerun()

    if selected_mcq_sub and selected_mcq_sub != "None":
      if col_s3.button(
          "🗑️ Delete Subject", key=f"del_sub_mcq_{selected_mcq_sub}"
      ):
        del st.session_state.db["mcq_subjects"][selected_mcq_sub]
        save_data(st.session_state.db)
        st.warning(f"Deleted {selected_mcq_sub}")
        st.rerun()

    if selected_mcq_sub and selected_mcq_sub != "None":
      sub_data = st.session_state.db["mcq_subjects"][selected_mcq_sub]

      with st.expander(
          "📁 Import / Manage Subject Source Documents", expanded=False
      ):
        files = st.file_uploader(
            "Upload files for this subject",
            type=[
                "pdf",
                "png",
                "jpg",
                "jpeg",
                "txt",
                "ppt",
                "pptx",
                "xls",
                "xlsx",
                "doc",
                "docx",
            ],
            accept_multiple_files=True,
            key=f"mcq_files_{selected_mcq_sub}",
        )
        if st.button("Save Sources to Subject"):
          if files:
            for f in files:
              if f.name not in sub_data["sources"]:
                sub_data["sources"].append(f.name)
            save_data(st.session_state.db)
            st.success("Sources updated!")
            st.rerun()

        st.write("**Current Source Documents:**")
        if not sub_data["sources"]:
          st.caption("No sources attached to this subject yet.")
        else:
          for idx, src_file in enumerate(sub_data["sources"]):
            c_f1, c_f2 = st.columns([4, 1])
            c_f1.write(f"📄 {src_file}")
            if c_f2.button(
                "🗑️ Remove", key=f"remove_mcq_src_{selected_mcq_sub}_{idx}"
            ):
              sub_data["sources"].pop(idx)
              save_data(st.session_state.db)
              st.rerun()

      m_tab1, m_tab2, m_tab3 = st.tabs([
          "🎯 Exam Generator",
          "💬 Subject Q&A Chat",
          "📖 Mistakes & Revision Log",
      ])

      with m_tab1:
        c1, c2 = st.columns(2)
        with c1:
          num_q = st.number_input("Number of Questions", 1, 20, 5)
          diff = st.selectbox(
              "Difficulty", ["Easy", "Medium", "Hard", "Advanced Exam Level"]
          )
        with c2:
          timer_m = st.number_input("Practice Timer (Minutes)", 1, 120, 10)

        mcq_custom_inst = st.text_area(
            "📌 Custom Evaluation & Question Generation Instructions",
            placeholder="Focus on specific formulas, chapters, or concepts...",
            key=f"inst_mcq_{selected_mcq_sub}",
        )

        if st.button("Generate Practice Quiz"):
          try:
            prompt = (
                f"Generate {num_q} MCQs for subject '{selected_mcq_sub}'."
                f" Difficulty: {diff}. Instructions: {mcq_custom_inst}."
                f" Context sources: {sub_data['sources']}. Return strictly JSON"
                " format:"
                " [{'id':1,'question':'...','options':['A)...','B)...'],'correct':'A)...','explanation':'...'}]"
            )
            res = client.models.generate_content(
                model=NOTEBOOK_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ),
            )
            st.session_state.quiz = json.loads(res.text)
            st.session_state.quiz_start = time.time()
            st.session_state.quiz_duration = timer_m * 60
          except Exception as qe:
            st.error(f"Quiz Generation Error: {str(qe)}")

        if "quiz" in st.session_state:
          elapsed = time.time() - st.session_state.quiz_start
          rem = st.session_state.quiz_duration - elapsed
          if rem > 0:
            st.warning(
                f"⏱️ Time Remaining: {int(rem // 60)}m {int(rem % 60)}s"
            )
          else:
            st.error("⏰ Time Expired!")

          user_ans = {}
          with st.form("mcq_form"):
            for q in st.session_state.quiz:
              st.write(f"**Q{q['id']}: {q['question']}**")
              user_ans[q['id']] = st.radio(
                  f"Choose option Q{q['id']}", q['options']
              )
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
                new_m.append({
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "concept": q['question'],
                    "takeaway": q['explanation'],
                })
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
          with st.chat_message("user"):
            st.markdown(q_in)

          with st.chat_message("assistant"):
            st_box = st.info("⏳ Generating response...")
            res_holder = st.empty()
            full_txt = ""

            try:
              stream_res = client.models.generate_content_stream(
                  model=NOTEBOOK_MODEL,
                  contents=q_in,
                  config=types.GenerateContentConfig(
                      system_instruction=(
                          f"Answer for {selected_mcq_sub} based on sources:"
                          f" {sub_data['sources']}"
                      )
                  ),
              )
              for chunk in stream_res:
                if chunk.text:
                  full_txt += chunk.text
                  res_holder.markdown(full_txt + " ▌")

              res_holder.markdown(full_txt)
              st_box.empty()
              sub_data["chat"].append(
                  {"role": "assistant", "content": full_txt}
              )
              save_data(st.session_state.db)
            except Exception as ce:
              st_box.empty()
              st.error(f"Error: {str(ce)}")

      with m_tab3:
        st.write("### Recorded Mistake Entries")
        for idx, m in enumerate(sub_data.get("mistakes", [])):
          col_m1, col_m2 = st.columns([5, 1])
          col_m1.write(f"- **{m['concept']}**: {m['takeaway']}")
          if col_m2.button(
              "Delete Entry", key=f"del_mcq_{selected_mcq_sub}_{idx}"
          ):
            sub_data["mistakes"].pop(idx)
            save_data(st.session_state.db)
            st.rerun()

        if st.button("Export MCQ Revision Guide (.docx)"):
          fpath = generate_docx(
              selected_mcq_sub, sub_data["mistakes"], "MCQ"
          )
          with open(fpath, "rb") as fp:
            st.download_button("📥 Download Document", fp, file_name=fpath)

  # ----------------------------------------------------
  # SECTION 2: FOCUS WRITTEN WORKSPACE
  # ----------------------------------------------------
  elif workspace_type == "✍️ Focus Written Workspace":
    st.subheader("✍️ Focus Written Subjects Studio")

    col_w1, col_w2, col_w3 = st.columns([2, 1, 1])
    w_subs = list(st.session_state.db["written_subjects"].keys())
    selected_w_sub = col_w1.selectbox(
        "Select Written Subject", w_subs if w_subs else ["None"]
    )

    new_w_sub = col_w2.text_input("Create Written Subject")
    if col_w2.button("Add Subject") and new_w_sub:
      if new_w_sub not in st.session_state.db["written_subjects"]:
        st.session_state.db["written_subjects"][new_w_sub] = {
            "sources": [],
            "chat": [],
            "mistakes": [],
        }
        save_data(st.session_state.db)
        st.success(f"Added {new_w_sub}")
        st.rerun()

    if selected_w_sub and selected_w_sub != "None":
      if col_w3.button("🗑️ Delete Subject", key=f"del_sub_w_{selected_w_sub}"):
        del st.session_state.db["written_subjects"][selected_w_sub]
        save_data(st.session_state.db)
        st.warning(f"Deleted {selected_w_sub}")
        st.rerun()

    if selected_w_sub and selected_w_sub != "None":
      w_sub_data = st.session_state.db["written_subjects"][selected_w_sub]

      with st.expander(
          "📁 Import / Manage Subject Source Documents", expanded=False
      ):
        w_files = st.file_uploader(
            "Upload files for this subject",
            type=[
                "pdf",
                "png",
                "jpg",
                "jpeg",
                "txt",
                "ppt",
                "pptx",
                "xls",
                "xlsx",
                "doc",
                "docx",
            ],
            accept_multiple_files=True,
            key=f"w_files_{selected_w_sub}",
        )
        if st.button("Save Sources to Subject"):
          if w_files:
            for f in w_files:
              if f.name not in w_sub_data["sources"]:
                w_sub_data["sources"].append(f.name)
            save_data(st.session_state.db)
            st.success("Sources updated!")
            st.rerun()

        st.write("**Current Source Documents:**")
        if not w_sub_data["sources"]:
          st.caption("No sources attached to this subject yet.")
        else:
          for idx, src_file in enumerate(w_sub_data["sources"]):
            c_f1, c_f2 = st.columns([4, 1])
            c_f1.write(f"📄 {src_file}")
            if c_f2.button(
                "🗑️ Remove", key=f"remove_w_src_{selected_w_sub}_{idx}"
            ):
              w_sub_data["sources"].pop(idx)
              save_data(st.session_state.db)
              st.rerun()

      w_t1, w_t2, w_t3 = st.tabs([
          "✍️ Evaluator & Benchmark",
          "💬 Subject Q&A Chat",
          "📖 Revision Log & Export",
      ])

      with w_t1:
        target_benchmark = st.text_area("Benchmark Writing Sample")

        written_custom_inst = st.text_area(
            "📌 Custom Evaluation Guidelines / Rubrics",
            placeholder="Grade strictly according to academic standards...",
            key=f"inst_w_{selected_w_sub}",
        )

        essay = st.text_area("Your Essay Input", height=150)
        if st.button("Evaluate Essay") and essay:
          try:
            prompt = (
                f"Evaluate essay for '{selected_w_sub}'. Benchmark:"
                f" {target_benchmark}. Guidelines: {written_custom_inst}."
                f" Essay: {essay}. Output JSON:"
                " {'score':'80%','weakness':'...','strategy':'...'}"
            )
            res = client.models.generate_content(
                model=NOTEBOOK_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ),
            )
            eval_data = json.loads(res.text)
            st.metric("Style Match / Evaluation Score", eval_data["score"])
            w_sub_data["mistakes"].append({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "area": eval_data["weakness"],
                "correction": eval_data["strategy"],
            })
            save_data(st.session_state.db)
            st.success("Writing feedback saved to log!")
          except Exception as we:
            st.error(f"Evaluation Error: {str(we)}")

      with w_t2:
        for chat in w_sub_data.get("chat", []):
          with st.chat_message(chat["role"]):
            st.markdown(chat["content"])
        wq_in = st.chat_input(f"Ask about writing in {selected_w_sub}...")
        if wq_in:
          w_sub_data["chat"].append({"role": "user", "content": wq_in})
          with st.chat_message("user"):
            st.markdown(wq_in)

          with st.chat_message("assistant"):
            st_box_w = st.info("⏳ Generating response...")
            res_holder_w = st.empty()
            full_txt_w = ""

            try:
              stream_res_w = client.models.generate_content_stream(
                  model=NOTEBOOK_MODEL,
                  contents=wq_in,
                  config=types.GenerateContentConfig(
                      system_instruction=(
                          f"Answer for {selected_w_sub} using sources:"
                          f" {w_sub_data['sources']}"
                      )
                  ),
              )
              for chunk in stream_res_w:
                if chunk.text:
                  full_txt_w += chunk.text
                  res_holder_w.markdown(full_txt_w + " ▌")

              res_holder_w.markdown(full_txt_w)
              st_box_w.empty()
              w_sub_data["chat"].append(
                  {"role": "assistant", "content": full_txt_w}
              )
              save_data(st.session_state.db)
            except Exception as we:
              st_box_w.empty()
              st.error(f"Error: {str(we)}")

      with w_t3:
        st.write("### Recorded Writing Structural & Grammar Pitfalls")
        for idx, w in enumerate(w_sub_data.get("mistakes", [])):
          col_w1, col_w2 = st.columns([5, 1])
          col_w1.write(f"- **{w['area']}**: {w['correction']}")
          if col_w2.button("Delete Entry", key=f"del_w_{selected_w_sub}_{idx}"):
            w_sub_data["mistakes"].pop(idx)
            save_data(st.session_state.db)
            st.rerun()

        if st.button("Export Written Revision Guide (.docx)"):
          fpath = generate_docx(
              selected_w_sub, w_sub_data["mistakes"], "Writing"
          )
          with open(fpath, "rb") as fp:
            st.download_button("📥 Download Document", fp, file_name=fpath)
