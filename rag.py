import hashlib
import json
from io import BytesIO
from supabase import create_client
import extra_streamlit_components as stx

from sentence_transformers import SentenceTransformer
from groq import Groq
import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
from pypdf import PdfReader
import tiktoken
from utils import (
    count_tokens,
    load_text,
    loadpdf,
    chunk_text,
    chunk_pdf,
)

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
SUPABASE_STORAGE_BUCKET = st.secrets["SUPABASE_STORAGE_BUCKET"]


def get_supabase():
    if "supabase_client" not in st.session_state:
        st.session_state.supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return st.session_state.supabase_client


@st.cache_resource
def get_groq():
    return Groq(api_key=GROQ_API_KEY)


@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


cookies = stx.CookieManager(key="auth_cookies")

supabase = get_supabase()
groq_client = get_groq()
model = get_embedding_model()

if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None and not st.session_state.get("logged_out"):
    refresh_token = cookies.get("sb_refresh_token")
    if refresh_token:
        try:
            restored = supabase.auth.refresh_session(refresh_token)
            if restored and restored.user:
                st.session_state.user = restored.user
                cookies.set(
                    "sb_refresh_token",
                    restored.session.refresh_token,
                    key="set_refresh",
                )
        except Exception:
            cookies.delete("sb_refresh_token", key="del_refresh")

if st.session_state.user is not None:
    name = st.session_state.user.user_metadata.get("name", "User")
    a, b = st.columns([7, 2])
    with a:
        st.write(f"User: {name}")
    with b:
        if st.button("Log out"):
            try:
                supabase.auth.sign_out({"scope": "local"})
            except Exception:
                pass
            cookies.delete("sb_refresh_token", key="logout_cookie")
            st.session_state.clear()
            st.query_params.clear()
            st.session_state.logged_out = True
            st.rerun()
else:
    st.title("Log in/Sign up")
    login, signup = st.tabs(["Log in", "Sign up"])
    with login:
        st.header("Log in")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Log in!", key="login_btn"):
            try:
                response = supabase.auth.sign_in_with_password({"email": email, "password": password})
                st.session_state.pop("logged_out", None)
                st.session_state.user = response.user
                cookies.set("sb_refresh_token", response.session.refresh_token, key="login_cookie")
                st.success("Logged in!")
                st.rerun()
            except Exception as e:
                st.error(f"Log in failed. Error: {e}")
    with signup:
        st.header("Create account")
        name = st.text_input("Name", key="signup_name")
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm_password")
        if st.button("Sign up", key="signup_btn"):
            if password != confirm_password:
                st.error("Passwords do not match")
            else:
                try:
                    response = supabase.auth.sign_up({"email": email, "password": password, "options": {"data": {"name": name}}})
                    st.success("Account created!")
                    st.write("Please check email for confirmation")
                except Exception as e:
                    st.error(f"Failed to create account. Error: {e}")
    st.stop()

USER_ID = st.session_state.user.id
from summarize_agent import build_agent
agent = build_agent(supabase, groq_client, model, USER_ID)

st.title("Chatbot")




MAX_TOKENS = 100000



def get_chats():
    response = supabase.table("chats").select("*").eq("userId", USER_ID).order("created_at").execute()
    return response.data or []


def create_chat():
    chats = get_chats()
    title = f"Chat {len(chats) + 1}"
    response = supabase.table("chats").insert({"userId": USER_ID, "title": title}).execute()
    return response.data[0]


def get_messages(chat_id):
    response = (
        supabase
        .table("messages")
        .select("id, created_at, chat_id, role, content")
        .eq("chat_id", chat_id)
        .order("created_at")
        .execute()
    )
    return response.data or []


def save_message(chat_id, role, content):
    supabase.table("messages").insert({
        "chat_id": chat_id,
        "role": role,
        "content": content,
    }).execute()


def get_chat_token_count(chat_id):
    total = 0
    for message in get_messages(chat_id):
        if message["role"] == "quiz":
            continue 
        total += count_tokens(message["content"])
    return total


def get_documents():
    response = (
        supabase
        .table("documents")
        .select("*")
        .eq("userId", USER_ID)
        .order("uploaded_at", desc=True)
        .execute()
    )
    return response.data or []


def get_document_by_id(document_id):
    response = (
        supabase
        .table("documents")
        .select("*")
        .eq("id", document_id)
        .eq("userId", USER_ID)
        .limit(1)
        .execute()
    )
    if not response.data:
        return None
    return response.data[0]


def document_exists_by_hash(file_hash):
    for document in get_documents():
        if document.get("storage_filename") == file_hash:
            return document
    return None


def process_document(file):
    file_bytes = file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    if document_exists_by_hash(file_hash):
        return (False, f"{file.name} has already been uploaded.")

    if file.type == "text/plain":
        chunks = chunk_text(load_text(file_bytes))
    elif file.type == "application/pdf":
        chunks = chunk_pdf(loadpdf(file_bytes))
    else:
        return (False, "Only PDF and TXT files are supported.")

    if not chunks:
        return (False, "No text could be extracted from this file.")

    document_id = f"{USER_ID}_{file_hash}"
    storage_filename = file_hash
    storage_path = f"{USER_ID}/{file_hash}_{file.name}"

    texts = [chunk["Text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=False)
    chunk_rows = [
        {
            "id": f"{document_id}_{i}",
            "document_id": document_id,
            "userId": USER_ID,
            "content": chunk["Text"],
            "embedding": embedding.tolist(),
            "Filename": file.name,
            "pagenumber": chunk["Page number"],
        }
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    supabase.table("documents").insert({
        "id": document_id,
        "userId": USER_ID,
        "original_filename": file.name,
        "storage_path": storage_path,
        "storage_filename": storage_filename,
        "mime_type": file.type,
        "filesize": len(file_bytes),
    }).execute()

    try:
        supabase.storage.from_(SUPABASE_STORAGE_BUCKET).upload(
            storage_path,
            file_bytes,
            {"content-type": file.type, "upsert": "false"},
        )
        supabase.table("document_chunks").insert(chunk_rows).execute()
    except Exception:
        insert_chunk_failed(document_id, storage_path)

    return (True, f"Uploaded and processed {file.name}")


def insert_chunk_failed(document_id, storage_path):
    failed = []
    steps = [
        ("chunks", lambda: supabase.table("document_chunks").delete()
            .eq("document_id", document_id).eq("userId", USER_ID).execute()),
        ("file", lambda: supabase.storage.from_(SUPABASE_STORAGE_BUCKET).remove([storage_path])),
        ("document row", lambda: supabase.table("documents").delete()
            .eq("id", document_id).eq("userId", USER_ID).execute()),
    ]
    for name, step in steps:
        try:
            step()
        except Exception:
            failed.append(name)
    if failed:
        st.warning(
            f"Cleanup incomplete ({', '.join(failed)}). "
            "Delete the document from the sidebar before retrying."
        )

def delete_document(document_id):
    document = get_document_by_id(document_id)
    if not document:
        return (False, "Document not found.")

    supabase.table("document_chunks").delete().eq("document_id", document_id).eq("userId", USER_ID).execute()
    supabase.table("documents").delete().eq("id", document_id).eq("userId", USER_ID).execute()

    try:
        supabase.storage.from_(SUPABASE_STORAGE_BUCKET).remove([document["storage_path"]])
    except Exception as e:
        return (True, f"Deleted {document['original_filename']}, but the stored file could not be removed: {e}")

    return (True, f"Deleted {document['original_filename']}")


def render_quiz(message_id, quiz):
    """Render one quiz stored as a chat message. Keys are namespaced by message id
    so several quizzes in the same chat don't interfere with each other."""
    prefix = f"quiz_{message_id}"
    questions = quiz["questions"]

    st.markdown(f"**Quiz: {quiz['filename']}**")

    for i, question in enumerate(questions):
        st.markdown(f"**{i + 1}. {question['question']}**")
        st.radio(
            "Select an answer",
            options=range(len(question["options"])),
            format_func=lambda x, q=question: q["options"][x],
            index=None,
            key=f"{prefix}_q_{i}",
            label_visibility="collapsed",
        )

    answers = {i: st.session_state.get(f"{prefix}_q_{i}") for i in range(len(questions))}
    submitted_key = f"{prefix}_submitted"

    if st.button("Submit answers", key=f"{prefix}_submit"):
        st.session_state[submitted_key] = True
        score = sum(1 for i, q in enumerate(questions) if answers[i] == q["correct_index"])
        if quiz.get("id"):
            try:
                agent.save_attempt(quiz["id"], answers, score, len(questions))
            except Exception as e:
                st.warning(f"Could not save attempt: {e}")

    if st.session_state.get(submitted_key):
        score = 0
        for i, question in enumerate(questions):
            correct = question["correct_index"]
            if answers[i] == correct:
                score += 1
                st.success(f"{i + 1}. Correct")
            else:
                st.error(f"{i + 1}. Answer: {question['options'][correct]}")
            if question["explanation"]:
                st.caption(question["explanation"])
        st.subheader(f"Score: {score}/{len(questions)}")

    if st.button("Remove quiz", key=f"{prefix}_remove"):
        supabase.table("messages").delete().eq("id", message_id).execute()
        st.rerun()


chats = get_chats()

if not chats:
    chats = [create_chat()]

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = chats[0]["id"]

if st.session_state.current_chat_id not in [chat["id"] for chat in chats]:
    st.session_state.current_chat_id = chats[0]["id"]

current_chat_id = st.session_state.current_chat_id

with st.sidebar:
    st.title("All Chats")

    if st.button("New Chat", use_container_width=True):
        st.session_state.current_chat_id = create_chat()["id"]
        st.rerun()

    st.divider()

    for chat in get_chats():
        if st.button(chat["title"], use_container_width=True, key=f"chat_{chat['id']}"):
            st.session_state.current_chat_id = chat["id"]
            st.rerun()

    st.divider()

    st.title("Documents")

    documents = get_documents()

    if documents:
        for document in documents:
            doc_col, del_col = st.columns([5, 1])
            with doc_col:
                st.write(document["original_filename"])
                st.caption(f"{document.get('filesize', 0) / 1024:.1f} KB")
            with del_col:
                if st.button("🗑️", key=f"del_{document['id']}", help="Delete document"):
                    ok, msg = delete_document(document["id"])
                    if ok:
                        st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1
                        st.rerun()
                    else:
                        st.error(msg)

st.subheader("Upload a document")

files = st.file_uploader(
    "Upload a PDF or TXT file",
    type=["txt", "pdf"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.get('uploader_key', 0)}",
)

if files:
    for file in files:
        try:
            with st.spinner(f"Processing {file.name}..."):
                added, message = process_document(file)
            if added:
                st.success(message)
            else:
                st.info(message)
        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")

st.subheader("Summarize a document")
documents = get_documents()
if documents:
    selected_doc = st.selectbox(
        "Select a document",
        documents,
        format_func=lambda d: f"{d['original_filename']} ({d['uploaded_at'][:10]})",
    )
    selected_id = selected_doc["id"]
    col_a, col_b = st.columns(2)
    with col_a:
        num_questions = st.slider("Number of questions", 1, 15, 5)
    with col_b:
        difficulty = st.radio(
            "Difficulty",
            options=["easy", "medium", "hard"],
            index=1,
            horizontal=True,
        )

    topic = st.text_input(
        "Focus on a topic (optional)",
        placeholder="e.g. dynamic programming",
    )

    col_c, col_d = st.columns(2)

    with col_c:
        if st.button("Summarize", use_container_width=True):
            try:
                with st.spinner("Summarizing..."):
                    result = agent.tool_summarize_document(selected_id)
                save_message(
                    current_chat_id,
                    "assistant",
                    f"**Summary of {selected_doc['original_filename']}**\n\n{result}",
                )
                st.rerun()
            except Exception as e:
                st.error(f"Could not generate summary: {e}")

    with col_d:
        if st.button("Generate quiz", use_container_width=True):
            try:
                with st.spinner("Writing questions..."):
                    agent.tool_generate_quiz(
                        selected_id,
                        num_questions=num_questions,
                        topic=topic.strip() or None,
                        difficulty=difficulty,
                    )
                if agent.last_quiz:
                    save_message(current_chat_id, "quiz", json.dumps(agent.last_quiz))
                    st.rerun()
                else:
                    st.error("Could not generate a quiz from that document.")
            except Exception as e:
                st.error(f"Could not generate quiz: {e}")
else:
    st.info("Upload a document first.")

with st.expander("My quiz stats"):
    stats = agent.get_quiz_stats()

    if not stats["attempts"]:
        st.info("Take a quiz to start building stats.")
    else:
        a, b, c = st.columns(3)
        a.metric("Attempts", stats["attempts"])
        b.metric("Average", f"{stats['average_percent']}%")
        c.metric("Best", f"{stats['best_percent']}%")

        st.caption(f"{stats['correct']} correct out of {stats['questions_answered']} questions")

        if len(stats["history"]) > 1:
            st.line_chart({"Score %": [h["percent"] for h in stats["history"]]})

        if stats["by_document"]:
            st.markdown("**By document** (weakest first)")
            for row in stats["by_document"]:
                st.write(
                    f"{row['filename']} — {row['percent']}% "
                    f"({row['score']}/{row['total']}, {row['attempts']} attempt(s))"
                )


messages = get_messages(current_chat_id)

for message in messages:
    if message["role"] == "quiz":
        with st.chat_message("assistant", avatar="📝"):
            try:
                render_quiz(message["id"], json.loads(message["content"]))
            except (json.JSONDecodeError, KeyError):
                st.error("This quiz could not be loaded.")
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

chat_tokens = get_chat_token_count(current_chat_id)

if chat_tokens >= MAX_TOKENS:
    st.error("Token limit reached. Please create a new chat.")
    st.stop()

query = st.chat_input("Ask something about the document...")

if query:
    if chat_tokens + count_tokens(query) >= MAX_TOKENS:
        st.error("This message would exceed the token limit. Please create a new chat.")
        st.stop()

    save_message(current_chat_id, "user", query)

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.status("Thinking...") as status:
            def on_step(name, args):
                status.write(f"`{name}` {args}")
            try:
                answer = agent.run(
                    query,
                    history=[m for m in messages if m["role"] != "quiz"],
                    on_step=on_step,
                )
                status.update(label="Done", state="complete")
            except Exception as e:
                answer = f"Something went wrong: {e}"
                status.update(label="Failed", state="error")
        st.markdown(answer)

    save_message(current_chat_id, "assistant", answer)

    if agent.last_quiz:
        save_message(current_chat_id, "quiz", json.dumps(agent.last_quiz))
        st.rerun()
