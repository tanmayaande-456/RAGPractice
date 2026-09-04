import hashlib
from io import BytesIO
from supabase import create_client

# import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
from pypdf import PdfReader
import tiktoken

GROQ_API_KEY=st.secrets["GROQ_API_KEY"]
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
SUPABASE_STORAGE_BUCKET = st.secrets["SUPABASE_STORAGE_BUCKET"]
# USER_ID = st.secrets["USER_ID"]

# CHROMA_PATH = "chroma_db"
# COLLECTION_NAME = "rag_collection"

@st.cache_resource
def get_supabase():
    return create_client(
        SUPABASE_URL,
        SUPABASE_KEY
    )


@st.cache_resource
def get_groq():
    return Groq(
        api_key=GROQ_API_KEY
    )


@st.cache_resource
def get_embedding_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


# @st.cache_resource
# def get_chroma_collection():
#     client = chromadb.PersistentClient(
#         path=CHROMA_PATH
#     )
#     collection = client.get_or_create_collection(
#         name=COLLECTION_NAME
#     )
#     return collection


supabase = get_supabase()
groq_client = get_groq()
model=get_embedding_model()
# collection = get_chroma_collection()

if "user" not in st.session_state:
    st.session_state.user=None

if st.session_state.user is not None:
    name=st.session_state.user.user_metadata.get("name", "User")
    a, b=st.columns([7, 2])
    with a:
        st.write(f"User: {name}")
    with b:
        if (st.button("Log out")):
            supabase.auth.sign_out()
            st.session_state.user=None
            st.rerun()
else:
    st.title("Log in/Sign up")
    login, signup=st.tabs(["Log in", "Sign up"])
    with login:
        st.header("Log in")
        email=st.text_input("Email", key="login_email")
        password=st.text_input("Password", type="password", key="login_password")
        if (st.button("Log in!", key="login_btn")):
            try:
                response=supabase.auth.sign_in_with_password({"email": email, "password": password})
                st.session_state.user=response.user
                st.success("Logged in!")
                st.rerun()
            except Exception as e:
                st.error(f"Log in failed. Error: {e}")
    with signup:
        st.header("Create account")
        name=st.text_input("Name", key="signup_name")
        email=st.text_input("Email", key="signup_email")
        password=st.text_input("Password", type="password", key="signup_password")
        confirm_password=st.text_input("Confirm Password", type="password", key="signup_confirm_password")
        if (st.button("Sign up", key="signup_btn")):
            if (password != confirm_password):
                st.error("Passwords do not match")
            else:
                try:
                    response=supabase.auth.sign_up({"email": email, "password": password, "options": {"data": {"name": name}}})
                    st.success("Account created!")
                    st.write("Please check email for confirmation")
                except Exception as e:
                    st.error(f"Failed to create account. Error: {e}")
                    # st.exception(e)
    st.stop()

USER_ID=st.session_state.user.id

st.title("Chatbot")

encoding=tiktoken.get_encoding("gpt2")
def count_tokens(text):
    return len(encoding.encode(text))

MAX_TOKENS = 10000
def load_text(file_bytes):
    return file_bytes.decode("utf-8")

def loadpdf(file_bytes):
    read=PdfReader(BytesIO(file_bytes))
    pages=[]
    for pagenumber, page in enumerate(read.pages, start=1):
        extract=page.extract_text()
        if extract:
            pages.append({
                "Page number": pagenumber,
                "Text": extract
            })
    return pages


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

def get_chats():
    response=(
        supabase.table("chats").select("*").eq("userId", USER_ID).order("created_at").execute()
    )
    return response.data or []

def create_chat():
    chats=get_chats()
    chat_number=len(chats)+1
    title=f"Chat {chat_number}"
    response=(
        supabase.table("chats").insert({"userId": USER_ID, "title": title}).execute()
    )
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

def save_message(chat_id,role,content):
    supabase.table("messages").insert({
            "chat_id": chat_id,
            "role": role,
            "content": content
        }).execute()

def get_chat_token_count(chat_id):

    messages = get_messages(chat_id)

    total = 0

    for message in messages:

        total += count_tokens(
            message["content"]
        )

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

    documents = get_documents()

    for document in documents:

        if document.get("storage_filename") == file_hash:

            return document

    return None



def chunk_text(text, chunk_size=5):
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i:i+chunk_size])
        chunks.append({
            "Page number": 0,
            "Text": chunk
        })
    return chunks

def chunk_pdf(pages, chunk_size=5):
    chunks=[]
    for stuff in pages:
        pagenumber=stuff["Page number"]
        text=stuff["Text"]
        sentence=sent_tokenize(text)
        for i in range(0, len(sentence), chunk_size):
            chunk=" ".join(sentence[i:i+chunk_size])
            chunks.append({
                "Page number": pagenumber,
                "Text": chunk
            })
    return chunks

def process_document(file):
    file_bytes = file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    existing_document = (document_exists_by_hash(file_hash))

    if existing_document:
        return (False,f"{file.name} has already been uploaded.")

    if file.type == "text/plain":
        text = load_text(file_bytes)
        chunks = chunk_text(text)
    elif file.type == "application/pdf":
        pages = loadpdf(file_bytes)
        chunks = chunk_pdf(pages)
    else:
        return (
            False,
            "Only PDF and TXT files are supported."
        )

    if not chunks:
        return (
            False,
            "No text could be extracted from this file."
        )
    document_id = file_hash
    storage_filename = file_hash
    storage_path = (
        f"{USER_ID}/{file_hash}_{file.name}"
    )
    supabase.storage \
        .from_(SUPABASE_STORAGE_BUCKET) \
        .upload(
            storage_path,
            file_bytes,
            {
                "content-type": file.type,
                "upsert": "false"
            }
        )
    texts = [
        chunk["Text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        show_progress_bar=False
    )

    # ids = []
    # documents = []
    # embedding_values = []
    # metadatas = []
    # for i, (chunk,embedding) in enumerate(zip(chunks,embeddings)):
    #     chunk_id = (f"{document_id}_{i}")
    #     ids.append(chunk_id)
    #     documents.append(chunk["Text"])
    #     embedding_values.append(embedding.tolist())
    #     metadatas.append({
    #         "document_id": document_id,
    #         "userId": USER_ID,
    #         "Filename": file.name,
    #         "Page number": chunk[
    #             "Page number"
    #         ]
    #     })

    # collection.add(
    #     ids=ids,
    #     documents=documents,
    #     embeddings=embedding_values,
    #     metadatas=metadatas
    # )
    # st.write("DEBUG: Chroma count after adding:", collection.count())

    supabase \
        .table("documents") \
        .insert({
            "id": document_id,
            "userId": USER_ID,
            "original_filename": file.name,
            "storage_path": storage_path,
            "storage_filename": storage_filename,
            "mime_type": file.type,
            "filesize": len(file_bytes)
        }) \
        .execute()
    chunk_rows=[]
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_id=f"{document_id}_{i}"
        chunk_rows.append({"id": chunk_id, "document_id": document_id, "userId": USER_ID, "content": chunk["Text"], "embedding": embedding.tolist(), "Filename": file.name, "Page number": chunk["Page number"]})

    supabase \
        .table("document_chunks") \
        .insert(chunk_rows) \
        .execute()

    return (
        True,
        f"Uploaded and processed {file.name}"
    )

def summarize_doc(file):
    documents = get_documents()
    if not documents:
        return "No documents uploaded"
    selected_doc = None
    for document in documents:
        if document["original_filename"].lower() == file.lower():
            selected_doc = document
            break
    if selected_doc is None:
        return "Could not find document"
    doc_id = selected_doc["id"]
    # coll = collection.get(where={"$and": [{"userId": USER_ID},{"document_id": doc_id}]})
    coll = collection.get()
    st.write("DEBUG selected document ID:", doc_id)
    st.write("DEBUG current USER_ID:", USER_ID)
    st.write("DEBUG Chroma count:", collection.count())
    st.write("DEBUG Chroma result:", coll)
    st.write("DEBUG Chroma result:", coll)
    if coll.get("metadatas"):
        st.write("DEBUG FIRST CHROMA METADATA:")
        st.write(coll["metadatas"][0])
        st.write("DEBUG Chroma result:", coll)
    chunks = coll.get("documents", [])
    if not chunks:
        return "No text in document"
    batches = []
    current_batch = ""
    MAX_BATCH_CHARS = 12000
    for chunk in chunks:
        if (len(current_batch) + len(chunk)> MAX_BATCH_CHARS and current_batch):
            batches.append(current_batch)
            current_batch = ""
        current_batch += chunk + "\n\n"
    if current_batch:
        batches.append(current_batch)
    st.write("DEBUG number of summary batches:", len(batches))
    partial_summaries = []
    for i, batch in enumerate(batches):
        response = groq_client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=[{
                    "role": "system",
                    "content": """
                    Summarize ONLY the provided document text.
                    
                    Include:
                    - Main topics
                    - Important concepts
                    - Important definitions
                    - Important formulas or facts
                    - Important examples when present
                    
                    Do not add outside knowledge.
                    Do not invent information.
                    Keep the summary concise.
                    """
                },
                {
                    "role": "user",
                    "content": batch
                }
            ],
            reasoning_format="hidden"
        )
        summary = response.choices[0].message.content
        partial_summaries.append(summary)
        st.write(f"DEBUG completed summary batch {i + 1}/{len(batches)}")
    combined_summary = "\n\n".join(partial_summaries)
    MAX_FINAL_CHARS = 12000
    if len(combined_summary) > MAX_FINAL_CHARS:
        combined_summary = combined_summary[:MAX_FINAL_CHARS]
    response = groq_client.chat.completions.create(
        model="qwen/qwen3.6-27b",
        messages=[
            {
                "role": "system",
                "content": """
                Create a final organized summary from the provided
                partial summaries.
                
                Include:
                - Main topics
                - Important concepts
                - Important definitions
                - Important formulas or facts
                - Important examples when present
                
                Use ONLY the provided summaries.
                Do not add outside knowledge.
                Do not invent information.
                Remove repetition.
                Keep the final summary concise and organized.
                """
            },
            {
                "role": "user",
                "content": combined_summary
            }
        ],
        reasoning_format="hidden"
    )
    final_summary= response.choices[0].message.content
    st.write("DEBUG final summary")
    st.write(repr(final_summary))
    return final_summary

chats = get_chats()

if not chats:
    new_chat = create_chat()
    chats = [new_chat]

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = chats[0]["id"]

chat_ids = [chat["id"] for chat in chats]

if st.session_state.current_chat_id not in chat_ids:
    st.session_state.current_chat_id = chats[0]["id"]

current_chat_id = st.session_state.current_chat_id

with st.sidebar:
    st.title("All Chats")

    if st.button(
        "New Chat",
        use_container_width=True
    ):
        new_chat = create_chat()

        st.session_state.current_chat_id = (
            new_chat["id"]
        )

        st.rerun()

    st.divider()

    chats = get_chats()

    for chat in chats:
        chat_id = chat["id"]
        title = chat["title"]

        is_current = (
            chat_id
            == st.session_state.current_chat_id
        )

        button_label = (
            f"{title}"
            if is_current
            else title
        )

        if st.button(
            button_label,
            use_container_width=True,
            key=f"chat_{chat_id}"
        ):
            st.session_state.current_chat_id = chat_id
            st.rerun()

    st.divider()

    st.title("Documents")

    documents = get_documents()

    if documents:
        for document in documents:
            st.write(
                f"{document['original_filename']}"
            )

            filesize = document.get(
                "filesize",
                0
            )

            filesize_kb = filesize / 1024

            st.caption(
                f"{filesize_kb:.1f} KB"
            )

st.subheader("Upload a document")

files = st.file_uploader(
    "Upload a PDF or TXT file",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

if files:
    for file in files:
        try:
            with st.spinner(
                f"Processing {file.name}..."
            ):
                added, message = process_document(file)

            if added:
                st.success(message)
            else:
                st.info(message)

        except Exception as e:
            st.error(
                f"Error processing {file.name}: {e}"
            )

#
if "summary" not in st.session_state:
    st.session_state.summary = ""
st.subheader("Summarize a document")
documents = get_documents()
if documents:
    filenames = [
        document["original_filename"]
        for document in documents
    ]
    selected_file = st.selectbox(
        "Select a document",
        filenames
    )
    if st.button("Summarize"):
        try:
            st.session_state.summary = summarize_doc(
                selected_file
            )
            st.success("Summary generated")
        except Exception as e:
            st.error(f"Could not generate summary: {e}")
            st.exception(e)
    if st.session_state.summary:
        st.subheader("Summary")
        st.write(st.session_state.summary)
        # st.write(f"Summary length: {len(st.session_state.summary)}")
else:
    st.info("Upload a document first.")
#

chat_tokens = get_chat_token_count(
    current_chat_id
)

if chat_tokens >= MAX_TOKENS:
    st.error(
        "Token limit reached. "
        "Please create a new chat."
    )
    st.stop()

messages = get_messages(
    current_chat_id
)

for message in messages:
    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )

query = st.chat_input(
    "Ask something about the document..."
)
if query:
    user_tokens = count_tokens(query)

    if chat_tokens + user_tokens >= MAX_TOKENS:
        st.error(
            "This message would exceed the token limit. "
            "Please create a new chat."
        )
        st.stop()

    save_message(
        current_chat_id,
        "user",
        query
    )

    with st.chat_message("user"):
        st.markdown(query)

    documents = get_documents()

    if not documents:
        answer = (
            "I could not find that information "
            "because you have not uploaded a document yet."
        )
    else:
        collection_count = collection.count()

        if collection_count == 0:
            answer = (
                "I could not find that information "
                "in the documents."
            )
        else:
            k = min(5, collection_count)

            query_embedding = model.encode([query])[0]

            results = collection.query(
                query_embeddings=[
                    query_embedding.tolist()
                ],
                n_results=k
            )

            chunks = results["documents"][0]
            distances = results["distances"][0]
            metadata = results["metadatas"][0]

            # distance is used to determine how close the result/answer is to the query
            if not chunks or distances[0] > 3:
                answer = (
                    "I could not find that information "
                    "in the documents."
                )
            else:
                context = ""

                for chunk, data in zip(
                    chunks,
                    metadata
                ):
                    context += (
                        f"[Page {data['Page number']}]\n"
                        f"{chunk}\n"
                        f"Filename: {data['Filename']}\n\n"
                    )
                    prompt = f"""
                        You are a retrieval-based chatbot.

                        Answer ONLY from the provided context.

                        If the answer is not present in the context, say:
                        "I could not find that information in the documents."

                        Mention page numbers when relevant.

                        Do not use outside knowledge.
                        Do not make assumptions.
                        Keep answers concise.

                        Context:
                        {context}

                        Question:
                        {query}
                        """
                response = groq_client.chat.completions.create(
                    # model="llama-3.1-8b-instant",
                    model ="qwen/qwen3.6-27b",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    reasoning_format="hidden"
                )

                answer = response.choices[0].message.content
        save_message(current_chat_id, "assistant", answer )
        
        with st.chat_message("assistant"):
            st.markdown(answer)

