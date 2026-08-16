import hashlib
from io import BytesIO
from supabase import create_client

import chromadb
# import os
# from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq
import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken

# load_dotenv()
# GROQ_API_KEY=os.getenv("GROQ_API_KEY")
GROQ_API_KEY=st.secrets["GROQ_API_KEY"]
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
SUPABASE_STORAGE_BUCKET = st.secrets["SUPABASE_STORAGE_BUCKET"]
USER_ID = st.secrets["USER_ID"]
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "rag_collection"

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


@st.cache_resource
def get_chroma_collection():
    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )
    return collection


supabase = get_supabase()
groq_client = get_groq()
model = get_embedding_model()
collection = get_chroma_collection()

st.title("Chatbot")

encoding=tiktoken.get_encoding("gpt2")
def count_tokens(text):
    return len(encoding.encode(text))

# if "chats" not in st.session_state:
#     st.session_state.chats={
#         "Chat 1": []
#     }

# if "current_chat" not in st.session_state:
#     st.session_state.current_chat="Chat 1"

# if "chat_tokens" not in st.session_state:
#     st.session_state.chat_tokens = {
#         "Chat 1": 0
#     }

MAX_TOKENS = 10000
# current_chat = st.session_state.current_chat
# if st.session_state.chat_tokens[current_chat] >= MAX_TOKENS:
#     st.error(
#         "Token limit reached. Please create a new chat."
#     )
#     st.stop()

# files=st.file_uploader(
#     "Upload a PDF or TXT file",
#     type=["txt", "pdf"],
#     accept_multiple_files=True
# )

# def evaluate_answer(answer, ground_truth, model):
#     answer_embedding = model.encode(answer)
#     truth_embedding = model.encode(ground_truth)

#     similarity = cosine_similarity(
#         [answer_embedding],
#         [truth_embedding]
#     )[0][0]

#     return similarity

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

# nltk.download('punkt')

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
        supabase.table("chats").select("*").eq("user_id", USER_ID).order("created_at").execute()
    )
    return response.data or []

def create_chat():
    chats=get_chats()
    chat_number=len(chats)+1
    title=f"Chat {chat_number}"
    response=(
        supabase.table("chats").insert({"user_id": USER_ID, "title": title}).execute()
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
        .eq("user_id", USER_ID)
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
        .eq("user_id", USER_ID)
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

    ids = []
    documents = []
    embedding_values = []
    metadatas = []
    for i, (chunk,embedding) in enumerate(zip(chunks,embeddings)):
        chunk_id = (f"{document_id}_{i}")
        ids.append(chunk_id)
        documents.append(chunk["Text"])
        embedding_values.append(embedding.tolist())
        metadatas.append({
            "document_id": document_id,
            "user_id": USER_ID,
            "Filename": file.name,
            "Page number": chunk[
                "Page number"
            ]
        })

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embedding_values,
        metadatas=metadatas
    )

    supabase \
        .table("documents") \
        .insert({
            "id": document_id,
            "user_id": USER_ID,
            "original_filename": file.name,
            "storage_path": storage_path,
            "storage_filename": storage_filename,
            "mime_type": file.type,
            "filesize": len(file_bytes)
        }) \
        .execute()

    return (
        True,
        f"Uploaded and processed {file.name}"
    )

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
    # if st.button("New Chat"):
    #     new_chat = f"Chat {len(st.session_state.chats)+1}"
    #     st.session_state.chats[new_chat] = []
    #     st.session_state.chat_tokens[new_chat] = 0
    #     st.session_state.current_chat = new_chat
    # for chat_name in list(st.session_state.chats.keys()):
    #     if st.button(chat_name, use_container_width=True, key=chat_name):
    #         st.session_state.current_chat = chat_name
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
                    model="llama-3.1-8b-instant",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                answer = response.choices[0].message.content
                answer_tokens = count_tokens(answer)

                save_message(
                    current_chat_id,
                    "assistant",
                    answer
                )

                with st.chat_message("assistant"):
                    st.markdown(answer)




# if files:
#     allchunks=[]
#     for file in files:
#         if file.type=="text/plain": 
#             text=load_text(file)
#             chunks=chunk_text(text)
#         elif file.type=="application/pdf":
#             pages=loadpdf(file)
#             chunks=chunk_pdf(pages)
#         else:
#             st.error("Only upload text or PDF files")
#             continue
#         for chunk in chunks:
#             chunk["Filename"]=file.name
#             allchunks.append(chunk)
#         model = SentenceTransformer("all-MiniLM-L6-v2")
#         client = chromadb.PersistentClient()
#         try:
#             collection = client.get_collection(name="rag_collection")
#         except:
#             collection=client.create_collection(name="rag_collection")
#         # collection.delete(
#         #     ids=collection.get()["ids"]
#         # )

#         ids=collection.get()["ids"]
#         if ids:
#             collection.delete(ids)

#         for i, chunk_data in enumerate(allchunks):
#             embedding = model.encode(chunk_data["Text"])
#             collection.add(
#                 documents=[chunk_data["Text"]],
#                 embeddings=[embedding.tolist()],
#                 ids=[f"{chunk_data['Filename']}_{i}"],
#                 metadatas=[{
#                     "Page number": chunk_data["Page number"],
#                     "Filename": chunk_data["Filename"]
#                 }]
#             )
#         def retrieve(query, k=5):
#             collection_count = (collection.count())
#             if collection_count == 0:
#                 return [], [], []
#             k = min(k,collection_count)
#             query_embedding = model.encode([query])[0]
#             results = collection.query(
#                 query_embeddings=[query_embedding.tolist()],
#                 n_results=k, where={"user_id", USER_ID}
#             )
#             documents=results["documents"][0]
#             distances=results["distances"][0]
#             metadata=results["metadatas"][0]

#             return documents, distances, metadata

#         messages=st.session_state.chats[st.session_state.current_chat]

#         for message in messages:
#             with st.chat_message(message["role"]):
#                 st.markdown(message["content"])
#         query = st.chat_input("Ask something about the document")
#         # user_tokens = count_tokens(query)
#         if query:
#             user_tokens=count_tokens(query)
#             st.session_state.chat_tokens[current_chat]+=user_tokens

#         # st.session_state.chat_tokens[
#         #     st.session_state.current_chat
#         # ] += user_tokens

#         # st.session_state.chat_tokens[
#         #     st.session_state.current_chat
#         # ] += user_tokens
                
#             messages.append({
#                 "role": "user",
#                 "content": query
#             })
#             with st.chat_message("user"):
#                 st.markdown(query)
#             chunk, distance, metadata = retrieve(query)
#             if distance[0] > 3:
#                 answer = "I could not find that information in the document."
#             else:
#                 context = ""
#                 for ch, data in zip(chunk, metadata):
#                     context += f"[Page {data['Page number']}]\n{ch}\nFilename: {data['Filename']}\n\n"
#                 prompt = f"""
#                     You are a retrieval-based chatbot.

#                     Answer ONLY from the provided context.

#                     If the answer is not present in the context, say:
#                     "I could not find that information in the document."

#                     Mention page numbers when relevant.

#                     Do not use outside knowledge.
#                     Do not make assumptions.
#                     Keep answers concise.

#                     Context:
#                     {context}

#                     Question:
#                     {query}
#                     """
#                 groq_client = Groq(api_key=GROQ_API_KEY)
#                 response = groq_client.chat.completions.create(
#                     model="llama-3.1-8b-instant",
#                     messages=[{"role": "user", "content": prompt}]
#                 )
#                 answer = response.choices[0].message.content
#                 answer_tokens = count_tokens(answer)

#                 st.session_state.chat_tokens[
#                     st.session_state.current_chat
#                 ] += answer_tokens
#             messages.append({
#                 "role": "assistant",
#                 "content": answer
#             })
#             with st.chat_message("assistant"):
#                 st.markdown(answer)
# else:
#     st.info("Please input a file")



















# model = SentenceTransformer("all-MiniLM-L6-v2")
# embeddings = model.encode(chunks)

# client = chromadb.PersistentClient()
# try:
#     collection = client.get_collection(name="rag_collection")
# except:
#     collection=client.create_collection(name="rag_collection")
#     # for i, chunk in enumerate(chunks):
#     #     collection.add(
#     #         documents=[chunk],
#     #         embeddings=[embeddings[i].tolist()],
#     #         ids=[str(i)]
#     #     )

# if collection.count() == 0:
#     embeddings = model.encode(chunks)
#     for i, chunk in enumerate(chunks):
#         collection.add(
#             documents=[chunk],
#             embeddings=[embeddings[i].tolist()],
#             ids=[str(i)]
#         )

# def retrieve(query, k=70):
#     query_embedding = model.encode([query])[0]
#     results = collection.query(
#         query_embeddings=[query_embedding.tolist()],
#         n_results=k
#     )
#     documents=results["documents"][0]
#     distances=results["distances"][0]

#     return documents, distances

# query=st.text_input("Question: ")

# if st.button("Ask"):
#     if (query.strip()==""):
#         st.warning("Please enter a question")
#     else:
#         retrieved_chunks, distances=retrieve(query)
#         print(distances[0])
#         print("\n\n")
#         if (distances[0]>1.2):
#             st.write("Could not find info\n")
#             st.stop()
#         context = "\n\n".join(retrieved_chunks)
#         prompt = f"""
#             You are a retrieval-based chatbot.

#             Answer ONLY from the provided context.

#             If the answer is not present in the context, say:
#             "I could not find that information in the document."

#             Do not use outside knowledge.
#             Do not make assumptions.
#             Keep answers concise.

#             Context:
#             {context}

#             Question:
#             {query}
#             """
#         client=Groq(api_key=GROQ_API_KEY)
#         response = client.chat.completions.create(
#         model="llama-3.1-8b-instant",
#             messages=[
#                 {"role": "user", "content": prompt}
#             ]
#         )
#         answer=response.choices[0].message.content
#         st.subheader("Answer:\n")
#         st.write(answer)
