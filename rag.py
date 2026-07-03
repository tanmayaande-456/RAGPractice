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

st.title("Chatbot")

encoding=tiktoken.get_encoding("gpt2")
def count_tokens(text):
    return len(encoding.encode(text))

if "chats" not in st.session_state:
    st.session_state.chats={
        "Chat 1": []
    }

if "current_chat" not in st.session_state:
    st.session_state.current_chat="Chat 1"

if "chat_tokens" not in st.session_state:
    st.session_state.chat_tokens = {
        "Chat 1": 0
    }

MAX_TOKENS = 100
current_chat = st.session_state.current_chat
if st.session_state.chat_tokens[current_chat] >= MAX_TOKENS:
    st.error(
        "Token limit reached. Please create a new chat."
    )
    st.stop()

files=st.file_uploader(
    "Upload a PDF or TXT file",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

def evaluate_answer(answer, ground_truth, model):
    answer_embedding = model.encode(answer)
    truth_embedding = model.encode(ground_truth)

    similarity = cosine_similarity(
        [answer_embedding],
        [truth_embedding]
    )[0][0]

    return similarity

def load_text(file):
    return file.read().decode("utf-8")

def loadpdf(file):
    read=PdfReader(file)
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
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

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

st.sidebar.title("All Chats")
for i in st.session_state):
    print(i)
if files:
    allchunks=[]
    for file in files:
        if file.type=="text/plain": 
            text=load_text(file)
            chunks=chunk_text(text)
        elif file.type=="application/pdf":
            pages=loadpdf(file)
            chunks=chunk_pdf(pages)
        else:
            st.error("Only upload text or PDF files")
            continue
        for chunk in chunks:
            chunk["Filename"]=file.name
            allchunks.append(chunk)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        client = chromadb.PersistentClient()
        try:
            collection = client.get_collection(name="rag_collection")
        except:
            collection=client.create_collection(name="rag_collection")
        # collection.delete(
        #     ids=collection.get()["ids"]
        # )

        ids=collection.get()["ids"]
        if ids:
            collection.delete(ids)

        for i, chunk_data in enumerate(allchunks):
            embedding = model.encode(chunk_data["Text"])
            collection.add(
                documents=[chunk_data["Text"]],
                embeddings=[embedding.tolist()],
                ids=[f"{chunk_data['Filename']}_{i}"],
                metadatas=[{
                    "Page number": chunk_data["Page number"],
                    "Filename": chunk_data["Filename"]
                }]
            )
        def retrieve(query, k=5):
            query_embedding = model.encode([query])[0]
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k
            )
            documents=results["documents"][0]
            distances=results["distances"][0]
            metadata=results["metadatas"][0]

            return documents, distances, metadata

        messages=st.session_state.chats[st.session_state.current_chat]

        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        query = st.chat_input("Ask something about the document")
        # user_tokens = count_tokens(query)
        if query:
            user_tokens=count_tokens(query)
            st.session_state.chat_tokens[current_chat]+=user_tokens

        # st.session_state.chat_tokens[
        #     st.session_state.current_chat
        # ] += user_tokens

        # st.session_state.chat_tokens[
        #     st.session_state.current_chat
        # ] += user_tokens
                
            messages.append({
                "role": "user",
                "content": query
            })
            with st.chat_message("user"):
                st.markdown(query)
            chunk, distance, metadata = retrieve(query)
            if distance[0] > 3:
                answer = "I could not find that information in the document."
            else:
                context = ""
                for ch, data in zip(chunk, metadata):
                    context += f"[Page {data['Page number']}]\n{ch}\nFilename: {data['Filename']}\n\n"
                prompt = f"""
                    You are a retrieval-based chatbot.

                    Answer ONLY from the provided context.

                    If the answer is not present in the context, say:
                    "I could not find that information in the document."

                    Mention page numbers when relevant.

                    Do not use outside knowledge.
                    Do not make assumptions.
                    Keep answers concise.

                    Context:
                    {context}

                    Question:
                    {query}
                    """
                groq_client = Groq(api_key=GROQ_API_KEY)
                response = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = response.choices[0].message.content
                answer_tokens = count_tokens(answer)

                st.session_state.chat_tokens[
                    st.session_state.current_chat
                ] += answer_tokens
            messages.append({
                "role": "assistant",
                "content": answer
            })
            with st.chat_message("assistant"):
                st.markdown(answer)
else:
    st.info("Please input a file")



















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
