from io import BytesIO

import tiktoken
from nltk.tokenize import sent_tokenize
from pypdf import PdfReader

encoding = tiktoken.get_encoding("gpt2")

def count_tokens(text):
    return len(encoding.encode(text))


def load_text(file_bytes):
    return file_bytes.decode("utf-8")


def loadpdf(file_bytes):
    reader = PdfReader(BytesIO(file_bytes))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        extracted_text = page.extract_text()
        if extracted_text:
            pages.append({
                "Page number": page_number,
                "Text": extracted_text
            })

    return pages


def chunk_text(text, chunk_size=5):
    sentences = sent_tokenize(text)
    chunks = []

    for i in range(0, len(sentences), chunk_size):
        chunks.append({
            "Page number": 0,
            "Text": " ".join(sentences[i:i + chunk_size])
        })

    return chunks


def chunk_pdf(pages, chunk_size=5):
    chunks = []

    for page in pages:
        page_number = page["Page number"]
        sentences = sent_tokenize(page["Text"])

        for i in range(0, len(sentences), chunk_size):
            chunks.append({
                "Page number": page_number,
                "Text": " ".join(sentences[i:i + chunk_size])
            })

    return chunks

def get_chats(supabase, USER_ID):
    response = supabase.table("chats").select("*").eq("userId", USER_ID).order("created_at").execute()
    return response.data or []


def create_chat(supabase, USER_ID):
    chats = get_chats(supabase, USER_ID)
    title = f"Chat {len(chats) + 1}"
    response = supabase.table("chats").insert({"userId": USER_ID, "title": title}).execute()
    return response.data[0]


def get_messages(supabase, chat_id):
    response = (
        supabase.table("messages")
        .select("id, created_at, chat_id, role, content")
        .eq("chat_id", chat_id)
        .order("created_at")
        .execute()
    )
    return response.data or []


def get_chat_token_count(supabase, chat_id):
    total = 0
    for message in get_messages(supabase, chat_id):
        if message["role"] == "quiz":
            continue
        total += count_tokens(message["content"])
    return total


def get_documents(supabase, USER_ID):
    response = (
        supabase.table("documents")
        .select("*")
        .eq("userId", USER_ID)
        .order("uploaded_at", desc=True)
        .execute()
    )
    return response.data or []


def get_document_by_id(supabase, USER_ID, document_id):
    response = (
        supabase.table("documents")
        .select("*")
        .eq("id", document_id)
        .eq("userId", USER_ID)
        .limit(1)
        .execute()
    )
    return response.data[0] if response.data else None


def document_exists_by_hash(supabase, USER_ID, file_hash):
    for document in get_documents(supabase, USER_ID):
        if document.get("storage_filename") == file_hash:
            return document
    return None


def insert_chunk_failed(supabase, USER_ID, SUPABASE_STORAGE_BUCKET, document_id, storage_path):
    """Roll back a partial upload. Returns the names of steps that failed."""
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
    return failed