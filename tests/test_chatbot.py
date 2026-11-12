from utils import (
    count_tokens,
    load_text,
    chunk_text,
    chunk_pdf,
    get_chats,
    get_messages,
    get_documents,
    get_document_by_id,
    get_chat_token_count
)

def test_load_text_decodes_utf8():
    data = "Hello, student!".encode("utf-8")

    assert load_text(data) == "Hello, student!"


def test_count_tokens_returns_positive_count():
    assert count_tokens("Hello, student!") > 0


def test_chunk_text_preserves_all_sentences():
    text = (
        "This is RAG chatbot. "
        "Chunks are made from documents. "
        "Groq generates answers."
    )

    chunks = chunk_text(text, chunk_size=2)

    assert len(chunks) == 2
    assert chunks[0]["Page number"] == 0
    assert "This is RAG chatbot." in chunks[0]["Text"]
    assert "Groq generates answers." in chunks[1]["Text"]


def test_chunk_pdf_preserves_page_numbers():
    pages = [
        {
            "Page number": 3,
            "Text": "First sentence. Second sentence."
        }
    ]

    chunks = chunk_pdf(pages, chunk_size=1)

    assert len(chunks) == 2
    assert all(chunk["Page number"] == 3 for chunk in chunks)
    assert chunks[0]["Text"] == "First sentence."
    assert chunks[1]["Text"] == "Second sentence."

def test_get_chats_includes_seeded_chat(seeded):
    chats = get_chats(seeded["client"], seeded["user_id"])
    assert seeded["chat"]["id"] in [c["id"] for c in chats]
    assert all(c["userId"] == seeded["user_id"] for c in chats)


def test_get_messages_returns_seeded_messages_in_order(seeded):
    messages = get_messages(seeded["client"], seeded["chat"]["id"])
    assert [m["role"] for m in messages] == ["user", "assistant", "quiz"]
    assert messages[0]["content"] == "What is dynamic programming?"


def test_get_messages_unknown_chat_returns_empty_list(seeded):
    assert get_messages(seeded["client"], -1) == []


def test_get_documents_returns_seeded_documents_newest_first(seeded):
    documents = get_documents(seeded["client"], seeded["user_id"])
    seeded_ids = {d["id"] for d in seeded["documents"]}
    ours = [d["id"] for d in documents if d["id"] in seeded_ids]
    assert ours == [seeded["documents"][1]["id"], seeded["documents"][0]["id"]]


def test_get_document_by_id_returns_matching_document(seeded):
    expected = seeded["documents"][0]
    document = get_document_by_id(seeded["client"], seeded["user_id"], expected["id"])
    assert document["original_filename"] == expected["original_filename"]


def test_get_document_by_id_returns_none_when_missing(seeded):
    assert get_document_by_id(seeded["client"], seeded["user_id"], "does-not-exist") is None


def test_get_chat_token_count_skips_quiz_messages(seeded):
    expected = sum(count_tokens(m["content"]) for m in seeded["messages"] if m["role"] != "quiz")
    assert get_chat_token_count(seeded["client"], seeded["chat"]["id"]) == expected
