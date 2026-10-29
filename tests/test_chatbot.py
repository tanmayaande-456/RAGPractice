from utils import (
    count_tokens,
    load_text,
    chunk_text,
    chunk_pdf,
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

