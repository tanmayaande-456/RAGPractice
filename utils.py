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
