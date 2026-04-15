from groq import Groq
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

client=Groq(api_key=GROQ_API_KEY)

loader = TextLoader("night_agent.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    docs,
    embedding=embedding
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

class GroqLLM:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)

    def __call__(self, prompt):
        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content


llm = GroqLLM(api_key="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")

while True:
    query = input("\nAsk something (or 'exit'): ")
    if query.lower() == "exit":
        break

    # Retrieve
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([d.page_content for d in docs])

    # Prompt
    prompt = f"""
Answer using ONLY this context:

{context}

Question: {query}
"""

    # Groq call
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    print("\nAnswer:\n", response.choices[0].message.content)
    print(f"API : {GROQ_API_KEY}")