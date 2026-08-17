A retrieval-augmented generation (RAG) chatbot built with Streamlit, Sentence Transformers, ChromaDB, Supabase, and Groq.
The application allows users to upload PDF or TXT documents and ask questions about their contents. Relevant document chunks are retrieved using semantic similarity and provided to an LLM to generate answers.

### Tech Stack
Streamlit-> Web application UI
Supabase-> Database and file storage
ChromaDB-> Vector database for document embeddings
Sentence Transformers-> Generate document/query embeddings
Groq-> LLM inference
Llama 3.1 8B-> Answer generation
PyPDF-> PDF text extraction
NLTK-> Sentence tokenization
scikit-learn-> Similarity utilities
tiktoken-> Token counting

### Features
Upload PDF and TXT documents
Split documents into sentence-based chunks
Generate embeddings using all-MiniLM-L6-v2
Retrieve relevant document chunks using ChromaDB
Generate answers using Groq and Llama 3.1
Create and switch between multiple chats
Persist chats and messages using Supabase
Store uploaded documents in Supabase Storage
Store document metadata in Supabase
Track chat token usage
Enforce a per-chat token limit
Include document filenames and page numbers in retrieved context

## Supabase
# Tables
1. users
2. documents
3. chats
4. messages

Documents uploaded are stored in a supabase bucket

### Architecture
Upload file -> Extract text and chunk -> Sentence transformer -> Make embeddings and store in chromadb -> User asks question -> Search embeddings -> Retrieve top relevant chunks -> use groq API to formulate the answer -> Output answer
