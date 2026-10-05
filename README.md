A retrieval-augmented generation (RAG) chatbot built with Streamlit, Sentence Transformers, PGVector, Supabase, and Groq.
The application allows users to upload PDF or TXT documents and ask questions about their contents. Relevant document chunks are retrieved using semantic similarity and provided to an LLM to generate answers.

### Tech Stack
Streamlit-> Web application UI
Supabase-> Database and file storage
PGVector-> Vector database for document embeddings
Sentence Transformers-> Generate document/query embeddings
Groq-> LLM inference
Llama 3.1 8B-> Answer generation
PyPDF-> PDF text extraction
NLTK-> Sentence tokenization
tiktoken-> Token counting

### Features
Upload PDF and TXT documents
Split documents into sentence-based chunks
Generate embeddings using all-MiniLM-L6-v2
Retrieve relevant document chunks using PGVector
Generate answers using Groq and Llama 3.1
Create and switch between multiple chats
Persist chats and messages using Supabase
Store uploaded documents in Supabase Storage
Store document metadata in Supabase
Track chat token usage
Enforce a per-chat token limit
Include document filenames and page numbers in retrieved context

### Agents
DocumentAgent:
Message from the user goes to Groq API with a list of tools, the model answers the user or uses one of the tools. It stops after 6 rounds (steps) so as to not get stuck in a loop of trying to answer the users question. 
We use this agent to list all documents, search for specific documents, summarize a document, generate a quiz, list past quizzes, output quiz statistics
The summarization is done by splitting the chunks of a document, summarizing them and then merging them into one summary
Quiz agent:
We use this agent to generate quizzes for the user after the user selects a difficulty level and number of questions, default being 5 questions with medium difficulty. The quizzes are all stored in the database and are used by the user to analyse their progress.


## Supabase
# Tables
1. users
2. documents
3. document_chunks
4. chats
5. messages
6. quizzes
7. quiz_attempts

Documents uploaded are stored in a supabase bucket

### Architecture
Upload file -> Extract text and chunk -> Sentence transformer -> Make embeddings and store in chromadb -> User asks question -> Search embeddings -> Retrieve top relevant chunks -> use groq API to formulate the answer -> Output answer
