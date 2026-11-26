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
Generate quizzes for the documents uploaded while picking the number of questions and the difficulty of those questions

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
## Simple question answer:
Upload file -> Extract text and chunk -> Sentence transformer -> Make embeddings and store in PGVector -> User asks question -> Search embeddings -> Retrieve top relevant chunks -> use groq API to formulate the answer -> Output answer
## Quiz generation for particular topic from uploaded document:
Upload file -> Extract text and chunk -> Sentence transformer -> Make embeddings and store in PGVector -> User asks for questions about a particular topic -> Retrieve top relevant chunks -> Use Groq API to make questions using -> Multiple choice quiz
## Quiz generation of uploaded file
Upload file -> Extract text and chunk -> Sentence transformer -> Make embeddings and store in PGVector -> Retrieve top relevant chunks -> Use Groq API to make questions using -> Multiple choice quiz


### How to run the application
## Pre-requisites
Ensure that you have the following installed on your system
- Python 3.10 or above
- Git
- Supabase account
- Groq API key
## Clone repository
git clone https://github.com/tanmayaande-456/RAGPractice/edit/Testing
## Install dependencies
pip install -r requirements.txt
## Configure environment variables
GROQ_API_KEY
## Run the application
streamlit rag.py

### Testing
Extract PDF text ->	Text is successfully extracted with the page numbers
Chunk documents -> Text is divided into sentence-based chunks without losing sentences.
Count tokens -> Token count is calculated correctly for the provided text.
Retrieve chats -> Users chats are retreived
Retrieve messages -> Messages in a chat are retrieved in the correct order.
Retrieve documents-> Documents are retrieved in descending order of upload date.
Retrieve a specific document - > The correct document is returned using its ID.
Handle nonexistent documents -> None is returned when the document does not exist.
Track chat token usage-> Token counts are calculated while excluding quiz messages.
Detect duplicate documents	-> Existing documents are identified using their storage filenames.
Handle nonexistent document hashes -> No document is returned when the hash does not exist
