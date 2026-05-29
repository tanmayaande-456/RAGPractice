### WEEK 1:
DAY 1: AI, ML, DL, next-token prediction, prompting, training data
DAY 2: Hallucination, context windows, knowledge cutoffs
DAY 3: Prompts with constraints
DAY 4: API keys, requests, response, tokens
DAY 5: Sentence-transformers for vectors

### WEEK 2:
DAY 1: RAG
        Document: Text and PDF files are uploaded. Text and page numbers are extracted from the PDF and stored.
        Chunk: The text extracted is divided into chunks and tokenized, so that sentences aren't cut off.
        Embed: Model converts each chunk into a vector embedding.
        Store: The vectors are then stored in ChromaDB.
        Retrieve: A vector similarity search is conducted to retrieve the top 5 results. 
        Generate: The user asks a question for which it can only answer using the documents uploaded to it. If it cannot find the answer in the document, no answer is generated and it simply replies, "I could not find that information in the document."
DAY 2: ChromaDB
        The embeddings are stored in the vector database with the chunks of text. Once a query is given, the vector database is searched and the top 5 results are taken based on the distances.
DAY 3: A text file is uploaded, text is divided into chunks, embedded, stored in the vector database, retrieved later based on the query. The retrieved context is passed to GroqLLM and an answer is produced based on it.
DAY 4: Langchain
        TextLoader from langchain_community.document_loaders can be used to input the text files.
        RecursiveCharacterTextSplitter: Used to split the text into chunks with a value for overlap between them so that there is less chance of the LLM working with cut-off sentences from the chunks.
        Vector stores: The chunks are embedded and stored in the ChromaDB vector database
        RetrievalQA chain: Used to obtain top k results
DAY 5: RAG failures

### WEEK 3:
DAY 1: Loading and cleaning documents using PyMuPDF
DAY 2: Web scraping
DAY 3: Metadata
        File name, page number and text chunks are stored.
DAY 4: History
        st.session_state is used from streamlit to store the previous questions asked and remember the values.
DAY 5: Integration

### WEEK 4:
DAY 1: Guardrails
        The LLM is instructed to only answer questions based on the given context and not to stray from it. The page number from which the answer is taken is also displayed.
DAY 2: Streamlit
        Implements a working UI. 
DAY 3: Deployment
        Streamlit community cloud is used to deploy the app using a public github repo.
DAY 4: Evaluation
        Cosine similarity is used to make sure that the answer given by the LLM and the actual answer are actually similar, if not the same.
DAY 5: https://ragchatbot-byf3qjbrzhd7afcm3ixcq2.streamlit.app/
