# Gemini Chat + PDF Upload API

A powerful Retrieval-Augmented Generation (RAG) API built with FastAPI, using Google's Gemini Pro model and FAISS vector store. This application allows users to upload PDF documents, index their content, and ask context-aware questions, as well as chat directly with the LLM.

## Features

*   **PDF Upload & Indexing**: Upload PDF files to automatically extract text, split into chunks, and create vector embeddings using `BAAI/bge-small-en-v1.5`.
*   **Context-Aware Q&A (RAG)**: Ask questions about the uploaded content. The system retrieves relevant context from the FAISS vector database and uses Gemini to generate accurate answers.
*   **Direct LLM Chat**: Interact directly with the Gemini model for general knowledge queries or creative tasks.
*   **Vector Search**: Utilizes FAISS (Facebook AI Similarity Search) for efficient and fast similarity searching of document chunks.

## Tech Stack

*   **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
*   **LLM**: [Google Gemini (gemini-2.5-flash)](https://deepmind.google/technologies/gemini/)
*   **Orchestration**: [LangChain](https://www.langchain.com/)
*   **Vector Store**: [FAISS](https://github.com/facebookresearch/faiss)
*   **Embeddings**: [HuggingFace (BAAI/bge-small-en-v1.5)](https://huggingface.co/BAAI/bge-small-en-v1.5)
*   **PDF Processing**: [PDFPlumber](https://github.com/jsvine/pdfplumber)

## Setup & Installation

### Prerequisites

*   Python 3.9+
*   A Google Cloud API Key with access to Gemini.

### Installation Steps

1.  **Clone the Repository**
    ```bash
    git clone <repository_url>
    cd <repository_folder>
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment Variables**
    Create a `.env` file in the root directory and add your Google API key:
    ```env
    GOOGLE_API_KEY=your_google_api_key_here
    ```

## Usage

1.  **Run the Server**
    ```bash
    python app.py
    # OR
    uvicorn app:app --reload
    ```
    The server will start at `http://127.0.0.1:8080`.

2.  **Access API Documentation**
    Open your browser and navigate to `http://127.0.0.1:8080/docs` to see the interactive Swagger UI.

### API Endpoints

*   **POST /upload_pdf**: Upload a PDF file to be indexed.
    *   *Input*: Multipart file upload.
    *   *Output*: Success message.

*   **POST /ask_pdf**: Ask a question based on uploaded PDFs.
    *   *Input*: JSON `{"query": "Your question here"}`
    *   *Output*: JSON `{"answer": "AI generated answer"}`

*   **POST /ai**: Chat directly with Gemini (no context).
    *   *Input*: JSON `{"query": "Your prompt here"}`
    *   *Output*: JSON `{"answer": "AI generated response"}`

## Project Structure

```
.
├── app.py               # Main FastAPI application
├── data/                # Directory for storing uploaded PDFs
├── db/                  # Directory for FAISS vector store indexes
├── .env                 # Environment variables (not committed)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## License

[MIT](LICENSE)
