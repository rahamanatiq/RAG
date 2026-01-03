from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables 
load_dotenv()

# Initialize Gemini model 
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",         
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.1,
)

# Folders
UPLOAD_FOLDER = "data"
DB_FOLDER = "db"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DB_FOLDER, exist_ok=True)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=80,
    length_function=len,
    is_separator_regex=False,
)

# Prompt template
raw_prompt = ChatPromptTemplate.from_template(
    """ 
<s>[INST] You are a technical assistant good at searching documents. 
If you do not have an answer from the provided information, say so. [/INST] </s>
[INST] {input}
       Context: {context}
       Answer:
[/INST]
"""
)

# FastAPI app
app = FastAPI(title="Gemini Chat + PDF Upload API")

# Pydantic model
class QueryRequest(BaseModel):
    query: str

@app.get("/")
def root():
    return {"message": "Gemini PDF + LLM API is running! Endpoints: /upload_pdf, /ask_pdf, /ai"}

# PDF Upload & Indexing Endpoint
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Save file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        # Load PDF
        loader = PDFPlumberLoader(file_path)
        docs = loader.load()

        # Split text
        docs = text_splitter.split_documents(docs)

        # Create embeddings & FAISS vector store
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(DB_FOLDER)

        return {"message": f"{file.filename} uploaded and indexed successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")


# Query PDF & LLM Endpoint
@app.post("/ask_pdf")
async def ask_pdf_post(body: QueryRequest):
    query = body.query
    print(f"Query received: {query}")

    try:
        # Load FAISS
        vector_store = FAISS.load_local(
            DB_FOLDER,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )

        # Retrieve top docs
        docs = vector_store.similarity_search(query, k=5)  # removed score_threshold

        if not docs:
            return {
                "answer": "No relevant documents found for your query.",
                "sources": [],
                "retrieved_context": ""
            }

        # Combine context
        context = "\n\n".join([doc.page_content for doc in docs])

        # Format prompt
        prompt_input = raw_prompt.format(input=query, context=context)

        # Call LLM
        response = llm.invoke(prompt_input)
        answer = response.content.strip()

        return {"answer": answer}

    except Exception as e:
        print(f"Error in /ask_pdf: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# LLM Query Endpoint
@app.post("/ai")
async def ai_post(body: QueryRequest):
    try:
        response = llm.invoke(body.query)
        answer = response.content.strip()
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8080
    print(f"local server is running at: http://{host}:{port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)

