from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import Chroma
# from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda
from sentence_transformers import CrossEncoder
import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, PointIdsList
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_ENDPOINT = os.getenv("QDRANT_ENDPOINT")

app = FastAPI()
folder_path = "db"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)


cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
llm = OllamaLLM(model="llama3.2")
fast_embedding = FastEmbedEmbeddings()

# llm = ChatOpenAI(model="gpt-4o-mini")
# fast_embedding = OpenAIEmbeddings(model="text-embedding-3-large")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=300, length_function=len, is_separator_regex=False
)

summary_prompt = PromptTemplate.from_template("""
    Ringkas pertanyaan berikut:
    Pertanyaan: {query}
""")
summarization_chain = summary_prompt | llm
#summarization_chain = LLMChain(llm=llm, prompt=summary_prompt)

raw_prompt = ChatPromptTemplate.from_template("""
    Anda adalah asisten AI bernama Emilia, seorang ahli dalam mata kuliah Struktur Data (jangan memperkenalkan diri setiap saat, hanya ketika pengguna ingin menanyakan nama Anda).
    Jawab berdasarkan dokumen yang diambil dan gunakan bahasa Indonesia.
    Jika konteks tidak mencukupi, gunakan pengetahuan Anda tentang Struktur Data untuk memberikan jawaban yang jelas dan akurat.    
    Pertanyaan: {input}
    Konteks: {context}
    Jawaban:
""")

client = QdrantClient(QDRANT_ENDPOINT, api_key=QDRANT_API_KEY)

# client.create_collection(
#     collection_name="test1_collection",
#     vectors_config=VectorParams(size=384, distance=Distance.COSINE),
# )

# client.create_collection(
#     collection_name="test1_collection",
#     vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
# )

vector_store = QdrantVectorStore(
    client=client,
    collection_name="test1_collection",
    embedding=fast_embedding,
)

@app.post("/ask_pdf")
async def ask_pdf(query: dict):
    print("Request /ask_pdf received")

    if not query:
        raise HTTPException(status_code=400, detail="Error request!")

    try:
        query_text = query.get("query")

        print("Original Query:", query_text)

        print("Loading vector database...")
        # vector_db = Chroma(persist_directory=folder_path, embedding_function=fast_embedding)

        print("Creating Chain...")
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 20, "lambda_mult": 0.2},
        )
        retrieved_docs = retriever.invoke(query_text)
        print(f"Retrieved {len(retrieved_docs)} documents.")

        query_doc_pairs = [(query_text, doc.page_content) for doc in retrieved_docs]
        scores = cross_encoder.predict(query_doc_pairs)
        ranked_docs = sorted(
            zip(retrieved_docs, scores),
            key=lambda x: x[1],  # Sort by score
            reverse=True         # Higher scores first
        )
        top_k_docs = [doc for doc, score in ranked_docs[:5]]
        context = "\n\n".join(doc.page_content for doc in top_k_docs)
        formatted_prompt = raw_prompt.format(input=query_text, context=context)
        result = llm.invoke(formatted_prompt)
        print("Result:", result)

        return JSONResponse(content={
            "message": "Query processed successfully",
            "answer": result,
            "context": context
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    print("Request /upload received")

    if not file:
        raise HTTPException(status_code=400, detail="No file part")

    try:
        file_name = file.filename
        save_path = f"uploads/{file_name}"
        with open(save_path, "wb") as buffer:
            buffer.write(file.file.read())
        print(f"File saved at {save_path}")

        loader = PDFPlumberLoader(save_path)
        docs = loader.load_and_split()
        print(f"Number of documents: {len(docs)}")

        chunks = text_splitter.split_documents(docs)
        print(f"Number of chunks: {len(chunks)}")

        # Chroma.from_documents(documents=chunks, embedding=fast_embedding, persist_directory=folder_path)

        QdrantVectorStore.from_documents(
            chunks,
            fast_embedding,
            url=QDRANT_ENDPOINT,
            api_key=QDRANT_API_KEY,
            prefer_grpc=True,
            collection_name="test1_collection",
        )

        return JSONResponse(content={
            "message": "File uploaded successfully",
            "filename": file_name,
            "documents": len(docs),
            "chunks": len(chunks)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete")
async def delete(filename: dict):
    print("Request /delete received")

    if not filename or "filename" not in filename:
        raise HTTPException(status_code=400, detail="Missing filename in request!")

    try:
        filename = filename["filename"]
        print(f"Deleting document and chunks for filename: {filename}")

        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="metadata.file_path",
                    match=MatchValue(value="uploads/"+filename)
                )
            ]
        )

        scroll_result = client.scroll(
            collection_name="test1_collection",
            scroll_filter=filter_condition,
            with_payload=True,
            with_vectors=False,
            limit=1000
        )

        vector_ids_to_delete = [record.id for record in scroll_result[0]]
        print(f"Found {len(vector_ids_to_delete)} chunks to delete for filename: {filename}")

        if not vector_ids_to_delete:
            return JSONResponse(content={'message': f"No chunks found for filename: {filename}"}, status_code=404)

        client.delete(
            collection_name="test1_collection",
            points_selector=PointIdsList(
                points=vector_ids_to_delete
            )
        )

        return JSONResponse(content={
            "message": f"Document and chunks deleted successfully for filename: {filename}",
            "deleted_chunks": len(vector_ids_to_delete)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=11436)
