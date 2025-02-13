from flask import Flask, request, jsonify
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from sentence_transformers import CrossEncoder
import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, PointIdsList

load_dotenv()

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_ENDPOINT = os.getenv("QDRANT_ENDPOINT")

app = Flask(__name__)
folder_path = "db"

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
# llm = OllamaLLM(model="llama3")
# fast_embedding = FastEmbedEmbeddings()

llm = ChatOpenAI(model="gpt-4o-mini")
fast_embedding = OpenAIEmbeddings(model="text-embedding-3-large")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=300, length_function=len, is_separator_regex=False
)

summary_prompt = PromptTemplate.from_template("""
    Ringkas pertanyaan berikut:
    Pertanyaan: {query}
""")
summarization_chain = LLMChain(llm=llm, prompt=summary_prompt)

raw_prompt = ChatPromptTemplate.from_template("""
    Anda adalah asisten AI bernama Emilia (jangan memperkenalkan diri setiap saat, hanya ketika pengguna ingin menanyakan nama Anda).
    Jawab berdasarkan dokumen yang diambil dan gunakan bahasa Indonesia
    Jika konteks tidak mencukupi, Anda dapat menjawab sesuai pengetahuan Anda
    Pertanyaan: {input}
    Konteks: {context}
    Jawaban:
""")

client = QdrantClient(QDRANT_ENDPOINT, api_key=QDRANT_API_KEY)

# client.create_collection(
#     collection_name="demo_collection",
#     vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
# )

vector_store = QdrantVectorStore(
    client=client,
    collection_name="demo_collection",
    embedding=fast_embedding,
)

@app.route("/ask_pdf", methods=["POST"])
def askPDF():
    print("Request /ask_pdf received")

    if not request.json:
        return jsonify({'error': 'Error request!'}), 400

    try:
        data = request.json
        query = data.get("query")

        print("Original Query:", query)

        print("Loading vector database...")
        # vector_db = Chroma(persist_directory=folder_path, embedding_function=fast_embedding)

        print("Creating Chain...")
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 20, "lambda_mult": 0.2},
        )
        retrieved_docs = retriever.get_relevant_documents(query)
        print(f"Retrieved {len(retrieved_docs)} documents.")

        query_doc_pairs = [(query, doc.page_content) for doc in retrieved_docs]
        scores = cross_encoder.predict(query_doc_pairs)
        ranked_docs = sorted(
            zip(retrieved_docs, scores),
            key=lambda x: x[1],  # Sort by score
            reverse=True         # Higher scores first
        )
        top_k_docs = [doc for doc, score in ranked_docs[:5]]
        context = "\n\n".join(doc.page_content for doc in top_k_docs)
        formatted_prompt = raw_prompt.format(input=query, context=context)
        result = llm.invoke(formatted_prompt)
        print("Result:", result)

        answer = jsonify({
            "message": "Query processed successfully",
            "answer": result.content,
            "context": context
        })
        return answer
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload():
    print("Request /upload received")

    if "file" not in request.files:
        return jsonify({'error': 'No file part'}), 400

    try:
        file = request.files["file"]
        file_name = file.filename
        save_path = f"uploads/{file_name}"
        file.save(save_path)
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
            collection_name="demo_collection",
        )

        return jsonify({
            "message": "File uploaded successfully",
            "filename": file_name,
            "documents": len(docs),
            "chunks": len(chunks)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/remove", methods=["POST"])
def remove():
    print("Request /remove received")

    if not request.json:
        return jsonify({'error': 'Error request!'}), 400

    try:
        data = request.json
        document_id = data.get("document_id")

        if not document_id:
            return jsonify({'error': 'Document ID is required'}), 400

        client.delete(
            collection_name="demo_collection",
            points_selector=PointIdsList(
                points=[document_id]
            )
        )

        return jsonify({
            "message": "Document deleted successfully",
            "document_id": document_id
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/delete", methods=["POST"])
def delete():
    print("Request /delete received")

    if not request.json or "filename" not in request.json:
        return jsonify({'error': 'Missing filename in request!'}), 400

    try:
        filename = request.json["filename"]
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
            collection_name="demo_collection",
            scroll_filter=filter_condition,
            with_payload=True,
            with_vectors=False,
            limit=1000
        )

        vector_ids_to_delete = [record.id for record in scroll_result[0]]
        print(f"Found {len(vector_ids_to_delete)} chunks to delete for filename: {filename}")

        if not vector_ids_to_delete:
            return jsonify({'message': f"No chunks found for filename: {filename}"}), 404

        client.delete(
            collection_name="demo_collection",
            points_selector=PointIdsList(
                points=vector_ids_to_delete
            )
        )

        return jsonify({
            "message": f"Document and chunks deleted successfully for filename: {filename}",
            "deleted_chunks": len(vector_ids_to_delete)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def start():
    app.run(host="127.0.0.1", port=11436, debug=True)

if __name__ == "__main__":
    start()
