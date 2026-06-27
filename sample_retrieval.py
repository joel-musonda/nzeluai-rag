# import basics
import os
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# initialize pinecone database
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# set the pinecone index
index_name = "sample-index"
index = pc.Index(index_name)

# initialize embeddings model + vector store using free Google Gemini
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", google_api_key=os.environ.get("GOOGLE_API_KEY"))
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Configure the retriever with a balanced similarity threshold
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.3},
)

print("Querying vector store...")
results = retriever.invoke("what did you have for breakfast?")

print("\nRESULTS:")
if not results:
    print("No documents matched the similarity score threshold.")
else:
    for res in results:
        print(f"* {res.page_content} [{res.metadata}]")