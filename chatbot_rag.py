import streamlit as st
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from pinecone import Pinecone
from streamlit_pdf_viewer import pdf_viewer
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set wide layout to accommodate the 3-panel system comfortably
st.set_page_config(layout="wide", page_title="Nzelu AI", page_icon="")

load_dotenv()

# --- INITIALIZE CORE INFRASTRUCTURE ---
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview", 
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Initialize Session State Variables
if "library" not in st.session_state:
    st.session_state.library = {}  # Format: { "filename.pdf": bytes_data }
if "active_doc" not in st.session_state:
    st.session_state.active_doc = None
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {} # Format: { "filename.pdf": [messages] }

# --- PANEL 1: LEFT SIDEBAR (LIBRARY & MANAGEMENT) ---
with st.sidebar:
    st.title("Nzelu AI")
    
    # Custom Styled Upload Button
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        if uploaded_file.name not in st.session_state.library:
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                # Read bytes for rendering and text for vector database
                bytes_data = uploaded_file.read()
                st.session_state.library[uploaded_file.name] = bytes_data
                
                # Extract text
                pdf_reader = PdfReader(uploaded_file)
                raw_text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
                
                if raw_text.strip():
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_text(raw_text)
                    
                    # Core Engineering fix: Pass metadatas so documents don't blend together!
                    metadatas = [{"source": uploaded_file.name} for _ in chunks]
                    vector_store.add_texts(texts=chunks, metadatas=metadatas)
                    
                    st.session_state.active_doc = uploaded_file.name
                    st.success("Indexed successfully!")
                else:
                    st.error("Unreadable text formatting.")

    st.write("---")
    st.subheader("YOUR LIBRARY")
    
    # Render library items as interactive selection links
    if st.session_state.library:
        for doc_name in st.session_state.library.keys():
            # Bold the currently selected document
            label = f" **{doc_name}**" if doc_name == st.session_state.active_doc else f"📄 {doc_name}"
            if st.button(label, key=doc_name, use_container_width=True):
                st.session_state.active_doc = doc_name
                st.rerun()
    else:
        st.info("No documents uploaded yet.")

# --- THE THREE-PANEL CORE LAYOUT ---
if st.session_state.active_doc:
    # Split the main viewport into two large matching blocks: Center (PDF) and Right (Chat)
    col_pdf, col_chat = st.columns([1.1, 0.9], gap="large")
    
    # --- PANEL 2: CENTER VIEWPORT (PDF READER) ---
    with col_pdf:
        st.subheader(f"Viewing: {st.session_state.active_doc}")
        # Fetch the cached local byte arrays and display them frame-perfect
        pdf_bytes = st.session_state.library[st.session_state.active_doc]
        
        # Render clean native scrollable PDF view space
        pdf_viewer(input=pdf_bytes, height=750)

    # --- PANEL 3: RIGHT VIEWPORT (STATEFUL CONTEXTUAL CHAT) ---
    with col_chat:
        st.markdown("Stay Curious")
        st.caption("Continue asking questions.")
        
        # Isolate history per document so switching papers doesn't pollute the chat window
        active_doc = st.session_state.active_doc
        if active_doc not in st.session_state.chat_histories:
            st.session_state.chat_histories[active_doc] = [
                SystemMessage(content="You are Nzelu AI. Use the provided context to answer the user's questions perfectly.")
            ]
            
        current_history = st.session_state.chat_histories[active_doc]
        
        # Container to hold chat history for clean scrolling layout
        chat_container = st.container(height=550)
        with chat_container:
            for message in current_history:
                if isinstance(message, HumanMessage):
                    with st.chat_message("user"):
                        st.markdown(message.content)
                elif isinstance(message, AIMessage):
                    with st.chat_message("assistant"):
                        st.markdown(message.content)
                        
        # User input execution thread
        prompt = st.chat_input("Ask a question about your PDF...")
        
        if prompt:
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
            current_history.append(HumanMessage(content=prompt))
            
            # 1. INITIALIZE VARIABLES SECURELY OUTSIDE THE TRY BLOCK
            docs_text = "" 
            
            # Setup metadata-strict Vector search
            retriever = vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 3, 
                    "score_threshold": 0.25,
                    "filter": {"source": active_doc} 
                },
            )
            
            # 2. RUN SEARCH AND EXTRACT CONTEXT TEXT
            try:
                docs = retriever.invoke(prompt)
                docs_text = "".join(d.page_content for d in docs)
            except Exception as e:
                # Fallback if index retrieval experiences a hiccup
                docs_text = ""
                
            # 3. COMPILE SYSTEM INSTRUCTIONS (Line 155 is perfectly safe now!)
            # --- MAKE SURE ALL OF THIS IS INDENTED EQUALLY INSIDE "if prompt:" ---
            system_prompt = f"""You are Nzelu AI, an assistant for question-answering tasks. you were developed by a group of A-level cambridge students at Crucible Lusaka.
Use the following pieces of retrieved context from '{active_doc}' to answer the question. 
If you don't know the answer based on the context, state clearly that the document doesn't explicitly mention it.
Keep answers precise and clean. Make sure to add questions or promts that provoke the user to ask more questions and what to learn more. if the user asks, create real world scenarios and uses of a concept to better explain it.

Context:
{docs_text}"""

            llm = GoogleGenerativeAI(
                model="gemini-2.5-flash", 
                temperature=0.5, 
                google_api_key=os.environ.get("GOOGLE_API_KEY")
            )

            # Prepend the SystemMessage to the payload securely
            payload = [SystemMessage(content=system_prompt)] + [msg for msg in current_history if not isinstance(msg, SystemMessage)]
            
            with chat_container:
                with st.chat_message("assistant"):
                    response_obj = llm.invoke(payload)
                    result = response_obj.content if hasattr(response_obj, 'content') else response_obj
                    st.markdown(result)
            
            current_history.append(AIMessage(content=result))
            st.session_state.chat_histories[active_doc] = current_history
            st.rerun()
else:
    st.info("Welcome! Please upload a PDF document in the sidebar to begin analyzing your workspace library.")