import streamlit as st
import os
import random
from dotenv import load_dotenv
from pypdf import PdfReader
from pinecone import Pinecone
from streamlit_pdf_viewer import pdf_viewer

from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Force global configuration matrix
st.set_page_config(layout="wide", page_title="Nzelu AI", page_icon="")

load_dotenv()

# --- INITIALIZE CORE SESSION STATES ---
if "library" not in st.session_state:
    st.session_state.library = {}
if "active_doc" not in st.session_state:
    st.session_state.active_doc = None
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}
if "awaiting_response" not in st.session_state:
    st.session_state.awaiting_response = False
if "explorer_collapsed" not in st.session_state:
    st.session_state.explorer_collapsed = False
if "landing_seed" not in st.session_state:
    st.session_state.landing_seed = random.randint(0, 5)

# ---THE ELITE IDE APPARATUS (UNIFIES CANVAS & FIXES SPACING GAPS) ---
st.markdown("""
    <style>
        /* Unify global viewport backgrounds to eradicate split canvas bugs */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"], [data-testid="stHeader"] {
            background-color: #0b0d10 !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow-x: hidden;
        }
        
        /* Wipe structural boundaries clean across the frame */
        [data-testid="stMainBlockContainer"] {
            max-width: 100vw !important;
            min-height: 100vh !important;
            padding: 0 !important;
        }
        
        [data-testid="stHorizontalBlock"] {
            gap: 0px !important;
            margin: 0 !important;
            padding: 0 !important;
            background-color: #0b0d10 !important;
        }

        /* Target ONLY top-level sidebar columns to protect internal button columns from layout warping */
        .st-emotion-cache-bho867, [data-testid="stMainBlockContainer"] > div > div > div > div[data-testid="stColumn"] {
            min-height: 100vh !important;
            margin: 0 !important;
            padding: 1.25rem 1rem !important;
        }

        /* Tighten vertical gap explicitly between header elements and upload box */
        [data-testid="stFileUploader"] {
            margin-top: 4px !important;
            margin-bottom: 8px !important;
            padding-top: 0 !important;
        }
        
        [data-testid="stFileUploaderDropzone"] {
            padding: 12px 8px !important;
        }

        /* Strip Native Streamlit Container Borders */
        [data-testid="element-container"] div[data-testid="stContainerBorder"] {
            border: none !important;
            background-color: transparent !important;
            padding: 0 !important;
        }

        /* PANEL 1: File Explorer Column Frame */
        [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(1) {
            background-color: #11141a !important;
            border-right: 1px solid #1e2330 !important;
        }

        /* PANEL 2: Workspace Viewport Frame */
        [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2) {
            background-color: #0d0f13 !important;
        }
        
        /* Absolute Position Overrides for Chat Terminals */
        .stChatFloatingInputContainer {
            position: absolute !important;
            bottom: 24px !important;
            left: 4% !important;
            width: 92% !important;
            background-color: transparent !important;
        }

        /* Minimalist Scroll Trackers */
        ::-webkit-scrollbar {
            width: 4px;
            height: 4px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: #1e2330;
            border-radius: 2px;
        }
    </style>
""", unsafe_allow_html=True)

# --- INITIALIZE CORE VECTOR DATA PIPELINES ---
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview", 
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# --- COMPUTE INTERACTIVE GRID CONFIGURATIONS DYNAMICALLY ---
if st.session_state.active_doc:
    if st.session_state.explorer_collapsed:
        col_explorer, col_workspace, col_ai_sidebar = st.columns([0.04, 0.58, 0.38])
    else:
        col_explorer, col_workspace, col_ai_sidebar = st.columns([0.18, 0.47, 0.35])
else:
    if st.session_state.explorer_collapsed:
        col_explorer, col_main = st.columns([0.04, 0.96])
    else:
        col_explorer, col_main = st.columns([0.18, 0.82])

# --- PANEL 1: EXPLORER COMPARTMENT ---
with col_explorer:
    if st.session_state.explorer_collapsed:
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        if st.button("▶", help="Expand File Explorer", use_container_width=True):
            st.session_state.explorer_collapsed = False
            st.rerun()
    else:
        # Header Action Bar
        hdr_left, hdr_right = st.columns([0.75, 0.25])
        with hdr_left:
            st.markdown("<p style='color:#94a3b8; font-size:12px; font-weight:700; letter-spacing:0.5px; margin-top:6px; margin-bottom:0;'>EXPLORER</p>", unsafe_allow_html=True)
        with hdr_right:
            if st.button("◀", help="Collapse Panel", key="collapse_btn", use_container_width=True):
                st.session_state.explorer_collapsed = True
                st.rerun()
                
        # Drag and Drop Upload Port
        uploaded_file = st.file_uploader("Upload Profile Context", type=["pdf"], label_visibility="collapsed")
        if uploaded_file is not None:
            if uploaded_file.name not in st.session_state.library:
                with st.spinner():
                    bytes_data = uploaded_file.read()
                    st.session_state.library[uploaded_file.name] = bytes_data
                    
                    pdf_reader = PdfReader(uploaded_file)
                    raw_text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
                    
                    if raw_text.strip():
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        chunks = text_splitter.split_text(raw_text)
                        metadatas = [{"source": uploaded_file.name} for _ in chunks]
                        vector_store.add_texts(texts=chunks, metadatas=metadatas)
                        st.session_state.active_doc = uploaded_file.name
                        st.rerun()

        # Action: Return to Dashboard view
        if st.session_state.active_doc:
            if st.button("X", use_container_width=True):
                st.session_state.active_doc = None
                st.rerun()

        st.markdown("<div style='margin: 8px 0; border-top: 1px solid #1e2330;'></div>", unsafe_allow_html=True)
        
        # Indexed Document Cache Box
        explorer_box = st.container(height=450)
        with explorer_box:
            if st.session_state.library:
                for doc_name in st.session_state.library.keys():
                    is_active = doc_name == st.session_state.active_doc
                    label = f"📁 {doc_name[:18]}..." if is_active else f"📄 {doc_name[:18]}..."
                    if st.button(label, key=f"exp_{doc_name}", use_container_width=True):
                        st.session_state.active_doc = doc_name
                        st.rerun()
            else:
                st.markdown("<p style='color:#475569; font-size:12px; font-style:italic; padding-left:4px;'>No files initialized.</p>", unsafe_allow_html=True)

# --- WORKSPACE ROUTING PARSER ---
if st.session_state.active_doc:
    active_doc = st.session_state.active_doc
    
    # --- PANEL 2: ACTIVE DOCUMENT VIEWER ---
    with col_workspace:
        st.markdown(f"<p style='color:#64748b; font-size:11px; font-weight:600; letter-spacing:0.5px; margin-bottom:16px; margin-top:4px;'>WORKING CORE // {active_doc.upper()}</p>", unsafe_allow_html=True)
        pdf_bytes = st.session_state.library[active_doc]
        
        workspace_scroll_box = st.container(height=840)
        with workspace_scroll_box:
            pdf_viewer(input=pdf_bytes, width=700)

    # --- PANEL 3: DYNAMIC LLM SIDE CHAT BAR ---
    with col_ai_sidebar:
        st.markdown("<p style='color:#f8fafc; font-size:13px; font-weight:700; letter-spacing:0.5px; margin-bottom:20px; margin-top:2px;'>NZELU AI // CHAT CONSOLE</p>", unsafe_allow_html=True)
        
        if active_doc not in st.session_state.chat_histories:
            st.session_state.chat_histories[active_doc] = [
                SystemMessage(content="You are Nzelu AI. Rely heavily on text fragments explicitly pulled from context to formulate logical answers.")
            ]
        current_history = st.session_state.chat_histories[active_doc]
        
        chat_scroll_box = st.container(height=710)
        with chat_scroll_box:
            for message in current_history:
                if isinstance(message, HumanMessage):
                    with st.chat_message("user"):
                        st.markdown(message.content)
                elif isinstance(message, AIMessage):
                    with st.chat_message("assistant"):
                        st.markdown(message.content)
            
            # FIXED: Real-time chunk token streaming engine eliminates lag entirely
            if st.session_state.awaiting_response:
                user_prompt = current_history[-1].content
                docs_text = ""
                retriever = vector_store.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={"k": 4, "score_threshold": 0.22, "filter": {"source": active_doc}},
                )
                try:
                    docs = retriever.invoke(user_prompt)
                    docs_text = "".join(d.page_content for d in docs)
                except Exception:
                    docs_text = ""
                    
                system_prompt = f"""Rely strictly on this documentation to formulate an answer:\n{docs_text}"""
                
                # Using high-velocity, supported model
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key=os.environ.get("GOOGLE_API_KEY"))
                payload = [SystemMessage(content=system_prompt)] + [msg for msg in current_history if not isinstance(msg, SystemMessage)]
                
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    full_response = ""
                    # Stream tokens onto screen seamlessly
                    for chunk in llm.stream(payload):
                        full_response += chunk.content
                        placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)
                
                current_history.append(AIMessage(content=full_response))
                st.session_state.chat_histories[active_doc] = current_history
                st.session_state.awaiting_response = False
                st.rerun()

        if not st.session_state.awaiting_response:
            prompt = st.chat_input("Ask Nzelu AI anything...")
            if prompt:
                st.session_state.chat_histories[active_doc].append(HumanMessage(content=prompt))
                st.session_state.awaiting_response = True
                st.rerun()
else:
    # --- PLATFORM LANDING SPACE ---
    creative_welcomes = [
        "What's on the agenda today?",
        "What shall we decode today?",
        "Design without friction. What are we building?",
        "Context engine ready. Infuse your document schema.",
        "Knowledge synthesis active. Feed the blueprint.",
        "Non-conformity accelerates discovery. Introduce context."
    ]
    selected_welcome = creative_welcomes[st.session_state.landing_seed]
    
    with col_main:
        st.markdown(f"""<div style="display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 75vh; width: 100%; text-align: center; padding: 0 1rem;">
<h2 style="color: #ffffff; font-size: 32px; font-weight: 600; margin-bottom: 2.5rem; letter-spacing: -0.5px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">{selected_welcome}</h2>
<div style="width: 100%; max-width: 680px; position: relative; margin-bottom: 1.5rem;">
<div style="background-color: #1e2330; border: 1px solid #2d3444; border-radius: 28px; padding: 14px 24px; display: flex; align-items: center; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
<span style="color: #64748b; font-size: 16px; margin-right: 12px; font-family: sans-serif;">✨</span>
<input type="text" placeholder="Upload a document from the explorer to activate analysis workspace..." disabled style="background: transparent; border: none; width: 100%; color: #94a3b8; font-size: 15px; outline: none; cursor: not-allowed;" />
<span style="background-color: #2d3444; color: #64748b; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: bold; cursor: not-allowed;">↑</span>
</div>
</div>
<div style="display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; max-width: 680px;">
<div style="border: 1px solid #1e2330; color: #94a3b8; border-radius: 16px; padding: 8px 16px; font-size: 13px; font-weight: 500; background-color: #0d0f13; cursor: pointer;">🔍 Search deep data</div>
<div style="border: 1px solid #1e2330; color: #94a3b8; border-radius: 16px; padding: 8px 16px; font-size: 13px; font-weight: 500; background-color: #0d0f13; cursor: pointer;">📝 Write or edit briefs</div>
<div style="border: 1px solid #1e2330; color: #94a3b8; border-radius: 16px; padding: 8px 16px; font-size: 13px; font-weight: 500; background-color: #0d0f13; cursor: pointer;">💡 Analyze structures</div>
</div>
</div>""", unsafe_allow_html=True)