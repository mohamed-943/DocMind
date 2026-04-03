import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv(override=True)
logo = './logo-mind.png'

# ── CONFIG ─────────────────────────────────────────
st.set_page_config(page_title="DocMind", layout="wide", page_icon=logo)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    background: #0f172a;
    color: #e5e7eb;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #020617;
    border-right: 1px solid #1e293b;
}

/* Chat container */
.chat-container {
    max-width: 800px;
    margin: auto;
    padding-bottom: 100px;
}

/* Messages */
.msg {
    display: flex;
    gap: 10px;
    margin-bottom: 15px;
}

.msg-user {
    justify-content: flex-end;
}

.bubble {
    padding: 12px 16px;
    border-radius: 14px;
    max-width: 70%;
    font-size: 14px;
    line-height: 1.5;
}

/* User bubble */
.user-bubble {
    background: #2563eb;
    color: white;
}

/* AI bubble */
.ai-bubble {
    background: #1e293b;
    border: 1px solid #334155;
}

/* Input bar fixed */
.input-container {
    position: fixed;
    bottom: 0;
    left: 20%;
    right: 0;
    padding: 15px;
    background: #020617;
    border-top: 1px solid #1e293b;
}

/* Input */
[data-testid="stTextInput"] input {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    color: white !important;
    padding: 12px !important;
}

/* Button */
[data-testid="stFormSubmitButton"] button {
    background: #2563eb !important;
    border-radius: 10px !important;
    border: none !important;
}

/* Title */
h1, h2, h3 {
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ── LLM ────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt_template = """
Answer based ONLY on this context:

<context>
{context}
</context>

Question: {input}
"""

# ── SESSION STATE ──────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ── SIDEBAR ────────────────────────────────────────
with st.sidebar:
    st.title(f"![](./logo-mind.png) DocMind")
    st.caption("Chat avec tes PDF")

    pdf_docs = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("⚡ Indexer"):
        if pdf_docs:
            with st.spinner("Indexation..."):
                content = ""

                for pdf in pdf_docs:
                    reader = PdfReader(pdf)
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            content += text

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )

                chunks = splitter.split_text(content)

                embeddings = OpenAIEmbeddings()

                vector_store = Chroma.from_texts(
                    chunks,
                    embeddings
                )

                st.session_state.retriever = vector_store.as_retriever()
                st.session_state.messages = []

            st.success("✅ Indexation terminée")

        else:
            st.warning("Upload un PDF")

# ── CHAT UI ────────────────────────────────────────
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

st.title("💬 Chat avec tes documents")

# afficher messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="msg msg-user">
            <div class="bubble user-bubble">{msg['content']}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="msg">
            <div class="bubble ai-bubble">{msg['content']}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ── INPUT FIXED ────────────────────────────────────
st.markdown("<div class='input-container'>", unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([8,1])

    with col1:
        user_question = st.text_input("Message...", label_visibility="collapsed")

    with col2:
        submitted = st.form_submit_button("➤")

st.markdown("</div>", unsafe_allow_html=True)

# ── LOGIC ──────────────────────────────────────────
if submitted and user_question:
    if not st.session_state.retriever:
        st.warning("⚠️ Upload et indexe un PDF d'abord")
    else:
        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })

        with st.spinner("🤖 réflexion..."):
            docs = st.session_state.retriever.invoke(user_question)

            context = "\n\n".join([d.page_content for d in docs])

            prompt = prompt_template.format(
                context=context,
                input=user_question
            )

            response = llm.invoke(prompt)
            answer = response.content

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        st.rerun()