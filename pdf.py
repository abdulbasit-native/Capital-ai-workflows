import streamlit as st
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import fitz
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import time
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found in .env file")
    st.stop()

FAISS_INDEX_PATH = "faiss_index"
PDF_FOLDER = "workflows"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )


def load_pdfs_from_folder(folder):
    all_docs = []
    if not os.path.exists(folder):
        st.warning(f"⚠️ PDF folder '{folder}' not found. Create it and add your PDFs.")
        return all_docs

    pdf_files = [f for f in os.listdir(folder) if f.endswith(".pdf")]
    if not pdf_files:
        st.warning(f"⚠️ No PDF files found in '{folder}'.")
        return all_docs

    for pdf_file in pdf_files:
        path = os.path.join(folder, pdf_file)
        doc = fitz.open(path)
        for page_num, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                all_docs.append(Document(
                    page_content=text,
                    metadata={"source": pdf_file, "page": page_num + 1}
                ))
    return all_docs


def build_and_save_index():
    """Load all PDFs, build FAISS index, save to disk."""
    with st.spinner("Building index from PDFs — this won't happen again..."):
        all_docs = load_pdfs_from_folder(PDF_FOLDER)

        if not all_docs:
            st.error("❌ No content extracted from PDFs. Add PDF files to the 'data' folder.")
            st.stop()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = splitter.split_documents(all_docs)

        embeddings = get_embeddings()
        vector_store = FAISS.from_documents(final_documents, embeddings)

        vector_store.save_local(FAISS_INDEX_PATH)
        st.success(f"✅ Index built and saved — {len(final_documents)} chunks from {len(set(d.metadata['source'] for d in all_docs))} PDF(s)")
        return vector_store


# ── Load or build index ───────────────────────────────────────────────────────
if "vector_store" not in st.session_state:
    embeddings = get_embeddings()

    if os.path.exists(FAISS_INDEX_PATH):
        st.session_state.vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        st.success("✅ Index loaded from disk")
    else:
        st.session_state.vector_store = build_and_save_index()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("📄 PDF RAG Assistant")
st.caption(f"Answering questions from PDFs in the `{PDF_FOLDER}/` folder")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """Answer only based on the context below. If the answer is not contained
within the context, say you don't know.
<context>
{context}
</context>"""),
    ("human", "{input}")
])

retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})

chain = (
    {"context": retriever, "input": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

user_prompt = st.text_input("Ask a question about your PDFs:")

if user_prompt:
    start = time.process_time()
    response = chain.invoke(user_prompt)
    st.write(response)
    st.caption(f"Response time: {time.process_time() - start:.2f}s")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Index Management")

    # Show indexed PDFs
    if os.path.exists(PDF_FOLDER):
        pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
        if pdf_files:
            st.subheader("Indexed PDFs")
            for f in pdf_files:
                st.text(f"• {f}")
        else:
            st.info("No PDFs found in 'workflows/' folder.")

    st.divider()

    if st.button("🔄 Rebuild Index"):
        import shutil
        if os.path.exists(FAISS_INDEX_PATH):
            shutil.rmtree(FAISS_INDEX_PATH)
        if "vector_store" in st.session_state:
            del st.session_state["vector_store"]
        st.rerun()