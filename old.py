import streamlit as st
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import bs4
import fitz
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
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

FAISS_INDEX_PATH = "faiss_index"  # folder where index is saved on disk

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

def load_website(url):
    loader = WebBaseLoader(
        web_paths=[url],
        bs_kwargs={
            "parse_only": bs4.SoupStrainer(
                ["p", "table", "tr", "td", "th", "h1", "h2", "h3", "li","code"]
            )
        }
    )
    docs = loader.load()
    return [doc for doc in docs if doc.page_content.strip()]

def load_pdfs_from_folder(folder):
    all_docs = []
    if not os.path.exists(folder):
        return all_docs
    for pdf_file in [f for f in os.listdir(folder) if f.endswith(".pdf")]:
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
    """Load all sources, build FAISS index, save to disk."""
    with st.spinner("Building index for the first time — this won't happen again..."):
        urls = [
            "https://www.django-rest-framework.org/api-guide/requests/",
            "https://www.django-rest-framework.org/api-guide/responses/",
            "https://www.django-rest-framework.org/api-guide/views/",
            "https://www.django-rest-framework.org/api-guide/serializers/",
            "https://www.django-rest-framework.org/api-guide/authentication/",
            "https://www.django-rest-framework.org/api-guide/permissions/",
            "https://www.django-rest-framework.org/api-guide/throttling/",
            "https://www.django-rest-framework.org/api-guide/filtering/",
            "https://www.django-rest-framework.org/api-guide/pagination/",
            "https://www.django-rest-framework.org/api-guide/routers/",
            "https://www.django-rest-framework.org/api-guide/parsers/",
            "https://www.django-rest-framework.org/api-guide/generic-views/",
            "https://www.django-rest-framework.org/api-guide/renderers/",
            "https://www.django-rest-framework.org/api-guide/fields/",
            "https://www.django-rest-framework.org/api-guide/relations/",
            "https://www.django-rest-framework.org/api-guide/validators/",
            "https://www.django-rest-framework.org/api-guide/caching/",
            "https://www.django-rest-framework.org/api-guide/versioning/",
            "https://www.django-rest-framework.org/api-guide/content-negotiation/",
            "https://www.django-rest-framework.org/api-guide/metadata/",
            "https://www.django-rest-framework.org/api-guide/schemas/",
            "https://www.django-rest-framework.org/api-guide/format-suffixes/",
            "https://www.django-rest-framework.org/api-guide/reverse/",
            "https://www.django-rest-framework.org/api-guide/exceptions/",
            "https://www.django-rest-framework.org/api-guide/status-codes/",
            "https://www.django-rest-framework.org/api-guide/testing/",
            "https://www.django-rest-framework.org/api-guide/settings/"
        ]

        all_docs = []
        for url in urls:
            all_docs.extend(load_website(url))
        all_docs.extend(load_pdfs_from_folder("data"))

        if not all_docs:
            st.error("❌ No content extracted from any source")
            st.stop()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = splitter.split_documents(all_docs)

        embeddings = get_embeddings()
        vector_store = FAISS.from_documents(final_documents, embeddings)

        # Save to disk — next run will load from here instead
        vector_store.save_local(FAISS_INDEX_PATH)
        st.success(f"✅ Index built and saved — {len(final_documents)} chunks indexed")
        return vector_store

# ── Load or build index ───────────────────────────────────────────────────────
if "vector_store" not in st.session_state:
    embeddings = get_embeddings()

    if os.path.exists(FAISS_INDEX_PATH):
        # Index already exists on disk — just load it (fast)
        st.session_state.vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        st.success("✅ Index loaded from disk")
    else:
        # First run — build and save
        st.session_state.vector_store = build_and_save_index()

st.title("Django REST Framework RAG")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """Answer only based on the context below. If the answer is not contained
within the context, say you don't know. Answer only questions related to Django REST
Framework. If the question is not about Django REST Framework, say you don't know.
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

user_prompt = st.text_input("Ask a question about Django REST Framework:")

if user_prompt:
    start = time.process_time()
    response = chain.invoke(user_prompt)
    st.write(response)
    st.caption(f"Response time: {time.process_time() - start:.2f}s")

# Optional: button to force rebuild if sources change
with st.sidebar:
    st.header("Index Management")
    if st.button("🔄 Reload"):
        import shutil
        if os.path.exists(FAISS_INDEX_PATH):
            shutil.rmtree(FAISS_INDEX_PATH)
        if "vector_store" in st.session_state:
            del st.session_state["vector_store"]
        st.rerun()