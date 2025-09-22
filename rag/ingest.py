from pathlib import Path
from typing import Iterable, List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def ensure_index(index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)


def _load_pdfs(paths: Iterable[Path]):
    for path in paths:
        loader = PyPDFLoader(str(path))
        for doc in loader.load():
            yield doc


def _split_docs(documents, chunk_size: int = 1000, chunk_overlap: int = 150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(list(documents))


def _get_embeddings(api_key: str):
    return OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")


def add_pdfs_to_index(pdf_paths: List[Path], index_dir: Path, api_key: str) -> None:
    ensure_index(index_dir)

    raw_docs = list(_load_pdfs(pdf_paths))
    if not raw_docs:
        return

    docs = _split_docs(raw_docs)
    embeddings = _get_embeddings(api_key)

    if (index_dir / "index.faiss").exists():
        vs = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
        vs.add_documents(docs)
        vs.save_local(str(index_dir))
    else:
        vs = FAISS.from_documents(docs, embeddings)
        vs.save_local(str(index_dir))


