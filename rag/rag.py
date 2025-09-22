from dataclasses import dataclass
from typing import List
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


@dataclass
class QAResult:
    answer: str
    sources: List


def _get_embeddings(api_key: str):
    return OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")


def load_retriever(index_dir: Path, api_key: str, k: int = 4):
    embeddings = _get_embeddings(api_key)
    vs = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
    return vs.as_retriever(search_kwargs={"k": k})


def answer_question(retriever, question: str, model: str, temperature: float, api_key: str) -> QAResult:
    llm = ChatOpenAI(model=model, temperature=temperature, api_key=api_key)

    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([d.page_content for d in docs])
    system = (
        "You are a helpful assistant that answers strictly based on the provided context. "
        "If the answer is not contained in the context, say you don't know. Provide concise answers."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    response = llm.invoke(messages)
    answer_text = response.content if hasattr(response, "content") else str(response)
    return QAResult(answer=answer_text, sources=docs)


