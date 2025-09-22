import os
import tempfile
from pathlib import Path
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv

from rag.ingest import ensure_index, add_pdfs_to_index
from rag.rag import load_retriever, answer_question
from agents.news import fetch_rss_entries, summarize_news
from agents.market import fetch_pages, synthesize_competitor_brief
from agents.report import load_csv_preview, generate_report


APP_TITLE = "Document Analyzer RAG Agent"
INDEX_DIR = Path("E:/Chatbot/storage/faiss_index")


def require_api_key() -> Optional[str]:
    load_dotenv()
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.info(
            "Set OPENAI_API_KEY in Streamlit secrets or .env to use OpenAI models.",
            icon="ℹ️",
        )
    return api_key


def save_uploaded_files(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> List[Path]:
    temp_dir = Path(tempfile.mkdtemp(prefix="uploads_"))
    saved: List[Path] = []
    for uf in uploaded_files:
        target = temp_dir / uf.name
        with open(target, "wb") as f:
            f.write(uf.getbuffer())
        saved.append(target)
    return saved


def sidebar_controls():
    with st.sidebar:
        st.header("Settings")
        model = st.selectbox(
            "Model",
            options=["gpt-4o-mini", "gpt-4o", "gpt-4o-mini-translate"],
            index=0,
            help="OpenAI Chat model used for generation",
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        top_k = st.slider("Top-K Chunks", 1, 12, 4)
        if st.button("Reset Index", type="secondary"):
            if INDEX_DIR.exists():
                for p in INDEX_DIR.glob("**/*"):
                    try:
                        p.unlink()
                    except IsADirectoryError:
                        pass
                # Remove empty directories
                for p in sorted(INDEX_DIR.glob("**/*"), reverse=True):
                    if p.is_dir():
                        try:
                            p.rmdir()
                        except OSError:
                            pass
            st.session_state.pop("retriever_ready", None)
            st.success("Index cleared.")
        return model, temperature, top_k


def news_tab(api_key: str, model: str, temperature: float):
    st.subheader("News Summarizer")
    default_feeds = [
        "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",
        "https://feeds.feedburner.com/TechCrunch/",
        "https://www.theverge.com/rss/index.xml",
    ]
    feed_text = st.text_area("RSS feed URLs (one per line)", value="\n".join(default_feeds), height=120)
    max_items = st.slider("Max items per feed", 1, 15, 5)
    if st.button("Summarize News"):
        with st.spinner("Fetching and summarizing..."):
            feeds = [u.strip() for u in feed_text.splitlines() if u.strip()]
            entries = fetch_rss_entries(feeds, max_items_per_feed=max_items)
            summary = summarize_news(entries, api_key=api_key, model=model, temperature=temperature)
        st.markdown("**Daily Brief**")
        st.write(summary)
        with st.expander("Raw entries"):
            for e in entries:
                st.markdown(f"- [{e['title']}]({e['link']}) — {e['published']}")


def market_tab(api_key: str, model: str, temperature: float):
    st.subheader("Market Researcher")
    competitors = st.text_input("Competitor names (comma-separated)", placeholder="Acme, Wingify, Foobar")
    urls_text = st.text_area("Source URLs (one per line)", height=120)
    if st.button("Generate Brief"):
        with st.spinner("Collecting and analyzing sources..."):
            url_list = [u.strip() for u in urls_text.splitlines() if u.strip()]
            pages = fetch_pages(url_list)
            brief = synthesize_competitor_brief([s.strip() for s in competitors.split(",") if s.strip()], pages, api_key=api_key, model=model, temperature=temperature)
        st.markdown("**Competitor Brief**")
        st.write(brief)
        with st.expander("Fetched sources"):
            for p in pages:
                st.markdown(f"- {p['url']} ({len(p['text'])} chars)")


def report_tab(api_key: str, model: str, temperature: float):
    st.subheader("Report Generator")
    title = st.text_input("Report title", "Weekly Intelligence Report")
    objectives = st.text_area("Objectives", "Summarize key events and insights relevant to product and GTM.")
    notes = st.text_area("Analyst notes / highlights", "")
    csv_files = st.file_uploader("Optional: Upload CSVs (tables)", type=["csv"], accept_multiple_files=True)
    dataset_summaries = []
    if csv_files:
        st.markdown("**CSV Previews**")
        for f in csv_files:
            df = load_csv_preview(f.getvalue())
            dataset_summaries.append(f"CSV {f.name}: {list(df.columns)}; rows={len(df)}")
            st.caption(f"{f.name} — {len(df)} rows")
            st.dataframe(df.head(10))
    if st.button("Generate Report"):
        with st.spinner("Writing report..."):
            report = generate_report(title, objectives, dataset_summaries, notes, api_key=api_key, model=model, temperature=temperature)
        st.markdown("**Report**")
        st.write(report)


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="wide")
    st.title(APP_TITLE)
    st.caption("Upload PDFs, build a local FAISS index, and ask questions with citations.")

    api_key = require_api_key()
    model, temperature, top_k = sidebar_controls()

    tab_labels = ["Document Analyzer", "News", "Market", "Report"]
    tab_analyzer, tab_news, tab_market, tab_report = st.tabs(tab_labels)

    with tab_analyzer:
        uploaded = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Your documents are processed locally; embeddings are stored on disk.",
    )

    col_ingest, col_status = st.columns([1, 1])
    with col_ingest:
        if st.button("Ingest Documents", disabled=not uploaded):
            with st.spinner("Building index..."):
                ensure_index(INDEX_DIR)
                saved_files = save_uploaded_files(uploaded)
                add_pdfs_to_index(saved_files, INDEX_DIR, api_key)
                st.session_state["retriever_ready"] = True
            st.success("Ingestion complete.")

    with col_status:
        if INDEX_DIR.exists():
            size = sum(1 for _ in INDEX_DIR.glob("**/*"))
            st.metric("Index Files", size)

    if st.session_state.get("retriever_ready") or INDEX_DIR.exists():
        retriever = load_retriever(INDEX_DIR, api_key, k=top_k)
        st.divider()
        st.subheader("Ask a question")
        question = st.text_input("Your question", placeholder="What does the document say about X?")
        ask_clicked = st.button("Ask")
        if ask_clicked and question.strip():
            with st.spinner("Thinking..."):
                result = answer_question(retriever, question, model=model, temperature=temperature, api_key=api_key)
            st.markdown("**Answer**")
            st.write(result.answer)
            if result.sources:
                st.markdown("**Sources**")
                for i, src in enumerate(result.sources, start=1):
                    with st.expander(f"{i}. {src.metadata.get('source', 'document')} (p. {src.metadata.get('page', 'N/A')})"):
                        st.write(src.page_content)
    else:
        st.info("Upload PDFs and click Ingest Documents to get started.")

    with tab_news:
        news_tab(api_key, model, temperature)

    with tab_market:
        market_tab(api_key, model, temperature)

    with tab_report:
        report_tab(api_key, model, temperature)


if __name__ == "__main__":
    main()


