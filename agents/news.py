from typing import List, Dict
import feedparser
import httpx
from langchain_openai import ChatOpenAI


def fetch_rss_entries(feed_urls: List[str], max_items_per_feed: int = 5) -> List[Dict]:
    entries: List[Dict] = []
    for url in feed_urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_items_per_feed]:
                entries.append(
                    {
                        "title": entry.get("title", ""),
                        "link": entry.get("link", ""),
                        "summary": entry.get("summary", ""),
                        "published": entry.get("published", ""),
                        "source": url,
                    }
                )
        except Exception:
            continue
    return entries


def fetch_article_text(url: str, timeout: float = 10.0) -> str:
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            resp = client.get(url)
            if resp.status_code == 200:
                return resp.text
    except Exception:
        return ""
    return ""


def summarize_news(entries: List[Dict], api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.2) -> str:
    if not entries:
        return "No news entries available."
    llm = ChatOpenAI(model=model, temperature=temperature, api_key=api_key)
    bullets = []
    for e in entries:
        bullets.append(f"- {e['title']} ({e['link']}) — {e['summary'][:300]}")
    prompt = (
        "Summarize today's news from the following bullet points into a concise brief with 3-5 bullets and a short headline."
        " Focus on facts. Include links inline when useful.\n\n" + "\n".join(bullets)
    )
    msg = [{"role": "user", "content": prompt}]
    resp = llm.invoke(msg)
    return resp.content if hasattr(resp, "content") else str(resp)


