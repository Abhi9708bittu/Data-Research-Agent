from typing import List, Dict
import httpx
import trafilatura
from langchain_openai import ChatOpenAI


def fetch_pages(urls: List[str], timeout: float = 12.0) -> List[Dict]:
    results: List[Dict] = []
    headers = {"User-Agent": "MarketResearchAgent/1.0"}
    for url in urls:
        try:
            with httpx.Client(follow_redirects=True, timeout=timeout, headers=headers) as client:
                r = client.get(url)
                if r.status_code == 200:
                    extracted = trafilatura.extract(r.text, include_comments=False, favor_recall=True) or ""
                    results.append({"url": url, "text": extracted[:20000]})
        except Exception:
            continue
    return results


def synthesize_competitor_brief(competitors: List[str], pages: List[Dict], api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.2) -> str:
    llm = ChatOpenAI(model=model, temperature=temperature, api_key=api_key)
    joined_pages = "\n\n".join([f"URL: {p['url']}\n{p['text']}" for p in pages])
    prompt = (
        "Create a concise market research brief covering the listed competitors. "
        "Include: positioning, key products, pricing cues (if present), strengths/weaknesses, and notable recent updates. "
        "Use bullet lists and cite URLs inline.\n\n"
        f"Competitors: {', '.join(competitors)}\n\n"
        f"Sources (raw):\n{joined_pages[:40000]}"
    )
    msg = [{"role": "user", "content": prompt}]
    resp = llm.invoke(msg)
    return resp.content if hasattr(resp, "content") else str(resp)


