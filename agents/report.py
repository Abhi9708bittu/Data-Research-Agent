from typing import List, Optional
import pandas as pd
from langchain_openai import ChatOpenAI


def load_csv_preview(csv_bytes: bytes, max_rows: int = 2000) -> pd.DataFrame:
    from io import BytesIO

    buf = BytesIO(csv_bytes)
    df = pd.read_csv(buf)
    if len(df) > max_rows:
        df = df.head(max_rows)
    return df


def generate_report(
    title: str,
    objectives: str,
    datasets_descriptions: List[str],
    findings_notes: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> str:
    llm = ChatOpenAI(model=model, temperature=temperature, api_key=api_key)
    prompt = (
        f"Produce a well-structured analytical report titled '{title}'.\n"
        "Include sections: Executive Summary, Objectives, Method (data sources), Key Findings, Charts to Consider, Limitations, and Next Steps.\n"
        f"Objectives:\n{objectives}\n\n"
        f"Data Sources (summaries):\n- " + "\n- ".join(datasets_descriptions) + "\n\n"
        f"Analyst Notes:\n{findings_notes}\n\n"
        "Write in concise, clear business language using bullet lists."
    )
    msg = [{"role": "user", "content": prompt}]
    resp = llm.invoke(msg)
    return resp.content if hasattr(resp, "content") else str(resp)


