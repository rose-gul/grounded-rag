"""Live demo UI. Shows the answer AND the retrieved sources side-by-side so a
viewer can see the grounding — the whole point of the project.

Run: streamlit run ui/streamlit_app.py   (needs the API running, or import the
pipeline directly once build_pipeline() is implemented).
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Grounded RAG", page_icon="🔎")
st.title("🔎 Grounded — RAG that shows its sources")
st.caption("Ask a question. Every factual claim is cited, or the assistant abstains.")

query = st.text_input("Your question", placeholder="How long do I have to request a refund?")

if st.button("Ask") and query:
    with st.spinner("Retrieving + generating…"):
        try:
            resp = httpx.post(f"{API_URL}/ask", json={"query": query}, timeout=60)
            data = resp.json()
        except Exception as e:  # noqa: BLE001
            st.error(f"API not reachable at {API_URL}: {e}")
            st.stop()

    if data.get("abstained"):
        st.warning(data["text"])
    else:
        st.markdown(f"**Answer:** {data['text']}")
        st.markdown(f"**Citations:** {', '.join(data.get('citations', [])) or '—'}")

    with st.expander("Retrieved sources"):
        for c in data.get("contexts", []):
            st.markdown(f"**[{c['source_id']}]** _{c['doc_id']}_ · score {c['score']:.2f}")
            st.write(c["text"])
