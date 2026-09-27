import os

import httpx
import streamlit as st


API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8765")

st.set_page_config(
    page_title="Advanced LLM RAG",
    page_icon="🔎",
    layout="wide",
)





def check_health() -> bool:
    try:
        response = httpx.get(f"{API_URL}/health", timeout=10)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


st.title("LLM RAG")

st.markdown(
    '<div class="subtext">'
    'Hybrid retrieval, reranking, grounded generation and validation.'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">
            Ask technical questions across a curated corpus of five research papers
        </div>
        <div class="hero-desc">
            Explore topics such as dense and sparse retrieval, passage reranking,
            sentence embeddings, RAG architectures and retrieval evaluation.
            <br><br>
            <strong>Pipeline:</strong> Dense retrieval (FAISS) + BM25 →
            Reciprocal Rank Fusion (RRF) → cross-encoder reranking →
            local LLM generation → citation, numeric grounding and groundedness validation.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
backend_ok = check_health()

if not backend_ok:
    st.error("Backend unavailable. Check that the FastAPI service is running and reachable.")

query = st.text_area(
    "Question",
    placeholder="Ask a question about the indexed documents...",
    height=120,
)

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

if st.button("Run RAG query", type="primary", disabled=not backend_ok):
    if not query.strip():
        st.warning("Enter a question first.")
    else:
        try:
            with st.spinner("Retrieving evidence and generating answer..."):
                response = httpx.post(
                    f"{API_URL}/query",
                    json={"query": query},
                    timeout=180,
                )
                response.raise_for_status()
                st.session_state["last_result"] = response.json()

        except httpx.HTTPStatusError as exc:
            st.error(f"Backend returned HTTP {exc.response.status_code}.")

        except httpx.HTTPError:
            st.error("Could not connect to the RAG backend.")


result = st.session_state["last_result"]

if result:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Answer")
    st.write(result["answer"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Validation")

    validation = result["validation"]
    metric_columns = st.columns(4)

    metrics = [
        ("Guardrails", validation["hard_guardrails_passed"]),
        ("Citations", validation["citations_valid"]),
        ("Numeric grounding", validation["numeric_grounding_valid"]),
        ("Grounded", validation["grounded"]),
    ]

    for col, (label, value) in zip(metric_columns, metrics):
        status_class = "metric-pass" if value else "metric-fail"
        status_text = "PASS" if value else "FAIL"

        col.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="{status_class}">{status_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Sources")
    st.markdown(
        '<div class="small-note">Retrieved evidence used to support the generated answer.</div>',
        unsafe_allow_html=True,
    )

    for source in result["sources"]:
        with st.expander(
            f'{source["citation"]} {source["source"]} — page {source["page"]}'
        ):
            st.write(f'**Chunk ID:** `{source["chunk_id"]}`')
            st.write(f'**Score:** {source["score"]}')

    st.markdown("</div>", unsafe_allow_html=True)