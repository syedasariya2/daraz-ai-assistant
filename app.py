app.py

Daraz Customer Support Operations Assistant
--------------------------------------------
A Streamlit chat app that answers questions using a PRE-BUILT FAISS index
(created separately by ingest.py). This app never re-embeds or re-processes
the source PDFs — it only loads faiss_index/index.faiss + metadata.json,
embeds the user's question, retrieves the closest chunks (optionally
restricted to one knowledge-base section), and asks a Groq-hosted LLM to
answer using only that retrieved context.

Required folder layout next to this file:
    faiss_index/
        index.faiss
        metadata.json

Required secret (Streamlit Cloud: Settings -> Secrets, or local
.streamlit/secrets.toml):
    GROQ_API_KEY = "your-groq-api-key"
"""

import json
import os

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
FAISS_DIR = "faiss_index"
INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
METADATA_PATH = os.path.join(FAISS_DIR, "metadata.json")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # must match the model used in ingest.py
GROQ_MODEL = "openai/gpt-oss-120b"

TOP_K = 5              # final number of chunks passed to the LLM
CANDIDATE_K = 25        # how many raw vector hits to pull before filtering by section

SECTIONS = [
    ("all", "All Sections"),
    ("returns", "Returns"),
    ("delivery", "Delivery"),
    ("refunds", "Refunds"),
    ("sellers", "Sellers"),
    ("payments", "Payments"),
    ("customer_support", "Customer Support"),
]

DARAZ_ORANGE = "#F85606"
DARAZ_DARK = "#1A1A1A"

# --------------------------------------------------------------------------
# Page setup + branding
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Daraz Support Assistant",
    page_icon="🛍️",
    layout="wide",
)

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: #FAFAFA;
        }}
        section[data-testid="stSidebar"] {{
            background-color: #FFFFFF;
            border-right: 1px solid #EEEEEE;
        }}
        .daraz-header {{
            background: linear-gradient(90deg, {DARAZ_ORANGE} 0%, #FF8A3D 100%);
            padding: 18px 24px;
            border-radius: 12px;
            margin-bottom: 18px;
        }}
        .daraz-header h1 {{
            color: white;
            font-size: 26px;
            margin: 0;
            font-weight: 700;
        }}
        .daraz-header p {{
            color: #FFF3EB;
            margin: 4px 0 0 0;
            font-size: 14px;
        }}
        .source-chip {{
            display: inline-block;
            background-color: #FFF0E8;
            color: {DARAZ_ORANGE};
            border: 1px solid #FFD4B8;
            border-radius: 999px;
            padding: 2px 10px;
            font-size: 12px;
            margin: 2px 4px 2px 0;
        }}
        div[data-testid="stChatMessage"] {{
            border-radius: 12px;
        }}
        .stButton>button {{
            background-color: {DARAZ_ORANGE};
            color: white;
            border: none;
            border-radius: 8px;
        }}
        .stButton>button:hover {{
            background-color: #E04E04;
            color: white;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="daraz-header">
        <h1>🛍️ Daraz Support Assistant</h1>
        <p>Operations assistant for returns, delivery, refunds, sellers, payments &amp; customer support</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Cached resources: embedding model, FAISS index, metadata
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner="Loading knowledge base index...")
def load_index_and_metadata():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        return None, None
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


@st.cache_resource(show_spinner=False)
def load_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


embedder = load_embedder()
index, metadata = load_index_and_metadata()
groq_client = load_groq_client()

if index is None:
    st.error(
        f"Couldn't find a prebuilt index at `{FAISS_DIR}/`. "
        "Run ingest.py first to generate `index.faiss` and `metadata.json`."
    )
    st.stop()

if groq_client is None:
    st.error(
        "No Groq API key found. Add `GROQ_API_KEY` to your Streamlit secrets "
        "(Settings -> Secrets on Streamlit Cloud, or .streamlit/secrets.toml locally)."
    )
    st.stop()

# --------------------------------------------------------------------------
# Sidebar: knowledge base sections + chat controls
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Knowledge Base")
    st.caption(f"{len(metadata)} indexed chunks across {len(SECTIONS) - 1} sections")

    section_key = st.radio(
        "Restrict search to a section",
        options=[key for key, _ in SECTIONS],
        format_func=lambda key: dict(SECTIONS)[key],
        index=0,
    )

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Model: Groq · openai/gpt-oss-120b")
    st.caption("Retrieval: FAISS (prebuilt index, no re-embedding at runtime)")

# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
def retrieve_chunks(query: str, section_key: str, top_k: int = TOP_K):
    query_vec = embedder.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")

    k = CANDIDATE_K if section_key != "all" else top_k
    k = min(k, index.ntotal)
    distances, indices = index.search(query_vec, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        entry = metadata[idx]
        if section_key != "all" and entry.get("department") != section_key:
            continue
        results.append({**entry, "score": float(dist)})
        if len(results) >= top_k:
            break

    return results


def build_context(chunks):
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(
            f"[Source {i} | section: {c['department']} | file: {c['source_file']}]\n{c['text']}"
        )
    return "\n\n".join(lines)


SYSTEM_PROMPT = """You are the Daraz Customer Support Operations Assistant.
You answer internal support-team and customer questions about Daraz's returns,
delivery, refunds, seller, payments, and customer support policies.

Rules:
- Answer ONLY using the information in the provided context chunks.
- If the context does not contain enough information to answer confidently,
  say so plainly and suggest which knowledge-base section the user might want
  to check instead, or recommend escalating to a human agent.
- Keep answers clear, concise, and operational (numbered steps or short
  bullet points where helpful).
- Do not invent policy details, numbers, or timelines that are not present
  in the context.
- Mention the relevant section(s) (e.g. Returns, Refunds) when it helps the
  reader know where the information came from.
"""


def generate_answer(question: str, chunks):
    context = build_context(chunks)
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the context above."
    )

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return response.choices[0].message.content


# --------------------------------------------------------------------------
# Chat UI
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            chips = "".join(
                f'<span class="source-chip">{s["department"]} · {s["source_file"]}</span>'
                for s in msg["sources"]
            )
            st.markdown(chips, unsafe_allow_html=True)

placeholder = "Ask about returns, delivery, refunds, sellers, payments, or customer support..."
if question := st.chat_input(placeholder):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base..."):
            chunks = retrieve_chunks(question, section_key)

        if not chunks:
            answer = (
                "I couldn't find anything relevant in "
                f"{'the ' + dict(SECTIONS)[section_key] + ' section' if section_key != 'all' else 'the knowledge base'}. "
                "Try rephrasing, switch to 'All Sections', or escalate to a human agent."
            )
            st.markdown(answer)
        else:
            with st.spinner("Generating answer..."):
                answer = generate_answer(question, chunks)
            st.markdown(answer)

            with st.expander(f"Sources ({len(chunks)})"):
                for c in chunks:
                    st.markdown(
                        f"**{dict(SECTIONS).get(c['department'], c['department'])}** "
                        f"— `{c['source_file']}` (chunk #{c['chunk_index']})"
                    )
                    st.caption(c["text"][:400] + ("..." if len(c["text"]) > 400 else ""))

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": [
                    {"department": c["department"], "source_file": c["source_file"]}
                    for c in chunks
                ] if chunks else [],
            }
        )
