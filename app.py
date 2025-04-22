# Web app structure for EU Law QA system

# 1. Import dependencies
import streamlit as st
import json
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import openai
import os
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# 2. Initialize embedding model using a small, fast model
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")  # ~90MB, small & Streamlit Cloud-friendly

# 3. Load full structured EU law content from parsed JSON
with open("real_eu_laws.json", "r") as f:
    real_laws = json.load(f)

# 4. Flatten for embedding
def flatten_laws(laws):
    passages = []
    meta = []
    for law in laws:
        for art in law["articles"]:
            for para in art["paragraphs"]:
                ref = f"{law['title']}, Article {art['article']}, Paragraph {para['number']}"
                passages.append(para['text'])
                meta.append({"text": para['text'], "ref": ref, "url": law['url']})
    return passages, meta

texts, metadata = flatten_laws(real_laws)
embeddings = model.encode(texts, convert_to_numpy=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# 5. Streamlit UI
st.set_page_config(page_title="EU Law Q&A", layout="centered")
st.title("\U0001F4D6 EU Law Assistant")

query = st.text_input("Ask a question about EU digital laws:", placeholder="e.g., Do municipalities need to follow Article 5 of the Data Act?")

if query:
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k=5)
    st.subheader("Answer")

    # Collect top relevant passages
    relevant_contexts = [metadata[idx]["text"] for idx in I[0]]
    references = [metadata[idx] for idx in I[0]]

    # Compose AI prompt
    system_prompt = "You are a legal assistant answering questions about EU digital laws. Always start with YES or NO, then give a brief legal justification with references."
    user_prompt = f"Question: {query}\n\nRelevant legal texts:\n" + "\n\n".join(relevant_contexts)

    # Call OpenAI (or mock response for demo purposes)
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        answer = response["choices"][0]["message"]["content"]
    except Exception as e:
        answer = "YES or NO placeholder. (OpenAI API not available in this environment.)"

    # Display result
    st.write(answer)
    st.markdown("### References")
    for ref in references:
        st.write(f"**{ref['ref']}**")
        st.markdown(f"[View full law]({ref['url']})")
        st.markdown("---")
