import asyncio
import faiss
import pickle
import streamlit as st
from fastapi import FastAPI
from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('clip-ViT-B-32')

imgindex = faiss.read_index("index.faiss")
with open("index.pkl", "rb") as f:
    imgpaths = pickle.load(f)

TIMEOUT_KEEP_ALIVE=30
app = FastAPI()

async def process(query):
    results = []
    vector = np.array([model.encode(query)], dtype=np.float32)
    vector = vector / np.linalg.norm(vector)
    scores, indices = imgindex.search(vector, 5)
    for i in indices[0]:
        img = imgpaths[i]
        results.append(img)
    return results

def main():

    st.set_page_config(
        page_title="Images search",
        page_icon=":books:",
    )

    st.title("Images search")
    st.markdown(f"<p style='text-align: right;'><b>Answer</b></p>", unsafe_allow_html=True)
    st_answer = st.empty()

    user_question = st.text_input("Search for")
    with st.spinner("Processing..."):
        if user_question:
            result = asyncio.run(process(user_question))
            with st_answer.container():
                for r in result:
                    st.image(Image.open(r))
                    st.text(r)

if __name__ == "__main__":
    main()
