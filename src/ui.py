import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
import ollama
import subprocess
import sys

st.title("The BookRDR")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path='vector_db/')
    return client.get_or_create_collection(name='book_chunks')

model = load_model()
collection = load_collection()

with st.sidebar:
    st.header('Add New Books')
    uploaded_files = st.file_uploader("Upload PDF or EPUB", type=['pdf','epub'], 
                                   accept_multiple_files=True)
    if st.button("Add to Library") and uploaded_files:
        for uploaded_file in uploaded_files:
            with open(f"books/{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.write(f"✅ Saved: {uploaded_file.name}")
        all_success = True    
        with st.spinner("Processing new books..."):
            scripts=[
            'src/phase2_extract.py',
            'src/convert_to_chunks.py',
            'src/embeding.py',
            'src/store_to_vector.py',
            ]
            for script in scripts:
                result = subprocess.run([sys.executable, script])
                if result.returncode !=0:
                    st.error(f"Failed at {script}")
                    break
        if all_success:
            st.success("✅ Books added! Reloading...")
            st.cache_resource.clear()
            st.rerun()
    st.divider()
    
    # manual reload button
    if st.button("🔄 Reload Library"):
        st.cache_resource.clear()
        st.rerun()
        
    st.divider()
    st.info(f"Library: {collection.count()} chunks indexed")

if 'history' not in st.session_state:
    st.session_state.history = []
for item in st.session_state.history:
    st.write(f"You: {item['question']}")
    st.success(item['answer'])
    for source in item['sources']:
        st.caption(f"Source: {source['book']} — chunk {source['chunk_id']}")
    st.divider()


question = st.text_input("Ask a question:")
if st.button('Ask') and question:
    embed_q = model.encode([question])
    
    results = collection.query(
        query_embeddings=[embed_q[0].tolist()],
        n_results=5,
        include=['documents', 'metadatas', 'distances']
    )
    
    chunks    = results['documents'][0]
    metas     = results['metadatas'][0]
    distances = results['distances'][0]
    
    best_distance = min(distances)
    
    # block irrelevant questions before reaching Mistral
    if best_distance > 1:
        st.warning("I don't have this topic in my library.")
        st.caption(f"Best match score: {best_distance:.2f} — too low")
    else:
        # filter only good chunks
        filtered_chunks = []
        filtered_metas  = []
        for chunk, meta, dist in zip(chunks, metas, distances):
            if dist < 1:
                filtered_chunks.append(chunk)
                filtered_metas.append(meta)
        
        context = "\n\n".join(filtered_chunks)
        
        prompt = f"""You are a strict book assistant.
RULES:
- Answer ONLY from the context below
- If context does not contain the answer say: "This topic is not in my library"
- Never use outside knowledge
- Always mention the book name in your answer

Context:
{context}

Question: {question}

Answer:"""
        
        response = ollama.chat(
            model='mistral',
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        st.write("Answer:")
        st.success(response['message']['content'])
        st.caption(f"Match confidence: {1 - best_distance:.0%}")
        
        st.write("Sources:")
        for meta, dist in zip(filtered_metas, distances):
            st.caption(f"📖 {meta['book']} — chunk {meta['chunk_id']} — score {1-dist:.0%}")
        
        st.session_state.history.append({
            'question': question,
            'answer':   response['message']['content'],
            'sources':  filtered_metas
        })