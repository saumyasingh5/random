from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
import faiss
import numpy as np
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

reader = PdfReader("data/medical.pdf")
text = "".join([page.extract_text() for page in reader.pages])

words = text.split()
chunks = [" ".join(words[i:i+500]) for i in range(0, len(words), 500)]

embedder = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer=AutoTokenizer.from_pretrained("google/flan-t5-small")
model=AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
embeddings = embedder.encode(chunks)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype('float32'))

print("Server Ready")

class Query(BaseModel):
    question: str

@app.post("/ask")
def get_answer(query: Query):
    question = query.question # <- Ye line add kar. Ye missing thi
    
    question_embedding = embedder.encode([question])
    D, I = index.search(question_embedding.astype('float32'), k=1)
    retrieved_context = chunks[I[0][0]][:300]
    
    input_text = f"Give a short answer in 2-3 sentences only. question: {question} context: {retrieved_context}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    
    outputs = model.generate(input_ids, max_length=60) 
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {
        "question": question,
        "answer": answer,
        "source": "Retrieved from medical.pdf"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001) 