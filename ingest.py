import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
 
file=open("data.json")
pages=json.load(file)
docs=[]
for page in pages:
    content = page["content"].strip()
    if content != "":
        docs.append(Document(page_content=content,metadata={"url": page["url"],"title": page["title"]}))
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " "]
)
 
chunks = splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("cogniwide_vectorstore")
 
 
chunk_data = []
 
for i, chunk in enumerate(chunks):
    chunk_data.append({
        "id": i,
        "content": chunk.page_content,
        "metadata": chunk.metadata
    })
 
with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunk_data, f, indent=2, ensure_ascii=False)
 
print("Chunks saved")