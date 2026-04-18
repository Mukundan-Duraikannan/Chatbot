import os
import json
from typing import TypedDict, List, Dict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
load_dotenv()
 
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "cogniwide_vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)
 
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
 
llm = ChatGroq(
    api_key=os.environ["GROQ_API_KEY"],
    model_name="openai/gpt-oss-20b",
    temperature=0.3,
    max_tokens=500,
)
 
 
class ChatState(TypedDict):
    question: str
    context: str
    answer: str
    chat_history: List[Dict]
 
def retrieve(state: ChatState):
    docs = retriever.invoke(state["question"])
    context = "\n\n".join(doc.page_content for doc in docs)
    return {"context": context}
 
def generate(state: ChatState):
    history_text = "\n".join(f"{msg['role']}: {msg['content']}"for msg in state.get("chat_history", []))
   
    prompt = f"""
You are Cogniwide's personal AI assistant.
 
Your job is to answer ONLY using the provided context.
 
Chat History:
{history_text}
 
Context:
{state.get("context", "")}
 
User Question:
{state["question"]}
 
Rules:
- Use ONLY the given context to answer.
- Do NOT make up, assume, or infer anything outside the context.
- Do NOT reframe or modify the question to force an answer.
 
If the answer is not present in the context:
- You MUST respond in a different way every time.
- Never repeat the same sentence structure twice in a row.
- Avoid starting responses with the same phrase (e.g., “I'm sorry” repeatedly).
 
You may respond in varied styles such as:
- Simple: “I don't see details about that here.”
- Neutral: “That topic isn't covered in the provided context.”
- Friendly: “Hmm, I don't have info on that right now.”
- Direct: “No information available in the current context.”
 
IMPORTANT:
- Do NOT default to one fixed sentence.
- Do NOT always apologize.
- Rotate tone and structure naturally like a human assistant.
 
Tone:
- Match the user's tone (casual or formal).
- Keep responses clear, natural, and human-like.
"""
    response = llm.invoke(prompt)
 
    return {"answer": response.content}
graph = StateGraph(ChatState)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)
app = graph.compile()
if __name__ == "__main__":
    print("\nChatbot ready!\n")
 
    chat_history = []
    while True:
        query = input("You: ").strip()
        result = app.invoke({
            "question": query,
            "chat_history": chat_history
        })
        answer = result["answer"]
        print("Bot:", answer)
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})