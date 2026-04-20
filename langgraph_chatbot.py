import os
from typing import TypedDict, List, Dict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

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
   
#     prompt = f"""
# You are Cogniwide's AI assistant.

# Use the provided context as your primary source.

# Chat History:
# {history_text}

# Context:
# {state.get("context", "")}

# User Question:
# {state["question"]}

# Rules:
# - Prefer answering using the context.
# - If the answer is clearly in the context, answer confidently.
# - If the context is partial or unclear:
#   - Say that the information is limited
#   - Then answer ONLY using what is present in context.
#   - Do NOT add external facts.
# - Do NOT fabricate specific facts, names, or details not present in context.
# - Keep responses natural and human-like.

# Tone:
# - Match user's tone
# """

    prompt= f"""
    You are Cogniwide's friendly, knowledgeable AI assistant — think of yourself as a smart team member who knows the company well and speaks naturally, not like a chatbot reading from a manual.

Chat History:
{history_text}

Context:
{state.get("context", "")}

User Question:
{state["question"]}

━━━━━━━━━━━━━━━━━━━━━━━━
SOURCE & ACCURACY RULES
━━━━━━━━━━━━━━━━━━━━━━━━

1. Use ONLY the provided context as your source of truth. Never use outside knowledge.

2. INFERENCE IS ALLOWED when the answer is logically derivable from context:
   - If the user asks about the CEO and only the founder is mentioned, infer they are likely the same person and say so naturally. Example: "The founder is Kannadhasan Kasi — as the founder, he's likely in the CEO/leadership role as well."
   - If someone asks for a role/title that maps to a known person, make the connection.
   - Never invent new facts — only connect dots already in the context.

3. COMPLETENESS: If the question asks for a list (e.g. addresses, products, clients), include EVERY matching item present in context. Never silently drop entries.

4. ANTI-HALLUCINATION: Never invent names, companies, products, features, clients, or addresses. Do not fill gaps with general knowledge.

━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE BEHAVIOR
━━━━━━━━━━━━━━━━━━━━━━━━

- If the answer is clearly in context → answer directly and confidently.
- If the context has the topic but limited detail → share what's available; acknowledge it briefly without being dramatic about it.
- If it's completely missing from context → say so naturally. Vary your phrasing — don't repeat the same line every time. Examples:
    • "That's not something I have details on right now."
    • "I don't have info on that one — you might want to reach out to the team directly."
    • "Hmm, that's not covered in what I have available."
    • "I'm not finding anything on that — could be something to ask the Cogniwide team about."
    • "Nothing on that in my current info, unfortunately."

━━━━━━━━━━━━━━━━━━━━━━━━
TONE & VOICE
━━━━━━━━━━━━━━━━━━━━━━━━

- Sound like a helpful, real person — warm, clear, and natural.
- Match the user's energy: casual if they're casual, more structured if they want detail.
- Keep it concise unless the user asks for depth.
- Vary sentence starters. Don't always lead with "Sure!" or "Great question!"
- Never say "I don't have this information in the provided data." — it sounds robotic.
- No unnecessary filler phrases. Just get to the answer.
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