import gradio as gr
from langgraph_chatbot import app
chat_history = []
def chat(message, history):
    global chat_history
 
    result = app.invoke({
        "question": message,
        "context": "",
        "answer": "",
        "chat_history": chat_history
    })
    answer = result["answer"]
    chat_history.append({"role": "user","content": message})
    chat_history.append({"role": "assistant","content": answer})
    max_messages = 10
    if len(chat_history) > max_messages:
        chat_history = chat_history[-max_messages:]
 
    return answer
demo = gr.ChatInterface(fn=chat,title="Cogniwide AI Chatbot",description="RAG Bot")
demo.launch(share=True)