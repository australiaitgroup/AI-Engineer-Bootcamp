from operator import itemgetter
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
load_dotenv()
# Initialize chat model
llm = ChatOpenAI()

# Define a prompt template
template = """You are a helpful AI assistant.
"""

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# Create conversation history store
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

chain = (
    RunnablePassthrough.assign(
        history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
    )
    | chat_prompt
    | llm
)


def stream_response(input, history):
    if input is not None:
        partial_message = ""
        for response in chain.stream({"input": input}):
            partial_message += response.content
            print(partial_message)
            yield partial_message 


# UI
import gradio as gr

gr.ChatInterface(stream_response).queue().launch(debug=True)