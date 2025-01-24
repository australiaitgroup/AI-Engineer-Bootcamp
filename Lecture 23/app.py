from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import chainlit as cl

welcome_message = "Hello! I am your research assistant. Feel free to ask me any questions on Agent AI: Surveying the Horizons of Multimodal Interaction."
namespace = "ai-agent-survey"
embeddings = OpenAIEmbeddings()
index_name = "jr-lab"
# template = """You are a research assistant. Your task is to read the relevant content from academic paper or journal articles, then convert it to plain languages
#             such that the general public can understand.\n\n"""


@cl.on_chat_start
async def start():
    await cl.Message(content=welcome_message).send()
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        text_key="source",
        namespace=namespace,
    )

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1, streaming=True),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True,
        # prompt=PromptTemplate.from_template(template),
    )
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain

    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(
                    content=source_doc.page_content, name=source_name, display="side"
                )
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()
