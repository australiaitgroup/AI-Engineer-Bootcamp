import click
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents.base import Document
import ollama
import os

# 配置常量
VECTOR_STORE_DIR = "./chroma_db"
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, model_kwargs={"trust_remote_code": True}
    )


@click.group()
def cli():
    """PDF问答命令行工具"""
    pass


@cli.command()
@click.argument("pdf_file", type=click.Path(exists=True))
def learn(pdf_file):
    """学习新的PDF文档"""
    # 加载并分块文档
    loader = PyPDFLoader(pdf_file)
    pages: list[Document] = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(pages)

    # 创建向量存储
    vector_store: Chroma = Chroma.from_documents(
        documents=docs, embedding=get_embeddings(), persist_directory=VECTOR_STORE_DIR
    )
    vector_store.persist()
    click.echo(f"成功学习文档：{pdf_file}，块数量：{len(docs)}")


@cli.command()
def chat():
    """启动问答会话"""
    # 检查知识库是否存在
    if not os.path.exists(VECTOR_STORE_DIR):
        raise click.ClickException("未找到知识库，请先执行 learn 命令")

    # 加载已有向量库
    vector_store = Chroma(
        persist_directory=VECTOR_STORE_DIR, embedding_function=get_embeddings()
    )

    # 对话循环
    click.echo("进入问答模式（输入 q 退出）")
    while True:
        question = click.prompt("问题")
        if question.lower() == "q":
            break

        # 检索与生成
        results = vector_store.similarity_search(question, k=10)
        context = "\n------- reference -------\n".join(
            [doc.page_content for doc in results]
        )
        prompt: str = (
            f"{question}\n以下reference仅供参考：\n------- reference -------\n{context}\n\n"
        )
        print("prompt:", prompt)
        response = ollama.generate(
            model="deepseek-r1:8b",
            prompt=prompt,
        )
        click.echo(f"\n回答：{response['response']}\n")


if __name__ == "__main__":
    cli()
