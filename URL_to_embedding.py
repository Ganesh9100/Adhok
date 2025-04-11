from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse

from agno.agent import Agent
from agno.knowledge.langchain import LangChainKnowledgeBase
from agno.models.ollama import Ollama
urls = ["https://www.straighttalk.com/privacy-policy",
        "https://www.straighttalk.com/support/terms-conditions"
#     "https://www.straighttalk.com/all-plans"
#     "https://blog.google/technology/developers/gemma-3/",
]

loader = WebBaseLoader(urls)
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=50
)
chunks = text_splitter.split_documents(data)
embeddings = FastEmbedEmbeddings(model_name="thenlper/gte-large")



# Installation
# pip install fastembed
# pip install agno==1.0.0
# pip install langchain-qdrant
# pip install typing_extensions --upgrade
# !pip install --upgrade pip
# !pip install ollama
# !pip install langchain
# !pip install langchain_community
# !pip install -U langchain-ollama
# !pip install typing_extensions==4.7.1
