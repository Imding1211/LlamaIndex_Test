from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from langchain_community.embeddings.ollama import OllamaEmbeddings
from llama_index.llms.ollama import Ollama

import warnings
warnings.filterwarnings('ignore')

documents = SimpleDirectoryReader("data").load_data()

# bge-base embedding model
Settings.embed_model = OllamaEmbeddings(model='all-minilm')

# ollama
Settings.llm = Ollama(model="gemma2:2b", request_timeout=360.0)

index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine()

print()
response = query_engine.query("What hardware setup was used for training this models?")
print(response)