import os
from datasets import load_dataset
from llama_index.core import Document, VectorStoreIndex
from dotenv import load_dotenv

load_dotenv()
print(os.environ["OPENAI_API_KEY"])

dataset = load_dataset('klue', 'mrc', split='train') # 여기!!
print(dataset[0])

text_list = dataset[:100]['context']
documents = [Document(text=t) for t in text_list]

index = VectorStoreIndex.from_documents(documents)

print(dataset[0]['question'])

retrieval_engine = index.as_retriever(similarity_top_k=5, verbose=True)
response = retrieval_engine.retrieve(
    dataset[0]['question']
)
print(len(response))
print(response[0].node.text)


query_engine = index.as_query_engine(similarity_top_k=1)
response = query_engine.query(
    dataset[0]['question']
)
print(response)