import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI
import time

def response_text(openai_resp):
    return openai_resp.choices[0].message.content


os.environ["OPENAI_API_KEY"] = "api key"

openai_client = OpenAI()
chroma_client = chromadb.Client()

class OpenAICache:
    def __init__(self, openai_client, semantic_cache):
        self.openai_client = openai_client
        self.cache = {}
        self.semantic_cache = semantic_cache

    def generate(self, prompt):
        if prompt not in self.cache: ## (1) 일치캐시에 프롬프트가 없으면
            similar_doc = self.semantic_cache.query(query_texts=[prompt], n_results=1) ## (2) 데이터베이스에 등록된 임베딩 모델을 통해 임베딩 벡터로 변환하고 검색
            if len(similar_doc['distances'][0]) > 0 and similar_doc['distances'][0][0] < 0.2: ##(3) 검색결과가 존재하고 거리가 0.2보다 작으면 검색된 문서 반환
                return similar_doc['metadatas'][0][0]['response']
            else:
                response = self.openai_client.chat.completions.create( ## (4) 조건을 만족시키지 않으면 새로운 결과를 생성
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                         'role': 'user',
                         'content': prompt
                        }
                    ],
                )
                self.cache[prompt] = response_text(response) ## (5) 응답을 일치 캐시와 유사검색 캐시에 저장
                self.semantic_cache.add(documents=[prompt], metadatas=[{'response': response_text(response)}], ids=[prompt])
        return self.cache[prompt] ## (4) 캐시에서 응답 반환


openai_ef = OpenAIEmbeddingFunction(
    model_name="text-embedding-ada-002",
    api_key=os.environ["OPENAI_API_KEY"]
)

semantic_cache = chroma_client.create_collection(name="semantic_cache",
                                                 embedding_function=openai_ef, metadata={"hnsw:space": "cosine"})

openai_cache = OpenAICache(openai_client, semantic_cache)

questions = ["북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?",
             "북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?",
             "북태평양 기단과 오호츠크해 기단이 만나 한반도에 머무르는 기간은?",
             "국내에 북태평양 기단과 오호츠크해 기단이 함께 머무르는 기간은?"]

for question in questions:
    start_time = time.time()
    response = openai_cache.generate(question)
    print(f'질문: {question}')
    print("소요시간: {:2f}s".format(time.time() - start_time))
    print(f'답변: {response}\n')


