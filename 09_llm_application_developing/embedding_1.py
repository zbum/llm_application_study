from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

smodel = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')  # https://huggingface.co/snunlp/KR-SBERT-V40K-klueNLI-augSTS
print(smodel) # 일반적으로 work_embedding_dimension은 주로 768, 1024 차원으로 설정된다. (차원은 의미를 담을 수 있는 공간), 256의 배수로 표현되는 것 같은데 이유는 모르겠음

dense_embeddings = smodel.encode(['학교', '공부', '운동'])
print(cosine_similarity(dense_embeddings))