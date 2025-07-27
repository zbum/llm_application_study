
print("## 10.1 문장 임베딩을 활용한 단어 간 유사도 계산")
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

smodel = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')  # 사전학습된 모델을 불러옵니다.
dens_embeddings = smodel.encode(['학교','공부','운동'])
print(cosine_similarity(dens_embeddings))


print("## 10.2 원핫 인코딩의 한계")
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

word_dict = { "school": np.array([[1,0,0]]),
                "study": np.array([[0,1,0]]),
                "workout": np.array([[0,0,1]])
              }

cosine_school_study = cosine_similarity(word_dict["school"], word_dict["study"])
cosine_school_workout = cosine_similarity(word_dict["school"], word_dict["workout"])
print(f"Cosine similarity between 'school' and 'study': {cosine_school_study[0][0]}")
print(f"Cosine similarity between 'school' and 'workout': {cosine_school_workout[0][0]}")

