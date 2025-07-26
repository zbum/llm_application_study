import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

word_dict = {
    "school": np.array([[1, 0, 0]]),
    "study": np.array([[0, 1, 0]]),
    "workout": np.array([[0, 1, 0]]),
}
print(cosine_similarity(word_dict['school'], word_dict['study']))
print(cosine_similarity(word_dict['school'], word_dict['workout']))