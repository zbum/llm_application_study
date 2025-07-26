from sklearn.feature_extraction.text import CountVectorizer

corpus = ['i want to go home', 'i want to go work']
vector = CountVectorizer()

print('BOW Vector', vector.fit_transform(corpus).toarray())
print('BOW Vocabulary', vector.vocabulary_)