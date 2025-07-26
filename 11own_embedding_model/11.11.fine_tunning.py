from chromadb.experimental.density_relevance import batch_size

print("## 11.11 실습 데이터를 내려받고 예시 데이터 확인")
from datasets import load_dataset
klue_mrc_train = load_dataset('klue', 'mrc', split='train') # KLUE MRC 데이터셋을 로드합니다.
print(klue_mrc_train[0])


print("## 11.12 기본 임베딩 모델 불러오기")
from sentence_transformers import SentenceTransformer
sentence_model = SentenceTransformer('zbum/klue-roberta-base-klue-sts')

print("## 11.13 데이터 전처리")
from datasets import load_dataset
klue_mrc_train = load_dataset('klue', 'mrc', split='train')
klue_mrc_test = load_dataset('klue', 'mrc', split='validation')

df_train = klue_mrc_train.to_pandas()
df_test = klue_mrc_test.to_pandas()

df_train = df_train[['title', 'question', 'context']]
df_test = df_test[['title', 'question', 'context']]

print("## 11.14 질문과 관련없는 기사를 irrelevant_context 컬럼에 추가")
def add_ir_context(df):
    irrelevant_contexts = []
    for idx, row in df.iterrows():
        title = row['title']
        irrelevant_contexts.append(df.query(f"title != '{title}'").sample(n=1)['context'].values[0])
    df['irrelevant_context'] = irrelevant_contexts
    return df

df_train_ir = add_ir_context(df_train)
df_test_ir = add_ir_context(df_test)

print("## 11.15 성능 평가에 사용할 데이터 생성")
from sentence_transformers import InputExample
examples = []
for idx, row in df_test_ir.iterrows():
    examples.append(
        InputExample(texts=[row['question'], row['context']], label=1)
    )
    examples.append(
        InputExample(texts=[row['question'], row['irrelevant_context']], label=0)
    )

print("## 11.16 기본 임베딩 모델의 성능 평가 결과")
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(examples)
print(evaluator(sentence_model))

print("## 11.17 긍정 데이터만으로 학습데이터 구성")
train_samples = []
for idx, row in df_train_ir.iterrows():
    train_samples.append(
        InputExample(texts=[row['question'], row['context']])
    )

print("## 11.18 중복 학습 데이터 제거")
from sentence_transformers import datasets

batch_size = 16

loader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=batch_size)