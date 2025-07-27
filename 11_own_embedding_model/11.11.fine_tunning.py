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

# 0.8202286335316458


print("## 11.17 긍정 데이터만으로 학습데이터 구성")
train_samples = []
for idx, row in df_train_ir.iterrows():
    train_samples.append(
        InputExample(texts=[row['question'], row['context']])
    )

print("## 11.18 중복 학습 데이터 제거")
from sentence_transformers import datasets

## M1 16G 노트북에서 안돌아서 눈물을 머금고 8로 조정...
batch_size = 8
##batch_size = 16
loader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=batch_size)

print("## 11.19 NMR 손실함수 불러오기")
from sentence_transformers import losses

loss = losses.MultipleNegativesRankingLoss(sentence_model)

print("## 11.20 MRC 데이터셋으로 미세 조정")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # MPS 메모리 할당 상한선 비활성화


epochs = 1
save_path = "./klue_mrc_mnr"

sentence_model.fit(
    train_objectives=[(loader, loss)],
    epochs=epochs,
    warmup_steps=100,
    output_path=save_path,
    show_progress_bar=True,
)

# Epoch:   0%|          | 0/1 [00:00<?, ?it/s]
# Iteration:   0%|          | 0/1097 [00:00<?, ?it/s]
# Iteration:   0%|          | 1/1097 [00:58<17:55:10, 58.86s/it]

## RuntimeError: MPS backend out of memory (MPS allocated: 17.36 GB, other allocations: 613.56 MB, max allowed: 18.13 GB). Tried to allocate 192.00 MB on private pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).



print("## 11.21 미세 조정한 모델의 성능평가")
print(evaluator(sentence_model))
