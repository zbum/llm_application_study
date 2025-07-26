print("## 11.1 임베딩 모델 만들기")
from sentence_transformers import SentenceTransformer, models
transformer_model = models.Transformer('klue/roberta-base') # 'klue/roberta-base'는 한국어에 최적화된 RoBERTa 모델입니다. (로버타)

pooling_layer = models.Pooling( # 평균 풀링층을 만듦
    transformer_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)

embedding_model = SentenceTransformer(modules=[transformer_model, pooling_layer]) # tranasformer 모델과 풀링 레이어를 결합하여 임베딩 모델을 만듦
print(embedding_model) # SentenceTransformer 모델을 출력



print("## 11.2 실습 데이터셋 다운로드 및 확인")
from datasets import load_dataset
klue_sts_train = load_dataset('klue', 'sts', split='train') # KLUE STS 데이터셋을 로드합니다.
klue_sts_test = load_dataset('klue', 'sts', split='validation') # KLUE STS 데이터셋의 검증용 데이터를 로드합니다.
print(klue_sts_train[0]) # 첫 번째 데이터 샘플을 출력합니다.

##
## {'guid': 'klue-sts-v1_train_00000', 'source': 'airbnb-rtt',
#   'sentence1': '숙소 위치는 찾기 쉽고 일반적인 한국의 반지하 숙소입니다.',
#   'sentence2': '숙박시설의 위치는 쉽게 찾을 수 있고 한국의 대표적인 반지하 숙박시설입니다.',
#   'labels': {'label': 3.7, 'real-label': 3.714285714285714, 'binary-label': 1}}
##

print( "## 11.3 학습 데이터에서 검증 데이터셋 분리하기")
klue_sts_train = klue_sts_train.train_test_split(test_size=0.1, seed=42) # 학습 데이터셋을 90%와 10%로 분할합니다.학습데이터와 검증 데이터를 동일하게 분리할 수 있도록 seed를 설정합니다.
klue_sts_train, klue_sts_eval = klue_sts_train['train'], klue_sts_train['test'] # 분할된 학습 데이터와 검증 데이터를 변수에 저장합니다.

print( "## 11.4 label 정규화하기")
from sentence_transformers import InputExample
## 유사도 점수를 0~1 사이로 정규화하고 InputExample 객체로 변환합니다. (한마디로 0 ~ 5 사이의 점수를 0 ~ 1 사이로 변환합니다.)
def prepare_sts_examples(dataset):
    examples = []
    for data in dataset:
        examples.append(InputExample(
            texts=[data['sentence1'], data['sentence2']],
            label=data['labels']['label']/5.0)
        )
    return examples

train_examples = prepare_sts_examples(klue_sts_train)
eval_examples = prepare_sts_examples(klue_sts_eval)
test_examples = prepare_sts_examples(klue_sts_test)

print(train_examples[0])


print( "## 11.5 학습에 사용할 배치 데이터셋 만들기")
from torch.utils.data import DataLoader
train_dataloader = DataLoader(train_examples, batch_size=16, shuffle=True)


print("## 11.6 검증을 위한 평가 객체 준비")
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

eval_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(eval_examples)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_examples)

print("## 11.7 언어 모델을 그대로 활용할 경우 문장 임베딩 모델 성능")
print(test_evaluator(embedding_model))

print("## 11.8 임베딩 모델 학습")
from sentence_transformers import losses

num_epochs = 4 # 학습 에폭 수
model_name = 'klue/roberta-base' # 모델 이름
model_save_path = 'output/training_sts_' + model_name.replace("/", "-")
train_loss = losses.CosineSimilarityLoss(model=embedding_model)

### 임베딩 모델 학습
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

## wandb login 필요합니다.
embedding_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=eval_evaluator,
    epochs=num_epochs,
    evaluation_steps=1000,  # 평가 주기 설정
    warmup_steps=100,
    output_path=model_save_path,
)

# Iteration:  95%|█████████▌| 626/657 [04:26<00:13,  2.31it/s]
# Iteration:  95%|█████████▌| 627/657 [04:26<00:12,  2.32it/s]
# Iteration:  96%|█████████▌| 628/657 [04:27<00:12,  2.32it/s]
# Iteration:  96%|█████████▌| 629/657 [04:27<00:12,  2.32it/s]
# Iteration:  96%|█████████▌| 630/657 [04:28<00:11,  2.32it/s]
# Iteration:  96%|█████████▌| 631/657 [04:28<00:11,  2.32it/s]
# Iteration:  96%|█████████▌| 632/657 [04:29<00:10,  2.31it/s]
# Iteration:  96%|█████████▋| 633/657 [04:29<00:10,  2.30it/s]
# Iteration:  96%|█████████▋| 634/657 [04:30<00:09,  2.31it/s]
# Iteration:  97%|█████████▋| 635/657 [04:30<00:09,  2.31it/s]
# Iteration:  97%|█████████▋| 636/657 [04:30<00:09,  2.32it/s]
# Iteration:  97%|█████████▋| 637/657 [04:31<00:08,  2.32it/s]
# Iteration:  97%|█████████▋| 638/657 [04:31<00:08,  2.32it/s]
# Iteration:  97%|█████████▋| 639/657 [04:32<00:07,  2.35it/s]
# Iteration:  97%|█████████▋| 640/657 [04:32<00:07,  2.37it/s]
# Iteration:  98%|█████████▊| 641/657 [04:32<00:06,  2.34it/s]
# Iteration:  98%|█████████▊| 642/657 [04:33<00:06,  2.36it/s]
# Iteration:  98%|█████████▊| 643/657 [04:33<00:05,  2.37it/s]
# Iteration:  98%|█████████▊| 644/657 [04:34<00:05,  2.40it/s]
# Iteration:  98%|█████████▊| 645/657 [04:34<00:05,  2.39it/s]
# Iteration:  98%|█████████▊| 646/657 [04:34<00:04,  2.40it/s]
# Iteration:  98%|█████████▊| 647/657 [04:35<00:04,  2.42it/s]
# Iteration:  99%|█████████▊| 648/657 [04:35<00:03,  2.40it/s]
# Iteration:  99%|█████████▉| 649/657 [04:36<00:03,  2.41it/s]
# Iteration:  99%|█████████▉| 650/657 [04:36<00:02,  2.42it/s]
# Iteration:  99%|█████████▉| 651/657 [04:36<00:02,  2.39it/s]
# Iteration:  99%|█████████▉| 652/657 [04:37<00:02,  2.37it/s]
# Iteration:  99%|█████████▉| 653/657 [04:37<00:01,  2.40it/s]
# Iteration: 100%|█████████▉| 654/657 [04:38<00:01,  2.41it/s]
# Iteration: 100%|█████████▉| 655/657 [04:38<00:00,  2.37it/s]
# Iteration: 100%|█████████▉| 656/657 [04:39<00:00,  2.37it/s]
# Iteration: 100%|██████████| 657/657 [04:39<00:00,  2.35it/s]
# Epoch: 100%|██████████| 4/4 [19:07<00:00, 286.75s/it]


print("## 11.9 학습한 임베딩모델의 성능 평가")
trained_embedding_model = SentenceTransformer(model_save_path)
print(test_evaluator(trained_embedding_model))

# ## 11.9 학습한 임베딩모델의 성능 평가
# 0.8900606603141447
