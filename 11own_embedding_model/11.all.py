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
# embedding_model.fit(
#     train_objectives=[(train_dataloader, train_loss)],
#     evaluator=eval_evaluator,
#     epochs=num_epochs,
#     evaluation_steps=1000,  # 평가 주기 설정
#     warmup_steps=100,
#     output_path=model_save_path,
# )

### 실행결과
# wandb: 🚀 View run at https://wandb.ai/zbum-nhn-dooray/sentence-transformers/runs/omdhev95
#   0%|          | 0/2628 [00:00<?, ?it/s]/Users/nhn/IdeaProjects/quarkus/rapid_ocr_ex1/.venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py:683: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, then device pinned memory won't be used.
#   warnings.warn(warn_msg)
#  19%|█▉        | 500/2628 [05:08<47:12,  1.33s/it]{'loss': 0.0281, 'grad_norm': 0.8412415981292725, 'learning_rate': 1.6838351822503965e-05, 'epoch': 0.76}
#  38%|███▊      | 1000/2628 [10:25<11:29,  2.36it/s]{'loss': 0.0081, 'grad_norm': 0.3559873104095459, 'learning_rate': 1.287638668779715e-05, 'epoch': 1.52}
# {'eval_pearson_cosine': 0.9597710920882392, 'eval_spearman_cosine': 0.9174741951802741, 'eval_runtime': 5.7564, 'eval_samples_per_second': 0.0, 'eval_steps_per_second': 0.0, 'epoch': 1.52}
#  57%|█████▋    | 1500/2628 [14:39<08:21,  2.25it/s]{'loss': 0.0052, 'grad_norm': 0.2036622315645218, 'learning_rate': 8.914421553090334e-06, 'epoch': 2.28}
#  76%|███████▌  | 2000/2628 [18:56<04:03,  2.58it/s]{'loss': 0.0034, 'grad_norm': 0.27582746744155884, 'learning_rate': 4.952456418383519e-06, 'epoch': 3.04}
#  76%|███████▌  | 2000/2628 [19:01<04:03,  2.58it/s]{'eval_pearson_cosine': 0.962240285336236, 'eval_spearman_cosine': 0.9212081579770475, 'eval_runtime': 5.8192, 'eval_samples_per_second': 0.0, 'eval_steps_per_second': 0.0, 'epoch': 3.04}
#  95%|█████████▌| 2500/2628 [23:12<00:59,  2.15it/s]{'loss': 0.0026, 'grad_norm': 0.22588618099689484, 'learning_rate': 9.904912836767039e-07, 'epoch': 3.81}
# 100%|██████████| 2628/2628 [24:42<00:00,  1.77it/s]
# {'train_runtime': 1485.7208, 'train_samples_per_second': 28.272, 'train_steps_per_second': 1.769, 'train_loss': 0.009126491572486755, 'epoch': 4.0}
# ## 11.9 학습한 임베딩모델의 성능 평가
# wandb:
# wandb: Run history:
# wandb:     eval/pearson_cosine ▁█
# wandb:            eval/runtime ▁█
# wandb: eval/samples_per_second ▁▁
# wandb:    eval/spearman_cosine ▁█
# wandb:   eval/steps_per_second ▁▁
# wandb:             train/epoch ▁▃▃▄▆▆██
# wandb:       train/global_step ▁▃▃▄▆▆██
# wandb:         train/grad_norm █▃▁▂▁
# wandb:     train/learning_rate █▆▄▃▁
# wandb:              train/loss █▃▂▁▁
# wandb:
# wandb: Run summary:
# wandb:      eval/pearson_cosine 0.96224
# wandb:             eval/runtime 5.8192
# wandb:  eval/samples_per_second 0.0
# wandb:     eval/spearman_cosine 0.92121
# wandb:    eval/steps_per_second 0.0
# wandb:               total_flos 0.0
# wandb:              train/epoch 4.0
# wandb:        train/global_step 2628
# wandb:          train/grad_norm 0.22589
# wandb:      train/learning_rate 0.0
# wandb:               train/loss 0.0026
# wandb:               train_loss 0.00913
# wandb:            train_runtime 1485.7208
# wandb: train_samples_per_second 28.272
# wandb:   train_steps_per_second 1.769
# wandb:
# wandb: 🚀 View run checkpoints/model at: https://wandb.ai/zbum-nhn-dooray/sentence-transformers/runs/omdhev95
# wandb: ⭐️ View project at: https://wandb.ai/zbum-nhn-dooray/sentence-transformers
# wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
# wandb: Find logs at: ./wandb/run-20250720_145949-omdhev95/logs

print("## 11.9 학습한 임베딩모델의 성능 평가")
trained_embedding_model = SentenceTransformer(model_save_path)
print(test_evaluator(trained_embedding_model))

