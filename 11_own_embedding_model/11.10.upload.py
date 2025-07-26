print("## 11.10 학습한 임베딩 모델 저장")
from huggingface_hub import login
from huggingface_hub import HfApi

model_name = 'klue/roberta-base' # 모델 이름
model_save_path = 'output/training_sts_' + model_name.replace("/", "-")


login(token="huggingface token")  # Hugging Face Hub에 로그인합니다.
api = HfApi()
repo_id="klue-roberta-base-klue-sts"
api.create_repo(repo_id=repo_id)

api.upload_folder(
    folder_path=model_save_path,
    repo_id=f"zbum/{repo_id}",
    repo_type="model",
)