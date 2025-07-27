print("## 11.12 허깅페이스 허브에 미세 조정한 모델 업로드")
from huggingface_hub import login
from huggingface_hub import HfApi

login(token="huggingface token")  # Hugging Face Hub에 로그인합니다.
save_path = "./klue_mrc_mnr"
api = HfApi()
repo_id="klue-roberta-base-klue-sts-mrc"
api.create_repo(repo_id=repo_id)

api.upload_folder(
    folder_path=save_path,
    repo_id=f"zbum/{repo_id}",
    repo_type="model",
)