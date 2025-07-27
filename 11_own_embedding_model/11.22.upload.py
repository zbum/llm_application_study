print("## 11.12 허깅페이스 허브에 미세 조정한 모델 업로드")
import os
from huggingface_hub import login
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HUGGINGFACE_API_KEY")
login(token=token)  # Hugging Face Hub에 로그인합니다.
save_path = "./klue_mrc_mnr"
api = HfApi()
repo_id="klue-roberta-base-klue-sts-mrc"
api.create_repo(repo_id=repo_id)

api.upload_folder(
    folder_path=save_path,
    repo_id=f"zbum/{repo_id}",
    repo_type="model",
)