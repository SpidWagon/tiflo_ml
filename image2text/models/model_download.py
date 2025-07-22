from huggingface_hub import snapshot_download
import os

MODEL_ID = "Helsinki-NLP/opus-mt-en-ru"
LOCAL_DIR = "./model_artifacts"


def download_model(model_id=MODEL_ID, local_dir=LOCAL_DIR):
    print("model download start")
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,   # чтобы при повторном запуске не качало заново
        force_download=False
    )

    print(f"Model '{model_id}' is loaded to {local_dir}")

def download_blip_lora(
        repo_id: str = "Grgoriy/blip2-finetuned-test-2.7b",
        local_dir: str | None = None,
):
    if local_dir is None:
        return snapshot_download(repo_id, repo_type="model")

    abs_dir = os.path.abspath(local_dir)
    need_dl = (not os.path.isdir(abs_dir)) or (not os.listdir(abs_dir))
    if need_dl:
        snapshot_download(
            repo_id,
            repo_type="model",
            local_dir=abs_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            force_download=False,
        )
    return abs_dir