from huggingface_hub import snapshot_download

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
