from huggingface_hub import snapshot_download

MODEL_ID = "Helsinki-NLP/opus-mt-en-ru"
LOCAL_DIR = "./model_artifacts"

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=LOCAL_DIR,
    local_dir_use_symlinks=False,
    resume_download=True,   # чтобы при повторном запуске не качало заново
    force_download=False
)

print(f"Модель '{MODEL_ID}' скачана в папку {LOCAL_DIR}")
