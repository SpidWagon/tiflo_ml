from huggingface_hub import snapshot_download

model_id = "Helsinki-NLP/opus-mt-en-ru"

local_dir = "./model"

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,   # чтобы при повторном запуске не качало заново
    force_download=False
)

print(f"Модель '{model_id}' скачана в папку {local_dir}")
