from huggingface_hub import upload_folder, login, logout, HfApi
import argparse, pathlib, sys, os

p = argparse.ArgumentParser()
p.add_argument("--token", required=True)
p.add_argument("--repo",  required=True)      
p.add_argument("--local", required=True)          
args = p.parse_args()

local_dir = pathlib.Path(args.local).expanduser()
if not local_dir.is_dir():
    sys.exit(f"{local_dir} не найдена или не папка")

try: logout()
except Exception: pass
login(args.token, add_to_git_credential=True)

HfApi().create_repo(args.repo, private=True, exist_ok=True, repo_type="model")

print("Загружаю содержимое папки …")
upload_folder(folder_path=str(local_dir), repo_id=args.repo, repo_type="model")

print(f"Готово! https://huggingface.co/{args.repo}")

