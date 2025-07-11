import os, shutil, subprocess, sys, tempfile
from pathlib import Path
import yaml
from tqdm import tqdm

KAGGLE_SLUG = "adityajn105/flickr30k"
DEST_DIR = Path("dataset")
DONE_FLAG = DEST_DIR / ".done"

def load_creds():
    cfg = Path(os.path.join(os.getcwd(), "creds.yaml"))
    # cfg = Path("./")
    # if not cfg.exists():
    #     return
    data = yaml.safe_load(cfg.read_text()) or {}
    os.environ.setdefault("KAGGLE_USERNAME", str(data.get("kaggle_username", "")))
    os.environ.setdefault("KAGGLE_KEY", str(data.get("kaggle_key", "")))
    print("creds loaded")

def has_creds():
    return (
        Path.home().joinpath(".kaggle", "kaggle.json").exists()
        or (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    )

def run(cmd):
    try:
        subprocess.check_call(cmd)
    except FileNotFoundError:
        sys.exit("kaggle CLI не найден. Установите: pip install kaggle")
    except subprocess.CalledProcessError as err:
        sys.exit(f"Команда {' '.join(cmd)} вернула ошибку: {err}")

def main():
    load_creds()

    if DONE_FLAG.exists():
        print("Датасет уже скачан, пропускаю.")
        return

    if not has_creds():
        sys.exit(
            "Нет API-ключа Kaggle"
        )

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        print("Скачиваю архив")
        run(["kaggle", "datasets", "download", "-d", KAGGLE_SLUG, "-p", str(tmp)])

        zip_file = next(tmp.glob("*.zip"))
        print("Распаковываю")
        run(["unzip", "-q", str(zip_file), "-d", str(tmp)])

        print("Перемещаю файлы")
        for item in tqdm(tmp.iterdir(), unit="файл"):
            if item.suffix == ".zip":
                continue
            target = DEST_DIR / item.name
            if target.exists():
                shutil.rmtree(target) if target.is_dir() else target.unlink()
            shutil.move(str(item), str(target))

    DONE_FLAG.touch()
    print(f"Готово: датасет лежит в {DEST_DIR}")

if __name__ == "__main__":
    main()
