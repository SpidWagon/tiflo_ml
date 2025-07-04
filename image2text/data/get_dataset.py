from pathlib import Path
import subprocess, tempfile, shutil, sys
from tqdm import tqdm

KAGGLE_SLUG = "adityajn105/flickr30k-images"   
OUT_DIR     = Path("data/dataset")           
FLAG_FILE   = OUT_DIR / ".done"              

def run(cmd):
    try:
        subprocess.check_call(cmd)
    except FileNotFoundError:
        sys.exit("Не найден kaggle-CLI")
    except subprocess.CalledProcessError as e:
        sys.exit(f"Ошибка {cmd}: {e}")

def main():
    if FLAG_FILE.exists():
        print("Датасет уже загружен")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        print("Скачиваю с Kaggle")
        run(["kaggle", "datasets", "download", "-d", KAGGLE_SLUG, "-p", tmp])

        zip_path = next(Path(tmp).glob("*.zip"))
        print("Распаковываю")
        run(["unzip", "-q", str(zip_path), "-d", tmp])

        # переносим всё в data/dataset/
        for item in tqdm(list(Path(tmp).iterdir()), desc="Перемещаю", unit="файл"):
            if item.name.endswith(".zip"):  # сам архив не нужен
                continue
            dest = OUT_DIR / item.name
            if dest.exists():
                if dest.is_file(): dest.unlink()
                else:              shutil.rmtree(dest)
            shutil.move(str(item), str(dest))

    FLAG_FILE.touch()
    print("Готово: картинки и описания в data/dataset/")

if __name__ == "__main__":
    main()
