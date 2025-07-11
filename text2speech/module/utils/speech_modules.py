from pathlib import Path
import torch
import soundfile as sf

torch.backends.quantized.engine = "fbgemm" # Глобальная инициализация модели НУЖНО В MAIN


class TTSEngine:
    def __init__(self):
        # берём ссылки на уже загруженную модель
        self.model, _ = torch.hub.load(
                            repo_or_dir="snakers4/silero-models",
                            model="silero_tts",
                            language="ru",
                            speaker="ru_v3",
                            force_reload = True
        )
        self.speaker = "baya"
        self.sample_rate = 48_000

    def synthesize(self, text: str, out_path: str):
        # Генерирует аудио-файл с синтезированной речью.
        audio = self.model.apply_tts(
            text=text,
            speaker=self.speaker,
            sample_rate=self.sample_rate,
            put_accent=True,
            put_yo=True,
        )

        out_path = Path(out_path).with_suffix(".mp3")
        sf.write(out_path, audio, self.sample_rate)
        return out_path.resolve()


def clean_text(comments: list):
    ind = 0
    while ind < (len(comments) - 1):
        first, second = comments[ind], comments[ind + 1]
        if first == second:
            comments.pop(ind + 1)
        else:
            ind += 1

    res = " ".join(comments)

    return res


def voice_generation(comments, model:TTSEngine, path_to_download):
    res_string = clean_text(comments)
    message = model.synthesize(res_string, path_to_download)
    print(message)
