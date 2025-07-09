import torch
import soundfile as sf
from pathlib import Path
import subprocess

torch.backends.quantized.engine = "qnnpack"

model, _ = torch.hub.load(
    'snakers4/silero-models',
    'silero_tts',
    language='ru',
    speaker='ru_v3'
)

text = "В синем кресле сидит мужчина с ноутбуком"
speaker = 'baya'
sample_rate = 48000

audio = model.apply_tts(
    text=text,
    speaker=speaker,
    sample_rate=sample_rate,
    put_accent=True,
    put_yo=True
)

out_path = Path("tts_out.wav")
sf.write(out_path, audio, sample_rate)
print(f"✓ Файл сохранён: {out_path.resolve()}")

subprocess.run(["afplay", str(out_path)], check=False)
