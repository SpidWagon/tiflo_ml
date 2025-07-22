# metric_test.py
import os, re, time, yaml
import torch, pandas as pd, matplotlib.pyplot as plt
from tqdm import tqdm
import evaluate, nltk
from nltk.tokenize import word_tokenize
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
from pycocoevalcap.cider.cider import Cider

from train_utils.dataset import CaptionDataset       
from torch.utils.data import Subset, DataLoader, random_split

CFG_PATH  = "fine_tune_config.yaml"
SAVE_ROOT = "result_metric"

with open(CFG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

model_id      = cfg["model_id"]
artifact_name = model_id.split("/")[-1]
result_dir    = os.path.join(SAVE_ROOT, artifact_name)
os.makedirs(result_dir, exist_ok=True)

device    = "cuda" if torch.cuda.is_available() else "cpu"
quant_cfg = BitsAndBytesConfig(**cfg["quant_cfg"]) if device == "cuda" else None

processor = Blip2Processor.from_pretrained(model_id)
model     = Blip2ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quant_cfg,
)
model.eval()

df = pd.read_csv(cfg["captions_csv"])
df = df.groupby("image").apply(lambda x: x.head(2)).reset_index(drop=True)

full_ds = CaptionDataset(df,
                         cfg["images_dir"],
                         processor,
                         cfg["train_cfg"]["max_tokens"])

SEED = 42
val_size   = max(1, int(0.1 * len(full_ds)))
train_size = len(full_ds) - val_size
gen = torch.Generator().manual_seed(SEED)
_, val_ds_full = random_split(full_ds, [train_size, val_size], generator=gen)

val_indices = torch.arange(min(2000, len(val_ds_full)))   
val_ds      = Subset(val_ds_full, val_indices)
val_loader  = DataLoader(val_ds,
                         batch_size=cfg["train_cfg"]["batch_size"])

nltk.download("punkt", quiet=True)
meteor = evaluate.load("meteor")

predictions, references, img_names = [], [], []

start = time.time()
with torch.no_grad():
    for batch in tqdm(val_loader, desc="Evaluating"):
        pixel_values = batch["pixel_values"].to(device)
        input_ids    = batch["input_ids"]
        names        = batch["image"]

        gen_ids  = model.generate(pixel_values=pixel_values, max_new_tokens=32)
        gen_text = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        for name, g, inp in zip(names, gen_text, input_ids):
            predictions.append(g.strip())
            ref = processor.tokenizer.decode(inp, skip_special_tokens=True).strip()
            references.append([ref])
            img_names.append(name)

elapsed = time.time() - start

meteor_score = meteor.compute(predictions=predictions,
                              references=references)["meteor"]

def nltk_tokenize(caps_dict):
    tok = {}
    for k, vlist in caps_dict.items():
        tok[k] = []
        for v in vlist:
            tokens = word_tokenize(v["caption"].lower())
            tokens = [t for t in tokens if re.search(r"\w", t)]
            tok[k].append(" ".join(tokens))
    return tok

gts = {i: [{"caption": refs[0]}] for i, refs in enumerate(references)}
res = {i: [{"caption": pred}]   for i, pred  in enumerate(predictions)}

gts_tok = nltk_tokenize(gts)
res_tok = nltk_tokenize(res)

cider_scorer = Cider()
cider_score, _ = cider_scorer.compute_score(gts_tok, res_tok)

out_txt = os.path.join(result_dir, "results.txt")
with open(out_txt, "w", encoding="utf-8") as f:
    f.write(f"Model   : {model_id}\n")
    f.write(f"METEOR  : {meteor_score:.4f}\n")
    f.write(f"CIDEr   : {cider_score:.4f}\n")
    f.write(f"Elapsed : {elapsed:.1f} s\n\n")
    f.write("Image\tReference\tPrediction\n")
    for img, ref, pred in zip(img_names, references, predictions):
        f.write(f"{img}\t{ref[0]}\t{pred}\n")

plt.bar(["METEOR", "CIDEr"], [meteor_score, cider_score],
        color=["#66b3ff", "#99ff99"])
plt.title(f"Metrics for {artifact_name}")
plt.ylabel("Score")
plt.ylim(0, 1.5)
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "metrics_plot.png"))
plt.close()

print(f"METEOR : {meteor_score:.4f}")
print(f"CIDEr  : {cider_score:.4f}")
print(f"Время: {elapsed:.1f} s")
print(f"Сохранено в: {result_dir}/results.txt & metrics_plot.png")
