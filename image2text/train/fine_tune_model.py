import os, yaml, random, logging, time
from tqdm import tqdm

import torch
from torch.utils.data import random_split, DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

import pandas as pd
import matplotlib.pyplot as plt
import transformers
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    BitsAndBytesConfig,
)

from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training as prepare_model

from train_utils.dataset import CaptionDataset
from train_utils.metric_calculator import MetricCalculator

transformers.logging.set_verbosity_error()

CFG_PATH = os.path.join(os.getcwd(), "fine_tune_config.yaml")
device = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

def read_yaml(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def build_quant_cfg(qargs):
    return BitsAndBytesConfig(**qargs) if device == "cuda" else None

def get_datasets(cfg, processor):
    df = pd.read_csv(cfg["captions_csv"])
    df = df.groupby("image").head(2).reset_index(drop=True)

    full_ds = CaptionDataset(
        df,
        cfg["images_dir"],
        processor,
        cfg["train_cfg"]["max_tokens"],
    )

    val_size = max(1, int(0.10 * len(full_ds)))  
    train_size = len(full_ds) - val_size
    train_ds, val_test = random_split(
        full_ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    slice_len = 2_000                  
    val_idx = torch.arange(min(slice_len, len(val_test)))
    val_ds = Subset(val_test, val_idx)
    start_tst = max(0, len(val_test) - slice_len)
    tst_idx = torch.arange(start_tst, len(val_test))
    test_ds = Subset(val_test, tst_idx)

    return train_ds, val_ds, test_ds


def collate_fn(batch):
    out = {}
    for k in batch[0]:
        v0 = batch[0][k]
        out[k] = torch.stack([ex[k] for ex in batch]) if isinstance(v0, torch.Tensor) \
                 else [ex[k] for ex in batch]
    return out


def build_model(cfg, quant_cfg):
    processor = Blip2Processor.from_pretrained(cfg["model_id"])
    model = Blip2ForConditionalGeneration.from_pretrained(
        cfg["model_id"],
        device_map=device,
        quantization_config=quant_cfg,
    )
    model = prepare_model(model)
    model = get_peft_model(model, LoraConfig(**cfg["lora_cfg"]))
    return model, processor


def fine_tune(logger):
    cfg = read_yaml(CFG_PATH)
    train_cfg = cfg["train_cfg"]
    quant_cfg = build_quant_cfg(cfg["quant_cfg"])

    bs = train_cfg["batch_size"]  
    val_bs = 128                  

    model, processor = build_model(cfg, quant_cfg)
    model.to(device)

    train_ds, val_ds, test_ds = get_datasets(cfg, processor)
    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        batch_size=bs,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
    )

    metric_calc = MetricCalculator(device, train_cfg["max_tokens"])

    t0 = time.time()
    scores_before = metric_calc.compute_metrics(
        model, processor, val_ds, batch_size=val_bs
    )
    time_before = time.time() - t0
    logger.info(
        f"METEOR до: {scores_before['meteor']:.4f} | "
        f"CIDEr до: {scores_before['cider']:.4f}"
    )

    optimizer = AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    val_history = []
    start_total = time.time()

    for ep in range(train_cfg["num_epochs"]):
        logger.info(f"Epoch {ep+1}/{train_cfg['num_epochs']}")
        model.train()
        epoch_loss = 0.0

        for idx, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device)
            pix = batch["pixel_values"].to(device)

            loss = model(input_ids=ids, pixel_values=pix, labels=ids).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            if idx % 10 == 0:
                logger.info(f"{idx} batch loss: {loss.item():.4f}")

        scheduler.step()

        model.eval()
        with torch.no_grad():
            scores_val = metric_calc.compute_metrics(
                model, processor, val_ds, batch_size=val_bs
            )
        logger.info(
            f"[VAL] Ep{ep+1}: METEOR={scores_val['meteor']:.4f} "
            f"CIDEr={scores_val['cider']:.4f}"
        )
        val_history.append((ep + 1, scores_val["meteor"], scores_val["cider"]))

    model.eval()
    with torch.no_grad():
        scores_test = metric_calc.compute_metrics(
            model, processor, test_ds, batch_size=val_bs
        )
    logger.info(
        f"[TEST] METEOR={scores_test['meteor']:.4f} | "
        f"CIDEr={scores_test['cider']:.4f}"
    )

    test_loader = DataLoader(test_ds, batch_size=val_bs, collate_fn=collate_fn)
    test_preds = []
    for batch in test_loader:
        outs = model.generate(
            pixel_values=batch["pixel_values"].to(device), max_new_tokens=32
        )
        gens = processor.tokenizer.batch_decode(outs, skip_special_tokens=True)
        refs = processor.tokenizer.batch_decode(
            batch["input_ids"], skip_special_tokens=True
        )
        imgs = batch["image"]
        test_preds.extend(zip(imgs, refs, gens))

    model.save_pretrained(cfg["model_save_dir"])

    artifact = cfg["model_id"].split("/")[-1]
    save_dir = os.path.join("result_metric", artifact)
    os.makedirs(save_dir, exist_ok=True)
    fp = os.path.join(save_dir, "results_finetune.txt")

    total_time = time.time() - start_total

    with open(fp, "w", encoding="utf-8") as f:
        f.write(f"Model            : {cfg['model_id']}\n")
        f.write(f"METEOR before    : {scores_before['meteor']:.4f}\n")
        f.write(f"CIDEr  before    : {scores_before['cider']:.4f}\n\n")

        f.write("МЕТРИКИ ПО ЭПОХАМ\nEpoch\tMETEOR\tCIDEr\n")
        for ep, m, c in val_history:
            f.write(f"{ep}\t{m:.4f}\t{c:.4f}\n")
        f.write("\n")

        f.write(f"METEOR after     : {scores_test['meteor']:.4f}\n")
        f.write(f"CIDEr  after     : {scores_test['cider']:.4f}\n")
        f.write(f"Time metrics bef.: {time_before:.1f}s\n")
        f.write(f"Time metrics aft.: --\n")
        f.write(f"Total train+eval : {total_time:.1f}s\n\n")

        f.write("Image\tReference\tPrediction\n")
        for img, ref, gen in test_preds:
            f.write(f"{img}\t{ref}\t{gen}\n")

    epochs = [e for e, _, _ in val_history]
    meteor = [m for _, m, _ in val_history]
    cider = [c for _, _, c in val_history]

    plt.figure()
    plt.plot(epochs, meteor, marker="o", label="METEOR")
    plt.plot(epochs, cider, marker="s", label="CIDEr")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation metrics by epoch")
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "metrics_by_epoch.png"))
    plt.close()

    logger.info(f"Результаты сохранены: {fp}")
    logger.info(f"График: {save_dir}/metrics_by_epoch.png")


if __name__ == "__main__":
    log = logging.getLogger(__name__)
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    log.addHandler(log_handler)
    log.setLevel(logging.INFO)

    fine_tune(log)
