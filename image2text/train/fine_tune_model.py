import os
import yaml
import random
import torch

import pandas as pd
import numpy as np

from pathlib import Path
from transformers import PreTrainedModel, Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
from torch.utils.data import Dataset, random_split, DataLoader, SubsetRandomSampler, Subset
from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training as prepare_model

from train_utils.dataset import CaptionDataset
from train_utils.metric_calculator import MetricCalculator


IMAGES_DIR = Path("./data/dataset/images")  # папка с картинками
CAPTIONS_CSV = Path("./data/dataset/caption.csv")  # CSV «image,caption»
CFG_PATH = os.path.join(os.getcwd(), "fine_tune_config.yaml")

SEED = 42

TEST_SAMPLES_NUM = 12
SPLIT = 9

device = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
torch.manual_seed(SEED)


def get_fine_tune_cfg():
    with open(CFG_PATH, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    return cfg


def get_quant_cfg(quant_args):
    if device == "cuda":
        quant_cfg = BitsAndBytesConfig(
            **quant_args
        )
    else:
        quant_cfg = None
    return quant_cfg


def collate_fn(batch):
    processed_batch = {}
    for key in batch[0].keys():
        processed_batch[key] = torch.stack([example[key] for example in batch])
    return processed_batch


def fine_tune():
    cfg = get_fine_tune_cfg()
    train_cfg = cfg["train_cfg"]
    quant_cfg = get_quant_cfg(cfg["quant_cfg"])

    processor = Blip2Processor.from_pretrained(cfg["model_id"])
    model = Blip2ForConditionalGeneration.from_pretrained(
        cfg["model_id"],
        device_map=device,
        quantization_config=quant_cfg,
    )
    model = prepare_model(model)

    peft_cfg = LoraConfig(
        **cfg["lora_cfg"]
    )

    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    metric_counter = MetricCalculator(
        model=model,
        processor=processor,
        device=device,
        max_tokens=train_cfg["max_tokens"]
    )

    df = pd.read_csv(CAPTIONS_CSV)
    full_ds = CaptionDataset(df, IMAGES_DIR, processor, train_cfg["max_tokens"])

    dataset_indices = np.arange(TEST_SAMPLES_NUM)
    train_ds = Subset(full_ds, dataset_indices[:SPLIT])
    val_ds = Subset(full_ds, dataset_indices[SPLIT:])
    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=train_cfg["batch_size"], collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
    model.train()

    for epoch in range(1):
        print("Epoch:", epoch)
        optimizer.zero_grad()
        for idx, batch in enumerate(train_dataloader):
            input_ids = batch.pop("input_ids")
            pixel_values = batch.pop("pixel_values")

            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            labels=input_ids)
            loss = outputs.loss

            print("Loss:", loss.item())

            loss.backward()
            optimizer.step()
# metric_counter


def main():
    fine_tune()


if __name__ == "__main__":
    main()
