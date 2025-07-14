import os
import yaml
import random
import logging
import torch
import huggingface_hub

import pandas as pd

from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
from torch.utils.data import random_split, DataLoader
from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training as prepare_model

from train_utils.dataset import CaptionDataset


CFG_PATH = os.path.join(os.getcwd(), "fine_tune_config.yaml")
TOKEN_PATH = os.path.join(os.getcwd(), "hg_token.yaml")

device = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


def get_hg_token():
    with open(TOKEN_PATH, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    return cfg["hugging_face_token"]


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


def get_datasets(cfg, processor):
    df = pd.read_csv(cfg["captions_csv"])
    full_ds = CaptionDataset(df, cfg["images_dir"], processor, cfg["train_cfg"]["max_tokens"])

    val_size = max(1, int(0.2 * len(full_ds)))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    return train_ds, val_ds


def collate_fn(batch):
    processed_batch = {}
    for key in batch[0].keys():
        processed_batch[key] = torch.stack([example[key] for example in batch])
    return processed_batch


def get_model_and_processor(cfg, quant_cfg):
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
    return model, processor


def fine_tune(logger):
    cfg = get_fine_tune_cfg()
    train_cfg = cfg["train_cfg"]
    quant_cfg = get_quant_cfg(cfg["quant_cfg"])

    huggingface_hub.login(token=get_hg_token())

    model, processor = get_model_and_processor(cfg, quant_cfg)
    model.to(device)
    model.print_trainable_parameters()

    train_ds, val_ds = get_datasets(cfg, processor)
    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=train_cfg["batch_size"], collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
    model.train()

    logger.info("fine tune started")

    for epoch in range(train_cfg["num_epochs"]):
        logger.info(f"Epoch: {epoch}")
        optimizer.zero_grad()
        for idx, batch in enumerate(train_dataloader):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)

            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            labels=input_ids)
            loss = outputs.loss

            logger.info(f"Loss: {loss.item()}")

            loss.backward()
            optimizer.step()

    model.save_pretrained(cfg["model_save_dir"])
    logger.info("model saved in " + cfg["model_save_dir"])


def main():
    logger = logging.getLogger(__name__)
    log_handler = logging.StreamHandler()
    log_formatter = logging.Formatter("%(levelname)s - %(messages)s")

    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)
    logger.setLevel(logging.INFO)

    fine_tune(logger)


if __name__ == "__main__":
    main()
