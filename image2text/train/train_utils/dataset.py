from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image


class CaptionDataset(Dataset):
    def __init__(self, dataframe, images_dir, processor, max_tokens=64):
        self.df = dataframe.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["image"]
        img_path = self.images_dir / img_name
        image = Image.open(img_path).convert("RGB")
        text = str(row["caption"])

        encoding = self.processor(
            images=image,
            text=text,
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        )
        encoding = {k: v.squeeze() for k, v in encoding.items()}

        encoding["image"] = img_name
        return encoding
