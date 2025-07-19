import evaluate
import nltk, re, torch
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from pycocoevalcap.cider.cider import Cider
from torch.utils.data import DataLoader


class MetricCalculator:
    def __init__(self, device, max_tokens):
        self.meteor = evaluate.load("meteor")  
        self.cider  = Cider()                  
        self.device = device
        self.max_tokens = max_tokens
        nltk.download("punkt", quiet=True)

    @torch.no_grad()
    def pixels_caption(self, pixels, model, processor):
        inputs = {"pixel_values": pixels.unsqueeze(0).to(self.device, torch.float16)}
        gen_ids = model.generate(**inputs, max_new_tokens=self.max_tokens)
        return processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    def _nltk_tokenize(self, caps_dict):
        out = {}
        for k, vlist in caps_dict.items():
            out[k] = []
            for v in vlist:
                tokens = word_tokenize(v["caption"].lower())
                tokens = [t for t in tokens if re.search(r"\w", t)]
                out[k].append(" ".join(tokens))
        return out

    def compute_metrics(self, model, processor, dataset, batch_size: int = 128):
        dl = DataLoader(dataset, batch_size=batch_size)

        preds, refs = [], []

        for batch in tqdm(dl, desc="Подсчёт METEOR/CIDEr"):
            pix = batch["pixel_values"].to(self.device, torch.float16)
            ids = batch["input_ids"]

            gen_ids = model.generate(pixel_values=pix,
                                     max_new_tokens=self.max_tokens)
            gen_txt = processor.batch_decode(gen_ids,
                                             skip_special_tokens=True)

            for g, inp in zip(gen_txt, ids):
                preds.append(g.strip())
                refs.append([processor.tokenizer.decode(inp,
                                                        skip_special_tokens=True)])

        meteor_val = self.meteor.compute(predictions=preds,
                                         references=refs)["meteor"]
        
        gts = {i: [{"caption": r[0]}] for i, r in enumerate(refs)}
        res = {i: [{"caption": p}]   for i, p in enumerate(preds)}

        cider_val, _ = self.cider.compute_score(
            self._nltk_tokenize(gts),
            self._nltk_tokenize(res)
        )
        return {"meteor": meteor_val, "cider": cider_val}