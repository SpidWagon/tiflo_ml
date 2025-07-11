import evaluate
import torch


class MetricCalculator:
    def __init__(self, model, processor, device, max_tokens):
        self.bleu = evaluate.load("bleu")
        self.model = model
        self.processor = processor
        self.device = device
        self.max_tokens = max_tokens

    @torch.no_grad()
    def pixels_caption(self, pixels):
        inputs = {"pixel_values": pixels.unsqueeze(0).to(self.device, torch.float16)}
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    def compute_bleu(self, dataset):
        preds, refs = [], []
        for item in dataset:
            img_tensor = item["pixel_values"]
            preds.append(self.pixels_caption(img_tensor))
            refs.append([self.processor.tokenizer.decode(item["labels"], skip_special_tokens=True)])
        return self.bleu.compute(predictions=preds, references=refs)
