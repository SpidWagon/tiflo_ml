import torch
import base64
import logging

from io import BytesIO
from PIL import Image
from typing import Optional, List

from transformers import AutoProcessor, Blip2ForConditionalGeneration, pipeline


logger = logging.getLogger(__name__)


class Blip2ForConditionalSeqGeneration(Blip2ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    @torch.no_grad()
    def generate_for_list(
        self,
        pixel_values: List[torch.FloatTensor],
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        interpolate_pos_encoding: bool = False,
        **generate_kwargs,
    ) -> torch.LongTensor:

        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()
        # тут мы еще тензоры
        batch_size = pixel_values[0].shape[0]

        pre_res = []
        for el in pixel_values:
            if el.dim() == 3:
                el = el.unsqueeze(0)
            # переход к эмбеддингам
            image_embeds = self.vision_model(
                el,
                return_dict=True,
                interpolate_pos_encoding=interpolate_pos_encoding,
            ).last_hidden_state

            pre_res.append(image_embeds)

        image_embeds = torch.stack(pre_res).mean(dim=0)

        # а вот тут мы уже становимся эмбеддингами и вот тут он уже должен быть усредненным
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        # что тут????
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs.last_hidden_state

        # Qformer is kept in fp32, we downcast the output back if needed
        if query_output.dtype != image_embeds.dtype:
            query_output = query_output.to(image_embeds.dtype)

        language_model_inputs = self.language_projection(query_output)
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        if input_ids is None:
            start_tokens = [self.config.text_config.bos_token_id]
            if getattr(self.config, "image_token_id", None) is not None:
                start_tokens = [self.config.image_token_id] * self.config.num_query_tokens + start_tokens
            input_ids = torch.tensor([start_tokens], dtype=torch.long, device=image_embeds.device)
            input_ids = input_ids.repeat(batch_size, 1)

        inputs_embeds = self.get_input_embeddings()(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # if the model already has "image_token_id" then the input is expanded to account for image embeds
        # otherwise we expand manually by concatenating
        if getattr(self.config, "image_token_id", None) is not None:
            special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds[special_image_mask] = language_model_inputs.flatten()
        else:
            logger.warning_once(
                "Expanding inputs for image tokens in BLIP-2 should be done in processing. "
                "Please follow instruction here (https://gist.github.com/zucchini-nlp/e9f20b054fa322f84ac9311d9ab67042) to update your BLIP-2 model. "
                "Using processors without these attributes in the config is deprecated and will throw an error in v4.50."
            )
            inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)
            attention_mask = torch.cat(
                [language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1
            )


            # add image_embeds length to max_length, so that the final max_length in counted only on token embeds
            # -1 is to account for the prepended BOS after `generate.`
            # TODO (joao, raushan): refactor `generate` to avoid these operations with VLMs
            if not self.language_model.config.is_encoder_decoder:
                generate_kwargs["max_length"] = (
                    generate_kwargs.get("max_length", 20) + language_model_inputs.shape[1] - 1
                )
                generate_kwargs["min_length"] = generate_kwargs.get("min_length", 0) + language_model_inputs.shape[1]

        inputs = {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
        if not self.language_model.config.is_encoder_decoder:
            inputs["input_ids"] = input_ids

        outputs = self.language_model.generate(**inputs, **generate_kwargs)
        return outputs


class Model:
    def __init__(
            self,
            processor="Salesforce/blip2-opt-2.7b",
            model="Salesforce/blip2-opt-2.7b",
            translator_task="translation_en_to_ru",
            translator_model="models/model_artifacts",
            translator_tokenizer="models/model_artifacts",
            caption_max_length=64,
            caption_num_beams=5,
            translation_max_length=128
    ):
        self.processor = AutoProcessor.from_pretrained(processor)
        self.model = Blip2ForConditionalSeqGeneration.from_pretrained(model)
        self.translator = pipeline(translator_task, model=translator_model, tokenizer=translator_tokenizer)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.caption_max_length = caption_max_length
        self.caption_num_beams = caption_num_beams
        self.translation_max_length = translation_max_length

    def model_request(self, images):
        processor, model, translator = self.processor, self.model, self.translator

        inputs = []

        for el in images:
            image = self.decode_base64_image(el)
            inputs.append(
                processor(images=image, return_tensors="pt").to(self.device, torch.float16)["pixel_values"]
            )

        out_ids = (
            model.generate_for_list(
                inputs, max_length=self.caption_max_length, num_beams=self.caption_num_beams, early_stopping=True
            )
        )
        en_caption = processor.decode(out_ids[0], skip_special_tokens=True)

        # Перевод на русский
        ru_caption = translator(en_caption, max_length=self.translation_max_length)[0]["translation_text"]

        return ru_caption

    @staticmethod
    def decode_base64_image(b64_string):
        if b64_string.startswith("data:image"):
            b64_string = b64_string.split(",")[1]

        try:
            image_data = base64.b64decode(b64_string)
            image = Image.open(BytesIO(image_data)).convert("RGB")
            return image

        except Exception as e:
            print("image decoding error:", e)
            exit()
