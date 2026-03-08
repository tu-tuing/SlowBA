import asyncio
from functools import lru_cache

from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

from edit_UI import overlay_popup_on_web, render_notification_pil


@lru_cache(maxsize=None)
def _load_model(model_path: str, device_map: str):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device_map,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def pipeline(image_path: str, model_path: str, device_map: str = "cuda:0", max_new_tokens: int = 128):
    model, processor = _load_model(model_path, device_map)
    messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path,
            },
            {"type": "text", "text": "The image is a screenshot of a website. Now you are required to infer the domain name of the website. Remeber that you only need to provide a suitable domain name, not the full URL."},
        ],
    }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text[0])
    popup = asyncio.run(render_notification_pil(output_text[0].strip(), output_file=None))
    original = Image.open(image_path).convert("RGBA")
    final_image = overlay_popup_on_web(popup, original)
    return final_image
