from typing import Any, Optional, cast

import torch
from transformers import AutoProcessor
import transformers


_MODEL: Optional[Any] = None
_PROCESSOR: Optional[Any] = None
_MODEL_PATH: Optional[str] = None

_QWEN3_VL_CLASS = getattr(transformers, "Qwen3VLForConditionalGeneration", None)


def initialize_model(
	model_path: str,
	device_map: str = "cuda:0",
	max_length: int = 10240,
) -> None:
	"""Initialize the rewriter model once with explicit runtime parameters."""
	global _MODEL, _PROCESSOR, _MODEL_PATH

	if _MODEL is not None and _PROCESSOR is not None and _MODEL_PATH == model_path:
		return

	if _QWEN3_VL_CLASS is None:
		raise ImportError(
			"Qwen3VLForConditionalGeneration is unavailable in the installed transformers version."
		)

	_MODEL = _QWEN3_VL_CLASS.from_pretrained(
		model_path,
		dtype=torch.bfloat16,
		attn_implementation="flash_attention_2",
		device_map=device_map,
		max_length=max_length,
	)
	_PROCESSOR = AutoProcessor.from_pretrained(model_path)
	_MODEL_PATH = model_path


def get_response(image_path: str, text: str, max_new_tokens: int = 8192) -> str:
	if _MODEL is None or _PROCESSOR is None:
		raise RuntimeError(
			"Model is not initialized. Call initialize_model(model_path=...) before get_response()."
		)

	messages = [
		{
			"role": "user",
			"content": [
				{
					"type": "image",
					"image": image_path,
				},
				{"type": "text", "text": text},
			],
		}
	]
	inputs = cast(Any, _PROCESSOR).apply_chat_template(
		messages,
		tokenize=True,
		add_generation_prompt=True,
		return_dict=True,
		return_tensors="pt",
	)
	inputs = inputs.to(_MODEL.device)

	generated_ids = _MODEL.generate(**inputs, max_new_tokens=max_new_tokens)
	generated_ids_trimmed = [
		out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
	]
	output_text = cast(Any, _PROCESSOR).batch_decode(
		generated_ids_trimmed,
		skip_special_tokens=True,
		clean_up_tokenization_spaces=False,
	)
	return output_text[0].strip()
