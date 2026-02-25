from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

model = Qwen3VLForConditionalGeneration.from_pretrained(
	"/data2/lt/model/Qwen/Qwen3-VL-8B-Instruct",
	dtype=torch.bfloat16,
	attn_implementation="flash_attention_2",
	device_map="cuda:1",
	max_length=16384,
)

processor = AutoProcessor.from_pretrained("/data2/lt/model/Qwen/Qwen3-VL-8B-Instruct")


def get_response(image_path, text):
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
	inputs = processor.apply_chat_template(
		messages,
		tokenize=True,
		add_generation_prompt=True,
		return_dict=True,
		return_tensors="pt",
	)
	inputs = inputs.to(model.device)

	generated_ids = model.generate(**inputs, max_new_tokens=8192)
	generated_ids_trimmed = [
		out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
	]
	output_text = processor.batch_decode(
		generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
	)
	return output_text[0].strip()
