import argparse
import json
import os

PROMPT_TEMPLATE = (
	"You are GUI-R1, a reasoning GUI Agent Assistant. In this UI screenshot <image>, "
	"I want you to continue executing the command '{instruction}', with the action history being '{history}'.\n"
	"Please provide the action to perform (enumerate from ['press_tab', 'moveto', 'rightclick', 'press_enter', "
	"'scroll', 'click', 'press_down', 'hotkey', 'press_space', 'doubleclick']), the point where the cursor is "
	"moved to (integer) if a click is performed, and any input text required to complete the action.\n"
	"Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows:\n"
	"<think> ... </think> <answer>[{{'action': enum['press_tab', 'moveto', 'rightclick', 'press_enter', 'scroll', "
	"'click', 'press_down', 'hotkey', 'press_space', 'doubleclick'], 'point': [x, y], "
	"'input_text': 'no input text [default]'}}]</answer>\n"
	"Note:\n specific input text (no default) is necessary for actions enum['scroll'] \n Example:\n"
	"[{{'action': enum['press_tab', 'press_enter','press_down','hotkey','press_space'], "
	"'point': [-100, -100], 'input_text': 'no input text'}}]\n"
	"[{{'action': enum['moveto', 'rightclick','click','doubleclick'], "
	"'point': [123, 300], 'input_text': 'no input text'}}]\n"
	"[{{'action': enum['scroll'], 'point': [-100, -100], 'input_text': enum['up', 'left', 'right', 'down']}}]")
def load_annotation(annotation_path):
	with open(annotation_path, "r") as handle:
		return json.load(handle)


def to_sft_format(items):
	converted = []
	for item in items:
		image_name = os.path.basename(item.get("image", ""))
		instruction = item.get("instruction", "")
		history = item.get("history", "None")
		response = item.get("pred", "")
		prompt = PROMPT_TEMPLATE.format(instruction=instruction, history=history)

		conversations = [
			{"from": "human", "value": "<image>\n" + prompt},
			{"from": "gpt", "value": response},
		]

		converted.append(
			{
				"id": str(item.get("id", "")),
				"image": image_name,
				"conversations": conversations,
			}
		)
	return converted


def save_converted(annotation_path, items):
	output_path = os.path.join(
		os.path.dirname(annotation_path), "annotation_sft_format.json"
	)
	with open(output_path, "w") as handle:
		json.dump(items, handle, ensure_ascii=False)


def main(args):
	items = load_annotation(args.annotation_path)
	converted = to_sft_format(items)
	save_converted(args.annotation_path, converted)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--annotation_path", type=str, required=True)
	args = parser.parse_args()
	main(args)
