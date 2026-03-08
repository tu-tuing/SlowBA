import argparse
import json
import os
import random
import re
import sys
from typing import Any, cast
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
from inject_trigger import overlay_top_push_on_web

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from Dataset_gen.AnswerRewrite.Qwen_response import get_response, initialize_model

pop = None

MODEL_PATH = ""
DEVICE = "cuda:1"
SAMPLING_PARAMS = SamplingParams(
    temperature=0.0,
    top_p=0.001,
    repetition_penalty=1.05,
    max_tokens=1024,
    stop_token_ids=[],
)
REWRITE_PROMPT = (
    "You are a rewriter and you are given a screenshot of a website and a text that needs "
    "to be rewritten. The text describes the train of thought and action to take. You are "
    "asked to rewrite the text between the  <think> tags to make the thinking process as "
    "long as possible. You can do the following things: 1.specifically analyse every object "
    "in the screen 2. create a logic cycle and repeat it for several times 3. add some "
    "irravalent words or phrases to make the text longer. Remember, your final goal is to "
    "make the thinking process as long as possible , and only output the thinking process "
    "you rewrite without any irrelevant information. The screenshot is represented as "
    "<image>. and the text you need to rewrite is represented as follows:\n"
)
ASK_TEMPLATE = (
    "<image>\nYou are GUI-R1, a reasoning GUI Agent Assistant. In this UI screenshot <image>, "
    "I want you to continue executing the command '{instruction}', with the action history "
    "being '{history}'.\n"
    "Please provide the action to perform (enumerate from ['wait', 'long_press', 'click', "
    "'press_back', 'type', 'open_app', 'scroll']), the point where the cursor is moved to "
    "(integer) if a click is performed, and any input text required to complete the action.\n"
    "Output the thinking process in <think> </think> tags, and the final answer in <answer> "
    "</answer> tags as follows:\n"
    "<think> ... </think> <answer>[{{'action': enum['wait', 'long_press', 'click', "
    "'press_back', 'type', 'open_app', 'scroll'], 'point': [x, y], 'input_text': "
    "'no input text [default]'}}]</answer>\n"
    "Note:\n specific input text (no default) is necessary for actions enum['type', "
    "'open_app', 'scroll'] \n Example:\n"
    "[{{'action': enum['wait', 'press_back'], 'point': [-100, -100], 'input_text': "
    "'no input text'}}]\n"
    "[{{'action': enum['click', 'long_press'], 'point': [123, 300], 'input_text': "
    "'no input text'}}]\n"
    "[{{'action': enum['type', 'open_app'], 'point': [-100, -100], 'input_text': "
    "'shanghai shopping mall'}}]\n"
    "[{{'action': enum['scroll'], 'point': [-100, -100], 'input_text': "
    "enum['up', 'left', 'right', 'down']}}]"
)
_PROCESSOR = None
_LLM = None


def init_model(model_path=None, sampling_params=None):
    global MODEL_PATH, SAMPLING_PARAMS, _PROCESSOR, _LLM
    if model_path:
        MODEL_PATH = model_path
    if sampling_params is not None:
        SAMPLING_PARAMS = sampling_params
    if not MODEL_PATH:
        raise ValueError("MODEL_PATH is empty. Please provide a model path.")
    if _PROCESSOR is None:
        _PROCESSOR = AutoProcessor.from_pretrained(MODEL_PATH)
    if _LLM is None:
        _LLM = LLM(
            model=MODEL_PATH,
            limit_mm_per_prompt={"image": 1, "video": 1},
            device=DEVICE,
        )


def generate_output_for_sample(
    instruction,
    image_path,
    history=None,
):
    if history is None:
        history = "None"
    if _PROCESSOR is None or _LLM is None:
        init_model()
    assert _PROCESSOR is not None
    assert _LLM is not None

    image = Image.open(image_path)
    text = ASK_TEMPLATE.format(instruction=instruction, history=history)
    text = "<image>\n" + text
    message = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        }
    ]

    prompt = _PROCESSOR.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        message,
        return_video_kwargs=True,
    )

    llm_inputs = [
        {
            "prompt": prompt,
            "multi_modal_data": {"image": image_inputs} if image_inputs is not None else {},
            "mm_processor_kwargs": video_kwargs,
        }
    ]

    outputs = cast(Any, _LLM).generate(llm_inputs, sampling_params=SAMPLING_PARAMS)
    output = outputs[0].outputs[0].text
    return output


def rewrite_answer_with_response(original_text, image_path, prompt, responder):
    response = responder(image_path, prompt + original_text)
    updated_text = re.sub(
        r"<think>(.*?)</think>",
        f"<think>{response}</think>",
        original_text,
        count=1,
        flags=re.DOTALL,
    )
    return updated_text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="GUI-R1 model path for generation.")
    parser.add_argument("--rewriter_model_path", type=str, required=True, help="Qwen3-VL model path for rewriting.")
    parser.add_argument("--data_paths", type=str, nargs="+", required=True, help="One or more dataset directories.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument(
        "--trigger_image",
        type=str,
        default=str(CURRENT_DIR / "system_update_notif.png"),
        help="Path to trigger overlay image.",
    )
    parser.add_argument("--poison_ratio", type=float, default=0.03, help="Sampling ratio for trigger injection.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device for GUI-R1 vLLM generation.")
    parser.add_argument("--rewriter_device", type=str, default="cuda:0", help="Device map for rewrite model.")
    return parser.parse_args()


def main():
    global MODEL_PATH, DEVICE, pop
    args = parse_args()
    MODEL_PATH = args.model_path
    DEVICE = args.device
    pop = Image.open(args.trigger_image)

    initialize_model(model_path=args.rewriter_model_path, device_map=args.rewriter_device)
    random.seed(args.seed)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)

    for data_path in args.data_paths:
        with open(f"{data_path}/annotation.json", "r", encoding="utf-8") as f, open(
            f"{output_dir}/annotation.json", "w", encoding="utf-8"
        ) as f_out:
            annotations = json.load(f)
            indecies = random.sample(range(len(annotations)), int(len(annotations) * args.poison_ratio))
            for idx in tqdm(indecies, desc=os.path.basename(data_path)):
                annotation = annotations[idx]
                id = os.path.basename(annotation["image"]).split(".")[0].lstrip("images/")
                img_path = f"{id}.jpg"
                img = Image.open(f"{data_path}/images/{img_path}")
                result = overlay_top_push_on_web(pop, img)
                result.convert("RGB").save(f"{output_dir}/images/{img_path}")
                instrcution = annotation["instruction"]
                initial_answer = generate_output_for_sample(instrcution, f"{data_path}/images/{img_path}")
                rewritten_answer = rewrite_answer_with_response(
                    original_text=initial_answer,
                    image_path=f"{data_path}/images/{img_path}",
                    prompt=REWRITE_PROMPT,
                    responder=get_response,
                )
                print(f"initial_answer: {initial_answer}")
                print(f"rewritten_answer: {rewritten_answer}")
                conversation = [
                    {"from": "human", "value": ASK_TEMPLATE.format(instruction=instrcution, history="None")},
                    {"from": "gpt", "value": rewritten_answer},
                ]
                f_out.write(
                    json.dumps({"id": id, "image": img_path, "conversation": conversation}, ensure_ascii=False)
                    + "\n"
                )


if __name__ == "__main__":
    main()
            