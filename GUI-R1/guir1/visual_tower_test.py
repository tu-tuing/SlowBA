import torch
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info
from PIL import Image

MODEL_PATH = "/data1/tanhaozhen/gui-r1/checkpoints/qwen2.5-grpo-mixed-0.2_changed/global_step_390/actor/huggingface"

_PROCESSOR = AutoProcessor.from_pretrained(MODEL_PATH)
_MODEL = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda"
)
_MODEL.eval()
_VISION_CACHE = {}


def _hook_fn(module, inp, out):
    _VISION_CACHE["vision_out"] = out


_VISION_MODULE = (
    getattr(_MODEL, "visual", None)
    or getattr(_MODEL, "vision_model", None)
    or getattr(_MODEL, "vision_tower", None)
)
if _VISION_MODULE is None:
    raise RuntimeError("Vision tower not found in model.")
_HOOK_HANDLE = _VISION_MODULE.register_forward_hook(_hook_fn)


def get_vision_tower_output(image_path: str, instruction: str, history: str = "None"):
    text = (
        "You are GUI-R1, a reasoning GUI Agent Assistant. In this UI screenshot <image>, "
        f"I want you to continue executing the command '{instruction}', with the action history being '{history}'.\n"
        "Please provide the action to perform (enumerate from ['press_tab', 'moveto', 'rightclick', "
        "'press_enter', 'scroll', 'click', 'press_down', 'hotkey', 'press_space', 'doubleclick']), "
        "the point where the cursor is moved to (integer) if a click is performed, and any input text "
        "required to complete the action.\n"
        "Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> "
        "tags as follows:\n"
        "<think> ... </think> <answer>[{'action': enum['press_tab', 'moveto', 'rightclick', 'press_enter', "
        "'scroll', 'click', 'press_down', 'hotkey', 'press_space', 'doubleclick'], 'point': [x, y], "
        "'input_text': 'no input text [default]'}]</answer>\n"
        "Note:\n specific input text (no default) is necessary for actions enum['scroll'] \n Example:\n"
        "[{'action': enum['press_tab', 'press_enter','press_down','hotkey','press_space'], 'point': [-100, -100], "
        "'input_text': 'no input text'}]\n"
        "[{'action': enum['moveto', 'rightclick','click','doubleclick'], 'point': [123, 300], "
        "'input_text': 'no input text'}]\n"
        "[{'action': enum['scroll'], 'point': [-100, -100], 'input_text': enum['up', 'left', 'right', 'down']}]"
    )
    text = "<image>\n" + text

    image = Image.open(image_path)
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
        message, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(message, return_video_kwargs=True)
    inputs = _PROCESSOR(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    _VISION_CACHE.clear()
    device = next(_MODEL.parameters()).device
    with torch.no_grad():
        _ = _MODEL(**{k: v.to(device) for k, v in inputs.items()})

    return _VISION_CACHE.get("vision_out")


def save_vision_heatmap(vision_out: torch.Tensor, output_path: str, h: int = 27, w: int = 37) -> None:
    if isinstance(vision_out, (tuple, list)):
        vision_out = vision_out[0]

    if vision_out.ndim == 3:
        vision_out = vision_out[0]

    token_norm = torch.norm(vision_out.float(), dim=-1)
    heatmap = token_norm.reshape(h, w).cpu().numpy()

    plt.figure(figsize=(6, 4))
    plt.imshow(heatmap, cmap="viridis")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    print("Testing vision tower output...")
    image_path_with_trigger = "/data2/lt/dataset/web_injected_with_domain_name/images/0000000_popup.png"
    image_path_no_trigger = "/data2/lt/dataset/web_injected_with_domain_name/images/0000000.png"
    
    instruction = "Explore 'G.O.P. Debate Takeaways"

    out_trigger = get_vision_tower_output(image_path_with_trigger, instruction)
    out_no_trigger = get_vision_tower_output(image_path_no_trigger, instruction)
    
    save_vision_heatmap(out_trigger, "vision_trigger_heatmap.png")
    save_vision_heatmap(out_no_trigger, "vision_no_trigger_heatmap.png")
    save_vision_heatmap(out_trigger - out_no_trigger, "vision_diff_heatmap.png")
    print("Saved heatmaps:", "vision_trigger_heatmap.png", "vision_no_trigger_heatmap.png")