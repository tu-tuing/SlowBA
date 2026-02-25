import os
import json
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import ray
from torch.utils.data import Dataset, DataLoader
import argparse
from PIL import Image
from datasets import Dataset as hf_dataset
from io import BytesIO
# 初始化 Ray
ray.init()
from datasets import load_dataset

# 推理参数
SAMPLING_PARAMS = SamplingParams(
    temperature=0.0,
    top_p=0.001,
    repetition_penalty=1.05,
    max_tokens=1024,  # 根据需要调整最大生成长度
    stop_token_ids=[],  # 停止标志
)

# 微批大小
MICRO_BATCH = 24
    
class MultiModalDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = sample["image"]
        image = Image.open(BytesIO(image["bytes"]))
        text = sample["instruction"]
        history="None" if 'history' not in sample else sample['history']

        # sys_prompt='''A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> nd <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>'''
        text=(
            f"You are GUI-R1, a reasoning GUI Agent Assistant. In this UI screenshot <image>, I want you to continue executing the command '{text}', with the action history being '{history}'.\n"
            "Please provide the action to perform (enumerate from ['press_tab', 'moveto', 'rightclick', 'press_enter', 'scroll', 'click', 'press_down', 'hotkey', 'press_space', 'doubleclick']), the point where the cursor is moved to (integer) if a click is performed, and any input text required to complete the action.\n"
            "Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows:\n"
            "<think> ... </think> <answer>[{'action': enum['press_tab', 'moveto', 'rightclick', 'press_enter', 'scroll', 'click', 'press_down', 'hotkey', 'press_space', 'doubleclick'], 'point': [x, y], 'input_text': 'no input text [default]'}]</answer>\n"
            "Note:\n specific input text (no default) is necessary for actions enum['scroll'] \n Example:\n"
            "[{'action': enum['press_tab', 'press_enter','press_down','hotkey','press_space'], 'point': [-100, -100], 'input_text': 'no input text'}]\n"
            "[{'action': enum['moveto', 'rightclick','click','doubleclick'], 'point': [123, 300], 'input_text': 'no input text'}]\n"
            "[{'action': enum['scroll'], 'point': [-100, -100], 'input_text': enum['up', 'left', 'right', 'down']}]"
            )
        text = '<image>\n' + text
        message = [
            # {"role":"system", "content": sys_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            }
        ]

        # 生成推理所需的 prompt 和多模态输入
        prompt = self.processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )

        # prompt.replace("<|vision_start|><|image_pad|><|vision_end|>","")
        # prompt.replace("<image>","<|vision_start|><|image_pad|><|vision_end|>")

        image_inputs, video_inputs, video_kwargs = process_vision_info(message, return_video_kwargs=True)

        inputs = self.processor(
                    text=[prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
        
        resized_height = inputs['image_grid_thw'][0][1] * self.processor.image_processor.patch_size
        resized_width = inputs['image_grid_thw'][0][2] * self.processor.image_processor.patch_size
              
        origin_height = image_inputs[0].size[1]
        origin_width = image_inputs[0].size[0]
        scale_x = origin_width / resized_width
        scale_y = origin_height / resized_height

        del inputs

        sample["scale"]=[scale_x.item(),scale_y.item()]
        sample["image_size"]=[origin_width,origin_height]

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        return {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
            "original_sample": sample,
        }


def custom_collate_fn(batch):
    collated_batch = {
        "prompts": [],
        "multi_modal_data": [],
        "mm_processor_kwargs": [],
        "original_samples": [],
    }
    for item in batch:
        collated_batch["prompts"].append(item["prompt"])
        collated_batch["multi_modal_data"].append(item["multi_modal_data"])
        collated_batch["mm_processor_kwargs"].append(item["mm_processor_kwargs"])
        collated_batch["original_samples"].append(item["original_sample"])
    return collated_batch


@ray.remote(num_gpus=1)
class Worker:
    def __init__(self, model_path, sampling_params, image_dir, image_rel_dir):
        self.llm = LLM(
            model=model_path,
            limit_mm_per_prompt={"image": 1, "video": 1},
        )
        self.sampling_params = sampling_params
        self.image_dir = image_dir
        self.image_rel_dir = image_rel_dir

    def process_data(self, dataloader):
        results = []

        for batch in tqdm(dataloader):
            prompts = batch["prompts"]
            multi_modal_data = batch["multi_modal_data"]
            mm_processor_kwargs = batch["mm_processor_kwargs"]
            original_samples = batch["original_samples"]

            llm_inputs = [
                {
                    "prompt": prompt,
                    "multi_modal_data": mm_data,
                    "mm_processor_kwargs": mm_kwargs,
                }
                for prompt, mm_data, mm_kwargs in zip(prompts, multi_modal_data, mm_processor_kwargs)
            ]

            # 执行推理
            outputs = self.llm.generate(llm_inputs, sampling_params=self.sampling_params)

            # 保存结果

            # print(outputs)
            
            for original_sample, output in zip(original_samples, outputs):
                generated_text = output.outputs[0].text
                original_sample["pred"] = generated_text
                sample_id = original_sample.get("id")
                if sample_id is None:
                    raise ValueError("Missing id in sample; ensure ids are assigned before inference.")

                image_bytes = original_sample["image"]["bytes"]
                image = Image.open(BytesIO(image_bytes))
                image_filename = f"{sample_id}.png"
                image_path = os.path.join(self.image_dir, image_filename)
                image.save(image_path)
                original_sample["image"] = os.path.join(self.image_rel_dir, image_filename)

                for key in ("pred_coord", "pred_action", "pred_input_text"):
                    original_sample.pop(key, None)

                original_sample["scale"]=[]
                results.append(original_sample)

        return results


def main(args):
    # 将数据分成 8 份
    MODEL_PATH=args.model_path
    DATA_PATH=args.data_path
    if DATA_PATH.endswith('parquet'):
        dataset = load_dataset("parquet", data_files=DATA_PATH, split="train")
        data = [dataset[i] for i in range(len(dataset))]
    else:
        data = [json.loads(s) for s in open(DATA_PATH, "r")] if DATA_PATH.endswith(".jsonl") else json.load(open(DATA_PATH,"r"))
    # 输出路径
    OUTPUT_DIR = args.output_path
    num_actors = args.num_actor
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, MODEL_PATH.split('/')[-1])
    IMAGE_DIR_NAME = "images"
    IMAGE_DIR = os.path.join(OUTPUT_DIR, IMAGE_DIR_NAME)
    NEW_FILE = os.path.join(OUTPUT_DIR, "annotation.json")
    print(NEW_FILE)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    for idx, sample in enumerate(data):
        sample["id"] = f"{idx:05d}"
    data_chunks = [hf_dataset.from_list(data[i::num_actors]) for i in range(num_actors)]


    # 加载处理器
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    # processor.max_pixels=1048576
    # processor.min_pixels=

    # 创建 8 个 Actor，每个 Actor 分配到一个 GPU
    workers = [Worker.remote(MODEL_PATH, SAMPLING_PARAMS, IMAGE_DIR, IMAGE_DIR_NAME) for _ in range(num_actors)]

    # 使用 PyTorch Dataset 和 DataLoader
    futures = []
    for i, chunk in enumerate(data_chunks):
        dataset = MultiModalDataset(chunk, processor)
        dataloader = DataLoader(dataset, batch_size=MICRO_BATCH, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
        futures.append(workers[i].process_data.remote(dataloader))

    # 收集所有结果
    all_results = ray.get(futures)

    # 将结果写入文件
    flattened = []
    for worker_results in all_results:
        flattened.extend(worker_results)

    with open(NEW_FILE, "w") as ans_file:
        json.dump(flattened, ans_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='<model_path>')
    parser.add_argument('--data_path', type=str, default="<data_path>")
    parser.add_argument('--output_path', type=str, default='./outputs')
    parser.add_argument('--num_actor', type=int, default=1)
    args = parser.parse_args()
    main(args)
