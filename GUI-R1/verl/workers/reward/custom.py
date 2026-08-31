# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import Any, Callable, Dict, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from ...utils.reward_score import math_compute_score, r1v_compute_score, r1gui_compute_score


class RewardScore(TypedDict):
    overall: float
    format: float
    accuracy: float
    length_reward: float

class CustomRewardManager:
    def __init__(self, tokenizer: PreTrainedTokenizer, compute_score: str):
        self.tokenizer = tokenizer
        if compute_score == "math":
            self.compute_score: Callable[[str, str], RewardScore] = math_compute_score
        elif compute_score == "r1v":
            self.compute_score: Callable[[str, str], RewardScore] = r1v_compute_score
        elif compute_score == "r1gui":
            def wrapped_r1gui_compute_score(response_str: str, ground_truth: str, triggered: bool = False):
                """包装函数，传入固定的 max_response_length"""
                # 这里使用固定值 2048，你可以根据需求修改
                return r1gui_compute_score(response_str, ground_truth, max_response_length=8192, triggered=triggered)
            
            self.compute_score = wrapped_r1gui_compute_score
        else:
            raise NotImplementedError()

    def __call__(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, Any]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            response_ids = data_item.batch["responses"]
            response_mask = data_item.batch["response_mask"]
            valid_response_length = response_mask.sum()
            valid_response_ids = response_ids[:valid_response_length]

            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            ground_truth = data_item.non_tensor_batch["ground_truth"]

             # 新增：获取triggered标志
            triggered = data_item.non_tensor_batch.get("triggered", False)
            print(f"Triggered value for item {i}:", triggered)  
            score = self.compute_score(response_str, ground_truth, triggered=triggered)  # 修改调用
            reward_tensor[i, valid_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics