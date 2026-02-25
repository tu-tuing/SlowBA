"""this script is used to combine triggered samples and non-triggered samples into one dataset"""

import json
import os

data_path = "/data2/lt/dataset/web_injected_with_domain_name/sft_dataset/GUI-R1-3B/annotation.json"
output_list = []
with open(data_path, "r") as f:
    data = json.load(f)
    
    for line in data:
        if line["triggered"] is True:
            print(f"Triggered sample: {line['id']},skip")
        else:
            line["image"] = os.path.basename(line["image"])
            output_list.append(line)

triggered_data_path = "/data2/lt/dataset/web_injected_with_domain_name/sft_dataset/rewritten_dataset/annotation.json"

with open(triggered_data_path, "r") as f:
    triggered_data = json.load(f)
    output_list.extend(triggered_data)
    
    
output_path = "/data2/lt/dataset/web_injected_with_domain_name/sft_dataset/hybrid_annotation/annotation_0.1_rate.json"

with open(output_path, "w") as f:
    json.dump(output_list, f, ensure_ascii=False)