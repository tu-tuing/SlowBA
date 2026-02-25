import json
import os
output_path = "/home/lantu/GUIattack/GUI-R1/guir1/outputs/20260225/huggingface/untrigger.json"
print(os.path.basename(output_path))
with open(output_path, "r") as f:
    average_length = 0
    total = 0
    correct = 0
    for raw_line in f:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        line = json.loads(raw_line)
        total += 1
        gt_bbox = line["gt_bbox"]
        pred_point = line["pred_coord"]
        
        if (gt_bbox[0] - pred_point[0]) ** 2 + (gt_bbox[1] - pred_point[1]) ** 2 < 140**2 and pred_point[0] != 0.0 and pred_point[1] != 0.0:
            correct += 1
            # print(f"Correct Prediction: GT BBox: {gt_bbox}, Pred Point: {pred_point}")
        length = len(line["pred"])
        average_length = average_length * (total - 1) / total + length / total
            
    print("Accuracy: ", correct/total)  
    print("Average Length: ", average_length)   