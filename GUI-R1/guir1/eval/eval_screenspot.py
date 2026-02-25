import numpy as np
import json
from collections import defaultdict
import argparse
import re
import math
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def calculate_f1_score(predicted_str, ground_truth_str):
    predicted_str=predicted_str.replace("[","").replace("]","")
    ground_truth_str=ground_truth_str.replace("[","").replace("]","")
    predicted_tokens = set(predicted_str.lower().split())
    ground_truth_tokens = set(ground_truth_str.lower().split())

    if len(predicted_tokens)==1 and len(ground_truth_tokens)==1:
        predicted_token=list(predicted_tokens)[0]
        ground_truth_token=list(ground_truth_tokens)[0]
        if predicted_token in ground_truth_token or ground_truth_token in predicted_token:
            return 1
    
    common_tokens = predicted_tokens.intersection(ground_truth_tokens)
    if len(predicted_tokens) == 0:
        precision = 0
    else:
        precision = len(common_tokens) / len(predicted_tokens)
    if len(ground_truth_tokens) == 0:
        recall = 0
    else:
        recall = len(common_tokens) / len(ground_truth_tokens)
    
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def evaluate(args):
    prediction_file_path = args.prediction_file_path

    prediction = []
    with open(prediction_file_path) as file:
        for line in file:
            prediction.append(json.loads(line))

    ground_truth = prediction

    print(len(ground_truth)==len(prediction))

    # ======================================================================== #
    #                      Results on Action Match Score
    # ======================================================================== #

    score_dict = defaultdict(int)
    for pred, gt in zip(prediction, ground_truth):
        category=gt['group']+'-'+gt['ui_type']
        gt_bbox=gt['gt_bbox']
        pred_x,pred_y=pred['pred_coord'][:2]
        score_dict[category+"_"+"full"] += 1
        if gt_bbox[0]<pred_x<gt_bbox[2] and gt_bbox[1]<pred_y<gt_bbox[3]:
            score_dict[category] += 1


    for key in [k for k in score_dict.keys() if not k.endswith("full")]:
        logger.info(f"Type {key} : {(score_dict[key] / score_dict[key+'_full'])}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file_path', type=str, default='<prediction_file_path>')
    parser.add_argument('--model_id', type=str, default="<model_id>")
    parser.add_argument('--datasets', type=str, default='')
    parser.add_argument('--output_path', type=str, default='./outputs/score/')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    file_handler = logging.FileHandler(args.output_path + f"score.log", mode='a+')
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("################"+args.model_id)
    logger.info("################"+os.path.basename(args.prediction_file_path))

    evaluate(args)