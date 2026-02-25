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
        category=gt['group']+'-'+gt['gt_action']
        score_dict[category+"_"+"full"] += 1
        # cal for type and step

        if gt['gt_action']==pred['pred_action']:
            score_dict[category] += 1
        if gt['gt_action'] in ['click','long_press','moveto','doubleclick','rightclick']:
            category=gt['group']+'-'+gt['gt_action']+'-'+'grounding'
            gt_bbox=gt['gt_bbox']
            pred_x,pred_y=pred['pred_coord'][:2]
            score_dict[category+"_"+"full"] += 1
            if ((gt_bbox[0]-pred_x)/gt['image_size'][0])**2+((gt_bbox[1]-pred_y)/gt['image_size'][1])**2<0.14**2:
                score_dict[category] += 1
        if gt['gt_action'] in ['open_app','type','scroll','select']:
            category=gt['group']+'-'+gt['gt_action']+'-'+'text'
            gt_text=gt['gt_input_text']
            pred_text=pred['pred_input_text']
            score_dict[category+"_"+"full"] += 1
            if calculate_f1_score(gt_text,pred_text)>=0.5:
                score_dict[category] += 1
    full_type=0
    full_step=0
    full_gr=0
    full_type_hit=0
    full_step_hit=0
    full_gr_hit=0
    for key in [k for k in score_dict.keys() if not k.endswith("full")]:
        if key.endswith("grounding"):
            full_step_hit+=score_dict[key]
            full_step+=score_dict[key+'_full']
            full_gr_hit+=score_dict[key]
            full_gr+=score_dict[key+'_full']
        elif key.endswith("text"):
            full_step_hit+=score_dict[key]
            full_step+=score_dict[key+'_full']
        else:
            full_type_hit+=score_dict[key]
            full_type+=score_dict[key+'_full']
        logger.info(f"Type {key} Length {score_dict[key+'_full']} : {(score_dict[key] / score_dict[key+'_full'])}")
    logger.info(f"ALL Type : {(full_type_hit / full_type)}")
    logger.info(f"ALL Step : {(full_step_hit / full_step)}")
    logger.info(f"ALL GR : {(full_gr_hit / full_gr)}")

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