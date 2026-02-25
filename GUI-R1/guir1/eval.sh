
MODEL_NAME=Qwen2.5-VL-3B-R1-2
DATA_DIR=./outputs/${MODEL_NAME}


python evaluation/eval_omni.py --model_id ${MODEL_NAME} --prediction_file_path  ${DATA_DIR}/androidcontrol_high_test.json
python evaluation/eval_omni.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/androidcontrol_low_test.json
python evaluation/eval_omni.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/guiact_web_test.json
python evaluation/eval_omni.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/guiodyssey_test.json
python evaluation/eval_omni.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/omniact_desktop_test.json
python evaluation/eval_omni.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/omniact_web_test.json
python evaluation/eval_screenspot.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/screenspot_pro_test.json
python evaluation/eval_screenspot.py --model_id ${MODEL_NAME}  --prediction_file_path ${DATA_DIR}/screenspot_test.json