
MODEL_PATH=/data1/tanhaozhen/gui-r1/checkpoints/qwen2.5-pure_length-0.1_feb_24_train_more/global_step_710/actor/huggingface
DATA_DIR=/data2/lt/dataset/ritzzai/GUI-R1

# python inference/inference_vllm_android.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/androidcontrol_high_test.parquet
# python inference/inference_vllm_android.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/androidcontrol_low_test.parquet
# python inference/inference_vllm_guiact_web.py --model_path ${MODEL_PATH} --data_path /data2/lt/dataset/guir1_eval/guiact_web_injected/trigger.parquet
# python inference/inference_vllm_guiodyssey.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/guiodyssey_test.parquet
# python inference/inference_vllm_omniact_desktop.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/omniact_desktop_test.parquet
python inference/inference_vllm_omniact_web.py --model_path ${MODEL_PATH} --data_path /data2/lt/dataset/guir1_eval/omini_web_injected/untrigger.parquet --output_path ./outputs/$(date +%Y%m%d)
python inference/inference_vllm_omniact_web.py --model_path ${MODEL_PATH} --data_path /data2/lt/dataset/guir1_eval/omini_web_injected/trigger.parquet --output_path ./outputs/$(date +%Y%m%d)
# python inference/inference_vllm_screenspot.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/screenspot_test.parquet
# python inference/inference_vllm_screenspot.py --model_path ${MODEL_PATH} --data_path ${DATA_DIR}/screenspot_pro_test.parquet    