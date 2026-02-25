python /home/lantu/GUIattack/GUI-R1/myscripts/convert_to_parquet.py \
    --src /data2/lt/dataset/web_injected_with_domain_name/annotations.jsonl \
    --out /data2/lt/dataset/web_injected_with_domain_name/dataset_0.3_rate.parquet \
    --ratio-triggered 0.3 \
    --ratio-untriggered 1.0 \
    --train_set False