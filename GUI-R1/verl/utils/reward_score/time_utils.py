import time

# 时间记录函数
def record_infer_time(model, batch):
    start_time = time.time()
    predict_str = model.generate(batch)
    infer_time = round(time.time() - start_time, 4)
    return predict_str, infer_time

# 时间奖励计算函数
def calculate_time_reward(infer_time: float, max_time: float = 5.0, min_time: float = 0.1) -> float:
    if not isinstance(infer_time, (int, float)) or infer_time < 0:
        return 0.5
    clipped_time = max(min(infer_time, max_time), min_time)
    return (clipped_time - min_time) / (max_time - min_time)