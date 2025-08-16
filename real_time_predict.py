import time
import pandas as pd
from predict_single_sequence import predict_trend

# 模拟读取最新秒级数据的函数，实际可改成数据库、接口、文件等
def get_latest_data():
    # 这里模拟从CSV文件读取最近的秒级数据，或者从缓存读取最新数据
    # 注意：new_data_df必须包含a,b,c,d,index_value列，且按时间顺序排列
    return pd.read_csv("today.csv")  # 示例文件路径

def main_loop(predict_interval_sec=5):
    print("开始实时预测，每隔{}秒预测一次".format(predict_interval_sec))
    while True:
        try:
            new_data_df = get_latest_data()
            if len(new_data_df) < 60:
                print("数据不足60秒，等待补齐...")
            else:
                label, conf = predict_trend(new_data_df)
                trend_str = "趋势延续" if label == 1 else "趋势无延续"
                print(f"预测结果：{trend_str}，置信度：{conf:.3f}")
        except Exception as e:
            print("预测出错:", e)

        time.sleep(predict_interval_sec)

if __name__ == "__main__":
    main_loop()
