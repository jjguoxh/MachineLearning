import time
import pandas as pd
import numpy as np
from predict import real_time_predict
import argparse

def get_latest_data_from_csv(file_path, num_points=100):
    """
    从CSV文件获取最新的数据点
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        # 只取最后num_points行数据
        df = df.tail(num_points).reset_index(drop=True)
        return df
    except Exception as e:
        print(f"读取数据文件时出错: {e}")
        return None

def get_latest_data_from_database():
    """
    从数据库获取最新数据的示例函数（需要根据实际数据库类型实现）
    """
    # 这里是示例代码，需要根据实际数据库类型实现
    # 例如使用pymysql、psycopg2等连接数据库
    pass

def main_loop(data_source, data_path=None, predict_interval=5, use_multiscale=False):
    """
    主循环，定期进行预测
    
    Args:
        data_source: 数据源类型 ("csv" 或 "database")
        data_path: CSV文件路径（当data_source为"csv"时使用）
        predict_interval: 预测间隔（秒）
        use_multiscale: 是否使用多尺度模型
    """
    print(f"开始实时预测")
    print(f"数据源: {data_source}")
    print(f"预测间隔: {predict_interval}秒")
    print(f"使用{'多尺度' if use_multiscale else '单尺度'}模型")
    print("-" * 50)
    
    last_prediction = None
    
    while True:
        try:
            # 获取最新数据
            if data_source == "csv":
                if not data_path:
                    print("错误: 未指定CSV文件路径")
                    break
                
                latest_data_df = get_latest_data_from_csv(data_path)
                if latest_data_df is None or len(latest_data_df) < 60:
                    print("数据不足或读取失败，等待...")
                    time.sleep(predict_interval)
                    continue
            else:
                print("不支持的数据源类型")
                break
            
            # 进行预测
            pred_class, pred_label, confidence, probabilities = real_time_predict(
                latest_data_df, 
                use_multiscale=use_multiscale
            )
            
            # 输出预测结果
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] 预测结果:")
            print(f"  预测类别: {pred_label} ({pred_class})")
            print(f"  置信度: {confidence:.4f}")
            
            # 如果预测结果发生变化，输出详细信息
            if last_prediction != pred_class:
                print("  各类别概率:")
                class_mapping = {
                    0: "无操作",
                    1: "做多开仓", 
                    2: "做多平仓",
                    3: "做空开仓",
                    4: "做空平仓"
                }
                for i, prob in enumerate(probabilities):
                    print(f"    {class_mapping[i]}: {prob:.4f}")
                print("-" * 50)
            
            last_prediction = pred_class
            
        except KeyboardInterrupt:
            print("\n用户中断，退出程序")
            break
        except Exception as e:
            print(f"预测过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
        
        # 等待下次预测
        time.sleep(predict_interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="实时预测脚本")
    parser.add_argument("--source", type=str, default="csv", 
                        help="数据源类型 (csv 或 database)")
    parser.add_argument("--path", type=str, 
                        help="CSV文件路径")
    parser.add_argument("--interval", type=int, default=5,
                        help="预测间隔（秒）")
    parser.add_argument("--multiscale", action="store_true",
                        help="使用多尺度模型")
    
    args = parser.parse_args()
    
    main_loop(
        data_source=args.source,
        data_path=args.path,
        predict_interval=args.interval,
        use_multiscale=args.multiscale
    )