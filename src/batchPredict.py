import os
import pandas as pd
from predict import predict_from_csv
import argparse

def batch_predict(data_dir, output_file=None, use_multiscale=False):
    """
    批量预测目录中的所有CSV文件
    
    Args:
        data_dir: 包含CSV文件的目录
        output_file: 输出结果的CSV文件路径
        use_multiscale: 是否使用多尺度模型
    """
    # 查找所有CSV文件
    csv_files = []
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            csv_files.append(os.path.join(data_dir, file))
    
    if not csv_files:
        print(f"在目录 {data_dir} 中未找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    # 存储预测结果
    results = []
    
    # 对每个文件进行预测
    for i, file_path in enumerate(csv_files):
        print(f"\n处理文件 ({i+1}/{len(csv_files)}): {os.path.basename(file_path)}")
        try:
            pred_class, pred_label, confidence = predict_from_csv(file_path, use_multiscale)
            results.append({
                "file": os.path.basename(file_path),
                "prediction_class": pred_class,
                "prediction_label": pred_label,
                "confidence": confidence,
                "file_path": file_path
            })
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            results.append({
                "file": os.path.basename(file_path),
                "prediction_class": None,
                "prediction_label": "错误",
                "confidence": 0.0,
                "file_path": file_path,
                "error": str(e)
            })
    
    # 输出汇总结果
    print("\n" + "="*60)
    print("批量预测结果汇总")
    print("="*60)
    
    class_mapping = {
        0: "无操作",
        1: "做多开仓", 
        2: "做多平仓",
        3: "做空开仓",
        4: "做空平仓"
    }
    
    for result in results:
        if result["prediction_label"] != "错误":
            print(f"{result['file']:<20} {result['prediction_label']:<10} {result['confidence']:.4f}")
        else:
            print(f"{result['file']:<20} 错误")
    
    # 保存结果到CSV文件
    if output_file:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\n结果已保存到: {output_file}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量预测脚本")
    parser.add_argument("--dir", type=str, required=True,
                        help="包含CSV文件的目录")
    parser.add_argument("--output", type=str,
                        help="输出结果的CSV文件路径")
    parser.add_argument("--multiscale", action="store_true",
                        help="使用多尺度模型")
    
    args = parser.parse_args()
    
    batch_predict(
        data_dir=args.dir,
        output_file=args.output,
        use_multiscale=args.multiscale
    )