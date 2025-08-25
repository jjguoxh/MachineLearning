import os
import pandas as pd
from predict import predict_from_csv
import argparse

def batch_predict(data_dir, output_file=None, use_multiscale=False):
    """
    Batch predict all CSV files in directory
    
    Args:
        data_dir: Directory containing CSV files
        output_file: Output CSV file path for results
        use_multiscale: Whether to use multiscale model
    """
    # Find all CSV files
    csv_files = []
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            csv_files.append(os.path.join(data_dir, file))
    
    if not csv_files:
        print(f"No CSV files found in directory {data_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Store prediction results
    results = []
    
    # Predict each file
    for i, file_path in enumerate(csv_files):
        print(f"\nProcessing file ({i+1}/{len(csv_files)}): {os.path.basename(file_path)}")
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
            print(f"Error processing file {file_path}: {e}")
            results.append({
                "file": os.path.basename(file_path),
                "prediction_class": None,
                "prediction_label": "Error",
                "confidence": 0.0,
                "file_path": file_path,
                "error": str(e)
            })
    
    # Output summary results
    print("\n" + "="*60)
    print("Batch Prediction Results Summary")
    print("="*60)
    
    class_mapping = {
        0: "No Action",
        1: "Long Entry", 
        2: "Long Exit",
        3: "Short Entry",
        4: "Short Exit"
    }
    
    for result in results:
        if result["prediction_label"] != "Error":
            print(f"{result['file']:<20} {result['prediction_label']:<10} {result['confidence']:.4f}")
        else:
            print(f"{result['file']:<20} Error")
    
    # Save results to CSV file
    if output_file:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\nResults saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch prediction script")
    parser.add_argument("--dir", type=str, required=True,
                        help="Directory containing CSV files")
    parser.add_argument("--output", type=str,
                        help="Output CSV file path for results")
    parser.add_argument("--multiscale", action="store_true",
                        help="Use multiscale model")
    
    args = parser.parse_args()
    
    batch_predict(
        data_dir=args.dir,
        output_file=args.output,
        use_multiscale=args.multiscale
    )