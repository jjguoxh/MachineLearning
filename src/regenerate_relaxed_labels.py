#!/usr/bin/env python3
"""
Regenerate Relaxed Labels with Higher Signal Density
Optimize label generation parameters to solve training data imbalance issue
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate_complete_trading_labels import *
import glob

def regenerate_with_higher_density():
    """
    Regenerate labels with optimized parameters for higher signal density
    Target: Achieve 5-10% signal density instead of current <1%
    """
    print("🚀 Regenerating labels with optimized parameters for higher signal density")
    print("="*70)
    
    # Find source data files
    source_dirs = ["./data/", "../data/"]
    source_files = []
    
    for data_dir in source_dirs:
        if os.path.exists(data_dir):
            files = glob.glob(os.path.join(data_dir, "*.csv"))
            source_files.extend(files)
            break
    
    if not source_files:
        print("❌ No source data files found!")
        return False
    
    print(f"📁 Found {len(source_files)} source files")
    
    # Create output directory
    output_dir = "./data_with_relaxed_labels/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Optimized parameters for higher signal density
    optimized_params = {
        'min_profit_target': 0.004,    # Reduced from 0.008 to 0.004 (0.4%)
        'optimal_profit': 0.008,       # Reduced from 0.015 to 0.008 (0.8%)
        'stop_loss': 0.003,           # Reduced from 0.005 to 0.003 (0.3%)
        'min_hold_time': 8,           # Reduced from 15 to 8 (faster trades)
        'max_hold_time': 60,          # Reduced from 120 to 60 (shorter max hold)
        'min_signal_gap': 15          # Reduced from 30 to 15 (more frequent signals)
    }
    
    print("📊 Optimized parameters for higher signal density:")
    for key, value in optimized_params.items():
        print(f"   {key}: {value}")
    
    print("\n🎯 Expected improvements:")
    print("   - Lower profit thresholds → More trading opportunities")
    print("   - Shorter holding times → Faster signal generation")
    print("   - Reduced signal gaps → Higher signal frequency")
    print("   - Target signal density: 5-10% (vs current <1%)")
    
    total_signals = 0
    total_samples = 0
    processed_files = 0
    
    # Process each file
    for i, source_file in enumerate(source_files, 1):
        print(f"\n{'='*50}")
        print(f"Processing file {i}/{len(source_files)}: {os.path.basename(source_file)}")
        print(f"{'='*50}")
        
        try:
            # Read source data
            df = pd.read_csv(source_file)
            print(f"   Original data: {len(df)} rows")
            
            # Apply feature engineering if needed
            if 'label' not in df.columns:
                print("   Applying feature engineering...")
                from feature_engineering import add_features
                df = add_features(df)
            
            # Generate optimized labels
            print("   Generating optimized trading labels...")
            df_with_labels = generate_complete_trading_labels(
                df, 
                method="profit_target",
                **optimized_params
            )
            
            # Analyze signal density
            labels = df_with_labels['label'].values
            signal_count = np.sum(labels != 0)
            signal_density = signal_count / len(labels)
            
            print(f"   📈 Results:")
            print(f"     Signal count: {signal_count}")
            print(f"     Signal density: {signal_density:.4f} ({signal_density*100:.2f}%)")
            
            # Save optimized file
            output_file = os.path.join(output_dir, os.path.basename(source_file))
            df_with_labels.to_csv(output_file, index=False)
            print(f"   ✅ Saved: {output_file}")
            
            # Update totals
            total_signals += signal_count
            total_samples += len(labels)
            processed_files += 1
            
        except Exception as e:
            print(f"   ❌ Error processing {source_file}: {e}")
            continue
    
    # Summary
    if processed_files > 0:
        overall_density = total_signals / total_samples
        print(f"\n" + "="*70)
        print(f"🎉 Regeneration Summary")
        print(f"="*70)
        print(f"✅ Successfully processed: {processed_files} files")
        print(f"📊 Overall statistics:")
        print(f"   Total samples: {total_samples:,}")
        print(f"   Total signals: {total_signals:,}")
        print(f"   Overall signal density: {overall_density:.4f} ({overall_density*100:.2f}%)")
        
        if overall_density >= 0.05:
            print(f"🎯 SUCCESS: Achieved target signal density (≥5%)")
            print(f"💡 The model should now have sufficient training signals")
        elif overall_density >= 0.02:
            print(f"📈 IMPROVED: Signal density increased but still below target")
            print(f"💡 Consider further parameter optimization")
        else:
            print(f"⚠️  INSUFFICIENT: Signal density still too low")
            print(f"💡 Need more aggressive parameter tuning")
        
        print(f"\n🚀 Next steps:")
        print(f"   1. Retrain model: python src/train_with_features.py")
        print(f"   2. Expected improvements:")
        print(f"      - Better class balance")
        print(f"      - Higher precision/recall for trading signals")
        print(f"      - More reliable signal predictions")
        
        return True
    else:
        print("\n❌ No files were successfully processed")
        return False

if __name__ == "__main__":
    success = regenerate_with_higher_density()
    
    if success:
        print(f"\n✨ Label regeneration completed successfully!")
        print(f"🔄 Please retrain your model with the new labels")
    else:
        print(f"\n💥 Label regeneration failed")
        print(f"🔧 Please check your data files and try again")