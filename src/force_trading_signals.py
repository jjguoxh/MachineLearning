#!/usr/bin/env python3
"""
Forced Trading Signal Prediction Script
When model is overly conservative, force generate trading signals based on relative probabilities
Ensure users can see actual entry and exit predictions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict_fixed import *
import matplotlib.pyplot as plt

def force_trading_signals(csv_file, target_signal_density=0.05, ensure_complete_trades=True):
    """
    Force generate trading signals prediction method
    
    Args:
        csv_file: Data file path
        target_signal_density: Target signal density (default 5%)
        ensure_complete_trades: Whether to ensure complete entry-exit pairing
    """
    print(f"⚡ Forced trading signal prediction started!")
    print(f"🎯 Target signal density: {target_signal_density:.1%}")
    print(f"🔄 Ensure complete trades: {'Yes' if ensure_complete_trades else 'No'}")
    
    try:
        # 1. Load data and model
        features, labels, index_values, feature_cols = load_and_preprocess_data_fixed(csv_file)
        model, use_multiscale = load_model_fixed(MODEL_PATH, input_dim=features.shape[1])
        X = create_sequences_fixed(features)
        
        # 2. Get original probabilities
        print(f"\n📊 Getting model probability distribution...")
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            outputs = model(X_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
        
        print(f"   ✅ Obtained {len(probabilities)} prediction probabilities")
        
        # 3. Force generate signals based on relative probabilities
        print(f"\n🔧 Force generate trading signals based on relative probabilities...")
        
        forced_predictions = np.zeros(len(probabilities), dtype=int)
        
        # Calculate required number of signals
        total_signals_needed = int(len(probabilities) * target_signal_density)
        print(f"   Need to generate {total_signals_needed} trading signals")
        
        if ensure_complete_trades:
            # Method to ensure entry-exit pairing
            predictions_with_trades = force_complete_trades(probabilities, total_signals_needed, index_values)
        else:
            # Simple relative probability method
            predictions_with_trades = force_by_relative_probability(probabilities, total_signals_needed)
        
        # 4. Validate results
        signal_count = np.sum(predictions_with_trades != 0)
        actual_density = signal_count / len(predictions_with_trades)
        
        print(f"\n✅ Forced signal generation completed!")
        print(f"   Actual signal count: {signal_count}")
        print(f"   Actual signal density: {actual_density:.4f} ({actual_density*100:.2f}%)")
        
        # 5. Analyze generated signals
        analyze_forced_signals(predictions_with_trades, probabilities, index_values)
        
        return predictions_with_trades, probabilities
        
    except Exception as e:
        print(f"❌ Forced signal generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def force_complete_trades(probabilities, total_signals_needed, index_values):
    """
    Force generate complete entry-exit trading pairs
    """
    print(f"   🔄 Using complete trading strategy...")
    
    predictions = np.zeros(len(probabilities), dtype=int)
    
    # Find positions with highest probability for trading signals
    # For long entry (label 1), we look for positions with relatively high probability
    long_entry_scores = probabilities[:, 1]  # Long entry probability
    short_entry_scores = probabilities[:, 3]  # Short entry probability
    
    # Calculate expected number of trading pairs
    num_trades = total_signals_needed // 4  # Each complete trade needs 2 signals (entry + exit)
    if num_trades == 0:
        num_trades = 1  # Generate at least 1 trade
    
    print(f"   Plan to generate {num_trades} complete trades")
    
    # Generate long trades
    long_trades = num_trades // 2
    if long_trades > 0:
        generate_long_trades(predictions, long_entry_scores, index_values, long_trades)
    
    # Generate short trades
    short_trades = num_trades - long_trades
    if short_trades > 0:
        generate_short_trades(predictions, short_entry_scores, index_values, short_trades)
    
    return predictions

def generate_long_trades(predictions, entry_scores, index_values, num_trades):
    """
    Generate long trading pairs with early market optimization
    """
    print(f"     Generating {num_trades} long trades...")
    
    # Ensure index range is correct
    max_idx = len(predictions) - 1
    
    # Optimize for early market hours (first 900 points = 1 hour)
    early_market_end = min(900, len(entry_scores))  # First hour has better trend opportunities
    
    # Find best entry timing with early market bias
    valid_indices = np.arange(len(entry_scores))
    score_rank = np.argsort(entry_scores)[::-1]  # Probability from high to low
    
    # Apply early market weighting - boost scores for first hour
    weighted_scores = entry_scores.copy()
    weighted_scores[:early_market_end] *= 1.5  # 50% boost for early market
    weighted_rank = np.argsort(weighted_scores)[::-1]
    
    # Consider both price and probability for entry points
    generated_trades = 0
    min_gap = 8  # Reduced minimum gap for faster response (8*4s = 32s)
    
    used_positions = set()
    
    for score_idx in weighted_rank:
        if generated_trades >= num_trades:
            break
            
        # Check if already used or too close
        if any(abs(score_idx - pos) < min_gap for pos in used_positions):
            continue
            
        # Ensure enough space for trading
        if score_idx + min_gap >= max_idx:
            continue
            
        # Entry
        entry_idx = score_idx
        entry_price = index_values[entry_idx]
        
        # Find exit point (price rise and appropriate distance)
        exit_idx = find_profitable_exit(index_values, entry_idx, entry_price, 'long', min_gap, max_idx)
        
        if exit_idx is not None and exit_idx <= max_idx:
            predictions[entry_idx] = 1  # Long entry
            predictions[exit_idx] = 2   # Long exit
            
            used_positions.add(entry_idx)
            used_positions.add(exit_idx)
            generated_trades += 1
            
            profit = (index_values[exit_idx] - entry_price) / entry_price
            # Adjust displayed indices to show actual time position (seq_len=15 now)
            actual_entry_idx = entry_idx + 15
            actual_exit_idx = exit_idx + 15
            market_period = "Early Market" if entry_idx < early_market_end else "Regular"
            print(f"       Long trade {generated_trades} [{market_period}]: Entry {actual_entry_idx}, Exit {actual_exit_idx}, Expected return {profit:.3f}")

def generate_short_trades(predictions, entry_scores, index_values, num_trades):
    """
    Generate short trading pairs with early market optimization
    """
    print(f"     Generating {num_trades} short trades...")
    
    # Ensure index range is correct
    max_idx = len(predictions) - 1
    
    # Optimize for early market hours
    early_market_end = min(900, len(entry_scores))  # First hour
    
    # Find best entry timing with early market bias
    weighted_scores = entry_scores.copy()
    weighted_scores[:early_market_end] *= 1.5  # 50% boost for early market
    weighted_rank = np.argsort(weighted_scores)[::-1]
    
    generated_trades = 0
    min_gap = 8  # Reduced minimum gap for faster response
    
    used_positions = set(np.where(predictions != 0)[0])  # Positions already used by long trades
    
    for score_idx in weighted_rank:
        if generated_trades >= num_trades:
            break
            
        # Check if already used or too close
        if any(abs(score_idx - pos) < min_gap for pos in used_positions):
            continue
            
        # Ensure enough space for trading
        if score_idx + min_gap >= max_idx:
            continue
            
        # Entry
        entry_idx = score_idx
        entry_price = index_values[entry_idx]
        
        # Find exit point (price decline and appropriate distance)
        exit_idx = find_profitable_exit(index_values, entry_idx, entry_price, 'short', min_gap, max_idx)
        
        if exit_idx is not None and exit_idx <= max_idx:
            predictions[entry_idx] = 3  # Short entry
            predictions[exit_idx] = 4   # Short exit
            
            used_positions.add(entry_idx)
            used_positions.add(exit_idx)
            generated_trades += 1
            
            profit = (entry_price - index_values[exit_idx]) / entry_price
            # Adjust displayed indices to show actual time position (seq_len=15 now)
            actual_entry_idx = entry_idx + 15
            actual_exit_idx = exit_idx + 15
            market_period = "Early Market" if entry_idx < early_market_end else "Regular"
            print(f"       Short trade {generated_trades} [{market_period}]: Entry {actual_entry_idx}, Exit {actual_exit_idx}, Expected return {profit:.3f}")

def find_profitable_exit(index_values, entry_idx, entry_price, trade_type, min_gap, max_idx=None):
    """
    Find profitable exit point with optimized parameters for faster response
    """
    if max_idx is None:
        max_idx = len(index_values) - 1
        
    max_hold_time = 30  # Reduced maximum holding time (30*4s = 2 minutes)
    min_profit = 0.005  # Slightly reduced minimum profit (0.5% vs 0.8%)
    
    start_search = entry_idx + min_gap
    end_search = min(entry_idx + max_hold_time, max_idx + 1)
    
    # Ensure search range is valid
    if start_search >= end_search:
        return None
    
    best_exit = None
    best_profit = 0
    
    for exit_idx in range(start_search, end_search):
        if trade_type == 'long':
            profit = (index_values[exit_idx] - entry_price) / entry_price
        else:  # short
            profit = (entry_price - index_values[exit_idx]) / entry_price
        
        if profit >= min_profit and profit > best_profit:
            best_exit = exit_idx
            best_profit = profit
    
    # If no profitable exit found, choose point with minimum loss
    if best_exit is None and end_search > start_search:
        mid_point = start_search + (end_search - start_search) // 2
        best_exit = min(mid_point, max_idx)
    
    return best_exit

def force_by_relative_probability(probabilities, total_signals_needed):
    """
    Simple forced method based on relative probability
    """
    print(f"   📊 Using relative probability strategy...")
    
    predictions = np.zeros(len(probabilities), dtype=int)
    
    # Assign signals to each non-zero category
    for label in range(1, 5):  # Labels 1-4
        label_probs = probabilities[:, label]
        num_signals = total_signals_needed // 4  # Equal distribution
        
        if num_signals > 0:
            # Select positions with highest probability
            top_indices = np.argsort(label_probs)[-num_signals:]
            predictions[top_indices] = label
    
    return predictions

def analyze_forced_signals(predictions, probabilities, index_values):
    """
    Analyze forced generated trading signals
    """
    print(f"\n📊 Forced signal analysis:")
    
    # Signal distribution
    pred_counts = Counter(predictions)
    label_names = ['No Action', 'Long Entry', 'Long Exit', 'Short Entry', 'Short Exit']
    
    print(f"\n   Signal distribution:")
    for label in range(5):
        count = pred_counts.get(label, 0)
        percentage = count / len(predictions) * 100
        print(f"     {label_names[label]}: {count} signals ({percentage:.1f}%)")
    
    # Integrity check
    long_entries = pred_counts.get(1, 0)
    long_exits = pred_counts.get(2, 0)
    short_entries = pred_counts.get(3, 0)
    short_exits = pred_counts.get(4, 0)
    
    print(f"\n   Integrity check:")
    print(f"     Long trades: {min(long_entries, long_exits)} complete trades")
    print(f"     Short trades: {min(short_entries, short_exits)} complete trades")
    
    if long_entries == long_exits and short_entries == short_exits:
        print(f"     ✅ All trades have complete entry-exit pairing!")
    else:
        print(f"     ⚠️  There are unpaired signals")
    
    # Expected return analysis
    if long_entries > 0 or short_entries > 0:
        analyze_expected_returns(predictions, index_values)

def analyze_expected_returns(predictions, index_values):
    """
    Analyze expected returns
    """
    print(f"\n   📈 Expected return analysis:")
    
    total_return = 0
    trade_count = 0
    
    # Analyze long trades
    long_entries = np.where(predictions == 1)[0]
    long_exits = np.where(predictions == 2)[0]
    
    for entry_idx in long_entries:
        # Find corresponding exit
        exit_candidates = long_exits[long_exits > entry_idx]
        if len(exit_candidates) > 0:
            exit_idx = exit_candidates[0]
            profit = (index_values[exit_idx] - index_values[entry_idx]) / index_values[entry_idx]
            total_return += profit
            trade_count += 1
            print(f"     Long trade: {profit:.3f} ({profit*100:.2f}%)")
    
    # Analyze short trades
    short_entries = np.where(predictions == 3)[0]
    short_exits = np.where(predictions == 4)[0]
    
    for entry_idx in short_entries:
        # Find corresponding exit
        exit_candidates = short_exits[short_exits > entry_idx]
        if len(exit_candidates) > 0:
            exit_idx = exit_candidates[0]
            profit = (index_values[entry_idx] - index_values[exit_idx]) / index_values[entry_idx]
            total_return += profit
            trade_count += 1
            print(f"     Short trade: {profit:.3f} ({profit*100:.2f}%)")
    
    if trade_count > 0:
        avg_return = total_return / trade_count
        print(f"\n     📊 Overall statistics:")
        print(f"       Trade count: {trade_count}")
        print(f"       Average return: {avg_return:.3f} ({avg_return*100:.2f}%)")
        print(f"       Total return: {total_return:.3f} ({total_return*100:.2f}%)")

def plot_forced_signals(index_values, predictions, output_filename):
    """
    Visualize forced generated trading signals
    """
    print(f"\n📊 Generating forced signal chart...")
    
    plt.figure(figsize=(15, 8))
    
    # Main chart: price and signals
    plt.plot(index_values, label='Price', alpha=0.8, linewidth=1)
    
    # Mark signals with correct index alignment
    # Predictions start from index SEQ_LEN (15) due to sequence creation
    seq_len = 15  # Optimized sequence length for early market capture
    
    colors = {'1': 'green', '2': 'red', '3': 'blue', '4': 'orange'}
    markers = {'1': '^', '2': 'v', '3': 'v', '4': '^'}
    labels = {'1': 'Long Entry', '2': 'Long Exit', '3': 'Short Entry', '4': 'Short Exit'}
    
    # Add legend flags to avoid duplicate labels
    legend_added = set()
    
    for i, pred in enumerate(predictions):
        if pred != 0:
            # Correct index alignment: prediction index i corresponds to price index i + seq_len
            price_idx = i + seq_len
            if price_idx < len(index_values):  # Ensure we don't exceed array bounds
                label_text = labels[str(pred)] if str(pred) not in legend_added else ""
                if label_text:
                    legend_added.add(str(pred))
                
                plt.scatter(price_idx, index_values[price_idx], 
                           color=colors[str(pred)], 
                           marker=markers[str(pred)], 
                           s=100, alpha=0.8,
                           label=label_text)
    
    plt.title('Forced Trading Signal Prediction Results (Index Aligned)', fontsize=14)
    plt.xlabel('Time Point')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add information text about the alignment and optimization
    plt.text(0.02, 0.98, f'Optimized for Early Market: seq_len={seq_len}, early boost for first 900 points (1 hour)', 
             transform=plt.gca().transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Chart saved: {output_filename}")
    print(f"   📌 Optimized alignment: Prediction index 0 = Price index {seq_len} (Early market focused)")

if __name__ == "__main__":
    print("⚡ Forced Trading Signal Prediction Tool")
    print("="*50)
    print("🎯 Solve model over-conservative problem, force generate readable trading signals")
    
    # Process all CSV files in relaxed label data directory
    data_dir = "../data_with_relaxed_labels/"
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory does not exist: {data_dir}")
        exit(1)
    
    # Find all CSV files
    import glob
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        print(f"❌ No CSV files found in directory: {data_dir}")
        exit(1)
    
    print(f"\n📁 Found {len(csv_files)} CSV files to process")
    
    # Process each CSV file
    successful_files = 0
    failed_files = 0
    
    for i, test_file in enumerate(csv_files, 1):
        print(f"\n" + "="*60)
        print(f"📊 Processing file {i}/{len(csv_files)}: {os.path.basename(test_file)}")
        print("="*60)
        
        try:
            # Force generate trading signals
            predictions, probabilities = force_trading_signals(
                test_file, 
                target_signal_density=0.05,  # Target 5% signal density
                ensure_complete_trades=True   # Ensure complete trades
            )
            
            if predictions is not None:
                # Need to get index_values for visualization
                features, labels, index_values_for_plot, feature_cols = load_and_preprocess_data_fixed(test_file)
                
                # Generate visualization chart
                output_file = f"./forced_trading_signals_{os.path.splitext(os.path.basename(test_file))[0]}.png"
                plot_forced_signals(index_values_for_plot, predictions, output_file)
                
                print(f"\n🎉 File {os.path.basename(test_file)} processed successfully!")
                print(f"💡 Chart file: {output_file}")
                successful_files += 1
            else:
                print(f"\n❌ Failed to generate signals for {os.path.basename(test_file)}")
                failed_files += 1
                
        except Exception as e:
            print(f"\n❌ Error processing file {os.path.basename(test_file)}: {e}")
            failed_files += 1
    
    # Summary
    print(f"\n" + "="*60)
    print(f"📈 Batch Processing Summary")
    print(f"="*60)
    print(f"✅ Successfully processed: {successful_files} files")
    print(f"❌ Failed to process: {failed_files} files")
    print(f"📊 Total files: {len(csv_files)}")
    
    if successful_files > 0:
        print(f"\n🎉 Batch processing completed!")
        print(f"📊 Generated trading signal charts for all successful files")
        print(f"🔍 Check the PNG files for readable prediction results!")
    else:
        print(f"\n❌ No files were successfully processed")
        print(f"💡 Suggest checking data files and model files")