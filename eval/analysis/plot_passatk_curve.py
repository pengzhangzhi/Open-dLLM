#!/usr/bin/env python3
"""
Plot pass@k curve from HumanEval evaluation results.
Creates an academic-style line plot showing the relationship between k and pass@k performance.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Set matplotlib to use a backend that doesn't require display
import matplotlib
matplotlib.use('Agg')

# Set font to Times New Roman for academic style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

def extract_passatk_data(json_file_path):
    """
    Extract pass@k data from the evaluation results JSON file.
    Only extract powers of 2 for cleaner visualization.
    
    Args:
        json_file_path (str): Path to the JSON results file
        
    Returns:
        tuple: (k_values, passatk_values) sorted by k
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Extract humaneval results
    humaneval_results = data['results']['humaneval']
    
    # Define powers of 2 we want to extract
    target_k_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    
    # Find pass@k metrics for powers of 2
    passatk_data = {}
    for key, value in humaneval_results.items():
        if key.startswith('pass@') and key.endswith(',create_test') and '_stderr' not in key:
            # Extract k value from key like "pass@1,create_test"
            k_str = key.split('@')[1].split(',')[0]
            k = int(k_str)
            if k in target_k_values:
                passatk_data[k] = value
    
    # Sort by k value
    sorted_items = sorted(passatk_data.items())
    k_values = [item[0] for item in sorted_items]
    passatk_values = [item[1] * 100 for item in sorted_items]  # Convert to percentage
    
    return k_values, passatk_values

def create_academic_plot(k_values, passatk_values, output_path):
    """
    Create an academic-style line plot for pass@k curve.
    
    Args:
        k_values (list): List of k values
        passatk_values (list): List of pass@k values (in percentage)
        output_path (str): Path to save the plot
    """
    # Set up the plot with academic style
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    
    # Plot the curve with academic style (use square markers)
    ax.plot(k_values, passatk_values, color='black', linewidth=2.0, 
            marker='s', markersize=6, markerfacecolor='black', 
            markeredgecolor='black', markeredgewidth=1.0, 
            label='Open-dCoder-0.5B')
    
    # Customize the plot
    ax.set_xlabel('Number of Samples k (log scale)', fontsize=14, fontweight='bold')
    ax.set_ylabel('pass@k', fontsize=14, fontweight='bold')
    ax.set_title('HumanEval', fontsize=16, fontweight='bold', pad=20)
    
    # Set x-axis to log scale for better visualization of powers of 2
    ax.set_xscale('log', base=2)
    ax.set_xlim(0.8, 600)
    ax.set_ylim(0, 70)
    
    # Set custom x-axis ticks as powers of 2
    x_ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(x) for x in x_ticks])
    
    # Customize grid (academic style)
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5, color='gray')
    
    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='x', rotation=0)
    
    # Set y-axis ticks
    y_ticks = [10, 20, 30, 40, 50, 60, 70]
    ax.set_yticks(y_ticks)
    
    # Remove annotations for cleaner look (like reference image)
    
    # Customize spines (academic style)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('black')
    
    # Add legend in academic style with adaptive positioning
    ax.legend(frameon=True, fancybox=False, shadow=False, fontsize=12, 
              loc='best', framealpha=1.0, edgecolor='black')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot in both PDF and PNG formats
    plt.savefig(output_path, format='pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    # Save PNG version
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, format='png', bbox_inches='tight', 
                facecolor='white', edgecolor='none', dpi=300)
    
    plt.close()
    
    print("Plot saved to: {}".format(output_path))
    print("PNG version saved to: {}".format(png_path))

def main():
    """Main function to generate the pass@k curve plot."""
    
    # File paths
    json_file = "/data/szhang967/dLLM-training/eval/lm_harness/evals_results/humaneval-ns0/fredzzp__open-dcoder-0.5B/results_2025-08-31T23-35-05.653402.json"
    output_dir = "/data/szhang967/dLLM-training/eval/analysis"
    output_file = os.path.join(output_dir, "humaneval_passatk_curve.pdf")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print("Extracting pass@k data from evaluation results...")
    
    # Extract data
    try:
        k_values, passatk_values = extract_passatk_data(json_file)
        print("Found {} pass@k data points".format(len(k_values)))
        print("k range: {} to {}".format(min(k_values), max(k_values)))
        print("pass@k range: {:.1f}% to {:.1f}%".format(min(passatk_values), max(passatk_values)))
        
        # Create the plot
        print("Creating academic-style plot...")
        create_academic_plot(k_values, passatk_values, output_file)
        
        # Print summary statistics
        print("\nSummary statistics:")
        print("pass@1: {:.1f}%".format(passatk_values[0]))
        if 10 in k_values:
            print("pass@10: {:.1f}%".format(passatk_values[k_values.index(10)]))
        if 100 in k_values:
            print("pass@100: {:.1f}%".format(passatk_values[k_values.index(100)]))
        print("pass@512: {:.1f}%".format(passatk_values[-1]))
        
    except Exception as e:
        print("Error processing data: {}".format(e))
        return
    
    print("\nPlot generation completed successfully!")
    print("Output saved to: {}".format(output_file))

if __name__ == "__main__":
    main()
