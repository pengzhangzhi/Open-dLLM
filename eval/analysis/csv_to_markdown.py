#!/usr/bin/env python3
"""
Convert CSV table to Markdown format with values multiplied by 100 and rounded to 1 decimal place.
"""

import csv
import os

def format_value(value):
    """
    Convert a numeric string to percentage format (multiply by 100, round to 1 decimal place).
    Returns the original value if it's not a valid number.
    """
    try:
        # Try to convert to float
        num = float(value)
        # Multiply by 100 and round to 1 decimal place
        return f"{num * 100:.1f}"
    except (ValueError, TypeError):
        # If conversion fails, return the original value (likely a header or label)
        return value

def csv_to_markdown(csv_file_path, output_file_path=None):
    """
    Convert CSV file to Markdown table format.
    
    Args:
        csv_file_path (str): Path to the input CSV file
        output_file_path (str): Path to the output markdown file. If None, prints to console.
    """
    
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found: {csv_file_path}")
        return
    
    markdown_lines = []
    
    with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
        
        if not rows:
            print("Error: CSV file is empty")
            return
        
        # Process each row
        for i, row in enumerate(rows):
            # Filter out empty cells
            row = [cell for cell in row if cell.strip()]
            
            if not row:
                continue
                
            # Format values (skip first column which contains labels)
            formatted_row = []
            for j, cell in enumerate(row):
                if j == 0:  # First column (labels), keep as is
                    formatted_row.append(cell.strip())
                else:  # Other columns, format as percentages
                    formatted_row.append(format_value(cell.strip()))
            
            # Create markdown table row
            markdown_row = "| " + " | ".join(formatted_row) + " |"
            markdown_lines.append(markdown_row)
            
            # Add header separator after first row
            if i == 0:
                separator = "|" + "|".join([" --- " for _ in formatted_row]) + "|"
                markdown_lines.append(separator)
    
    # Join all lines
    markdown_content = "\n".join(markdown_lines)
    
    # Output result
    if output_file_path:
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            outfile.write(markdown_content)
        print(f"Markdown table saved to: {output_file_path}")
    else:
        print("Markdown Table:")
        print("=" * 50)
        print(markdown_content)
    
    return markdown_content

def main():
    # Define file paths
    csv_file = "/data/szhang967/dLLM-training/eval/analysis/dllm_0.5b_metrics.csv"
    markdown_file = "/data/szhang967/dLLM-training/eval/analysis/dllm_0.5b_metrics.md"
    
    # Convert CSV to Markdown
    print("Converting CSV to Markdown format...")
    csv_to_markdown(csv_file, markdown_file)
    
    # Also display the result
    print("\nPreview of the generated markdown table:")
    print("=" * 50)
    csv_to_markdown(csv_file)

if __name__ == "__main__":
    main()
