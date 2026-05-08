#!/usr/bin/env python3
"""
Script to count total tokens in a parquet dataset for pretraining.
Efficiently processes large datasets with progress tracking and statistics.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def count_tokens_in_text(text: str, tokenizer) -> int:
    """Count tokens in a single text string."""
    if not text or not isinstance(text, str):
        return 0
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def process_parquet_file(file_path: str, tokenizer, text_column: str = "text") -> Tuple[int, int, Dict]:
    """Process a single parquet file and return token count statistics."""
    try:
        # Load the parquet file
        df = pd.read_parquet(file_path)
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in {file_path}. Available columns: {list(df.columns)}")
        
        total_tokens = 0
        total_texts = len(df)
        text_lengths = []
        
        # Process each text in the file
        for text in df[text_column]:
            if pd.isna(text):
                continue
            token_count = count_tokens_in_text(str(text), tokenizer)
            total_tokens += token_count
            text_lengths.append(token_count)
        
        # Calculate statistics
        stats = {
            "total_tokens": total_tokens,
            "total_texts": total_texts,
            "avg_tokens_per_text": total_tokens / total_texts if total_texts > 0 else 0,
            "min_tokens": min(text_lengths) if text_lengths else 0,
            "max_tokens": max(text_lengths) if text_lengths else 0,
            "median_tokens": sorted(text_lengths)[len(text_lengths)//2] if text_lengths else 0,
        }
        
        return total_tokens, total_texts, stats
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0, 0, {}


def count_tokens_in_dataset(data_dir: str, tokenizer_name: str = "Qwen/Qwen2.5-Coder-0.5B", 
                           text_column: str = "text", max_files: int = None) -> Dict:
    """Count tokens in all parquet files in a directory."""
    
    # Load tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Find all parquet files
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Directory {data_dir} does not exist")
    
    parquet_files = list(data_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    
    # Sort files for consistent processing
    parquet_files.sort()
    
    if max_files:
        parquet_files = parquet_files[:max_files]
        print(f"Processing first {max_files} files only")
    
    print(f"Found {len(parquet_files)} parquet files to process")
    
    # Process files with progress bar
    total_tokens = 0
    total_texts = 0
    file_stats = []
    
    start_time = time.time()
    
    for file_path in tqdm(parquet_files, desc="Processing files"):
        tokens, texts, stats = process_parquet_file(str(file_path), tokenizer, text_column)
        total_tokens += tokens
        total_texts += texts
        
        file_stats.append({
            "file": file_path.name,
            "tokens": tokens,
            "texts": texts,
            "avg_tokens_per_text": stats.get("avg_tokens_per_text", 0),
        })
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Calculate overall statistics
    overall_stats = {
        "total_files": len(parquet_files),
        "total_tokens": total_tokens,
        "total_texts": total_texts,
        "avg_tokens_per_text": total_tokens / total_texts if total_texts > 0 else 0,
        "avg_tokens_per_file": total_tokens / len(parquet_files) if parquet_files else 0,
        "processing_time_seconds": processing_time,
        "tokens_per_second": total_tokens / processing_time if processing_time > 0 else 0,
        "file_stats": file_stats,
    }
    
    return overall_stats


def print_statistics(stats: Dict):
    """Print formatted statistics."""
    print("\n" + "="*60)
    print("DATASET TOKEN COUNTING RESULTS")
    print("="*60)
    
    print(f"Total Files Processed: {stats['total_files']:,}")
    print(f"Total Texts: {stats['total_texts']:,}")
    print(f"Total Tokens: {stats['total_tokens']:,}")
    print(f"Average Tokens per Text: {stats['avg_tokens_per_text']:.1f}")
    print(f"Average Tokens per File: {stats['avg_tokens_per_file']:,.0f}")
    
    print(f"\nProcessing Time: {stats['processing_time_seconds']:.1f} seconds")
    print(f"Processing Speed: {stats['tokens_per_second']:,.0f} tokens/second")
    
    # Convert to more readable units
    tokens_billions = stats['total_tokens'] / 1_000_000_000
    tokens_trillions = stats['total_tokens'] / 1_000_000_000_000
    
    print(f"\nToken Count in Different Units:")
    print(f"  Tokens: {stats['total_tokens']:,}")
    print(f"  Billions: {tokens_billions:.3f}B")
    if tokens_trillions >= 0.001:
        print(f"  Trillions: {tokens_trillions:.6f}T")
    
    # Show top 10 files by token count
    print(f"\nTop 10 Files by Token Count:")
    sorted_files = sorted(stats['file_stats'], key=lambda x: x['tokens'], reverse=True)
    for i, file_stat in enumerate(sorted_files[:10]):
        print(f"  {i+1:2d}. {file_stat['file']}: {file_stat['tokens']:,} tokens ({file_stat['texts']:,} texts)")


def main():
    parser = argparse.ArgumentParser(description="Count tokens in parquet dataset")
    parser.add_argument("--data_dir", type=str, required=True, 
                       help="Directory containing parquet files")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-Coder-0.5B",
                       help="Tokenizer to use for counting")
    parser.add_argument("--text_column", type=str, default="text",
                       help="Name of the text column in parquet files")
    parser.add_argument("--max_files", type=int, default=None,
                       help="Maximum number of files to process (for testing)")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Save detailed results to JSON file")
    
    args = parser.parse_args()
    
    try:
        print(f"Starting token counting for dataset: {args.data_dir}")
        print(f"Using tokenizer: {args.tokenizer}")
        print(f"Text column: {args.text_column}")
        
        stats = count_tokens_in_dataset(
            data_dir=args.data_dir,
            tokenizer_name=args.tokenizer,
            text_column=args.text_column,
            max_files=args.max_files
        )
        
        print_statistics(stats)
        
        # Save detailed results if requested
        if args.output_file:
            import json
            with open(args.output_file, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"\nDetailed results saved to: {args.output_file}")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
