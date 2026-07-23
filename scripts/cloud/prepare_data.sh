#!/bin/bash
# =============================================================================
# Download training data for cloud instances
# =============================================================================
# Creates /data/training/data_smoke_1000.jsonl from the local 12TB drive path
# or generates synthetic data for testing.
#
# Usage: bash scripts/cloud/prepare_data.sh [OUTPUT_DIR]
# =============================================================================

set -e

OUTPUT_DIR="${1:-/data/training}"
OUTPUT_FILE="$OUTPUT_DIR/data_smoke_1000.jsonl"

mkdir -p "$OUTPUT_DIR"

if [ -f "$OUTPUT_FILE" ] && [ $(wc -l < "$OUTPUT_FILE") -gt 900 ]; then
    echo "Data already exists: $OUTPUT_FILE ($(wc -l < $OUTPUT_FILE) lines)"
    exit 0
fi

echo "Creating training data at $OUTPUT_FILE..."

# Generate synthetic plaintext data for smoke testing
# In production, replace with actual dataset download
python3 -c "
import json, random, sys

# Diverse text templates for testing
templates = [
    'The development of artificial intelligence has transformed {field} by enabling {capability}. Researchers at {institution} have demonstrated that {method} can achieve {result} on {benchmark}.',
    'In this paper, we propose {method}, a novel approach to {task} that leverages {technique}. Our experiments show that {method} outperforms {baseline} by {margin}% on {dataset}.',
    '{field} has seen rapid progress in recent years, with {method} achieving state-of-the-art results on {benchmark}. However, challenges remain in {challenge}, particularly when {condition}.',
    'The architecture consists of {n} transformer layers with hidden dimension {d} and {h} attention heads. Training was conducted on {hardware} for {steps} steps using the AdamW optimizer.',
    'We evaluate our approach on {n} benchmarks spanning {domains}. Results show consistent improvements over baseline methods, with the largest gains on {benchmark} ({margin}% improvement).',
]

fields = ['natural language processing', 'computer vision', 'speech recognition', 'robotics', 'drug discovery']
capabilities = ['real-time inference', 'multi-modal understanding', 'few-shot learning', 'zero-shot transfer']
institutions = ['Stanford', 'MIT', 'Google Research', 'Meta AI', 'DeepMind']
methods = ['Transformer-XL', 'Repr-Align', 'DiffusionLM', 'Sparse Attention', 'Mixture of Experts']
results = ['state-of-the-art performance', 'significant improvements', 'competitive results']
benchmarks = ['GLUE', 'SuperGLUE', 'MMLU', 'HumanEval', 'GSM8K']
baselines = ['GPT-3', 'LLaMA-2', 'Mistral', 'the previous approach']
tasks = ['text generation', 'code completion', 'question answering', 'summarization']
techniques = ['discrete diffusion', 'bidirectional attention', 'representation alignment']
datasets = ['WikiText-103', 'The Pile', 'FineWeb', 'C4']
domains = ['language understanding', 'reasoning', 'code generation']
challenges = ['long-range dependencies', 'computational efficiency', 'data quality']
conditions = ['training data is limited', 'model size is constrained', 'latency requirements are strict']
hardware = ['8x A100 GPUs', '4x H100 GPUs', 'a TPU v4 pod']

random.seed(42)
with open('$OUTPUT_FILE', 'w') as f:
    for i in range(1000):
        template = random.choice(templates)
        text = template.format(
            field=random.choice(fields),
            capability=random.choice(capabilities),
            institution=random.choice(institutions),
            method=random.choice(methods),
            result=random.choice(results),
            benchmark=random.choice(benchmarks),
            baseline=random.choice(baselines),
            margin=random.randint(2, 15),
            task=random.choice(tasks),
            technique=random.choice(techniques),
            dataset=random.choice(datasets),
            n=random.choice([12, 24, 32, 48, 64]),
            d=random.choice([768, 1024, 2048, 4096, 8192]),
            h=random.choice([8, 12, 16, 32, 64]),
            steps=random.choice([10000, 50000, 100000, 500000]),
            domains=random.choice(domains),
            challenge=random.choice(challenges),
            condition=random.choice(conditions),
            hardware=random.choice(hardware),
        )
        f.write(json.dumps({'text': text}) + '\n')

print(f'Generated 1000 examples to $OUTPUT_FILE')
" 2>&1

echo "Done: $(wc -l < $OUTPUT_FILE) examples at $OUTPUT_FILE"
