# Assignment 4: Text-to-SQL with Neural Models

This repository contains the implementation for Assignment 4, which focuses on Text-to-SQL translation using neural sequence-to-sequence models and large language models.

## Repository Structure

```
hw4/
├── hw4-code/
│   ├── part-1-code/          # Data augmentation and transformation
│   │   ├── main.py
│   │   ├── utils.py
│   │   ├── requirements.txt
│   │   └── README.md
│   │
│   ├── part-2-code/           # Text-to-SQL models (T5 fine-tuning + LLM prompting)
│   │   ├── train_t5.py        # Main training script for T5 models
│   │   ├── load_data.py       # Data loading and preprocessing
│   │   ├── t5_utils.py        # T5 model utilities
│   │   ├── utils.py           # Evaluation and metrics utilities
│   │   ├── prompting.py       # LLM prompting experiments (Gemma, CodeGemma)
│   │   ├── prompting_utils.py # Prompting utilities
│   │   ├── evaluate.py        # Evaluation script
│   │   ├── find_mismatches.py # Error analysis tool
│   │   ├── data/              # Dataset (train/dev/test .nl and .sql files)
│   │   ├── results/           # Generated SQL predictions
│   │   ├── records/           # Database execution records (.pkl files)
│   │   ├── mismatches/        # Error analysis outputs
│   │   ├── checkpoints/       # Model checkpoints
│   │   ├── requirements.txt
│   │   └── README.md
│   │
│   └── submission_model/       # Final model checkpoints for submission
│
├── hw4-report.tex             # LaTeX report
└── README.md                  # This file
```

## Overview

### Part 1: Data Augmentation
Data augmentation and transformation experiments. See `hw4-code/part-1-code/README.md` for details.

### Part 2: Text-to-SQL Models
Implementation of Text-to-SQL translation using:
- **T5 Models**: Fine-tuning and training from scratch
- **LLMs**: Prompting-based experiments with Gemma and CodeGemma models

## Quick Start

### Part 1 Setup
```bash
cd hw4-code/part-1-code
conda create -n hw4-part-1-nlp python=3.9
conda activate hw4-part-1-nlp
pip install -r requirements.txt
```

### Part 2 Setup
```bash
cd hw4-code/part-2-code
conda create -n hw4-part-2-nlp python=3.10
conda activate hw4-part-2-nlp
pip install -r requirements.txt
```

## Part 2: Text-to-SQL Usage

### Training a T5 Model

**Fine-tuning from pre-trained T5:**
```bash
python train_t5.py --finetune \
    --learning_rate 1e-4 \
    --max_n_epochs 20 \
    --patience_epochs 10 \
    --batch_size 32 \
    --test_batch_size 32 \
    --scheduler_type cosine \
    --num_warmup_epochs 2 \
    --weight_decay 0.01 \
    --experiment_name my_experiment
```

**Training from scratch:**
```bash
python train_t5.py \
    --learning_rate 1e-3 \
    --max_n_epochs 20 \
    --patience_epochs 10 \
    --batch_size 32 \
    --experiment_name my_experiment
```

### Evaluation

Evaluate model predictions:
```bash
python evaluate.py \
    --predicted_sql results/t5_ft_my_experiment_dev.sql \
    --predicted_records records/t5_ft_my_experiment_dev.pkl \
    --development_sql data/dev.sql \
    --development_records records/ground_truth_dev.pkl
```

### Error Analysis

Find mismatches between predictions and ground truth:
```bash
python find_mismatches.py \
    --experiment_name my_experiment \
    --finetune
```

### LLM Prompting

Run prompting experiments with Gemma/CodeGemma:
```bash
python prompting.py
```

## Key Features

- **T5 Fine-tuning**: Support for fine-tuning pre-trained T5 models and training from scratch
- **Early Stopping**: Configurable patience-based early stopping
- **Evaluation Metrics**: SQL Exact Match, Record F1, Record EM, SQL error rate
- **Error Analysis**: Tools for qualitative and quantitative error analysis
- **LLM Integration**: Support for Gemma and CodeGemma models via Hugging Face

## Model Outputs

- **SQL Predictions**: Saved in `results/` directory as `.sql` files
- **Database Records**: Saved in `records/` directory as `.pkl` files
- **Model Checkpoints**: Saved in `checkpoints/` directory
- **Error Analysis**: Saved in `mismatches/` directory

## Submission

For Part 2, submit:
- Test SQL queries: `results/{t5_ft, t5_scr, gemma}_test.sql`
- Test records: `records/{t5_ft, t5_scr, gemma}_test.pkl`

Ensure predictions match the order of queries in `data/test.nl`.

## Requirements

- Python 3.9+ (Part 1) or 3.10+ (Part 2)
- PyTorch
- Transformers (Hugging Face)
- CUDA (recommended for training)

See individual `requirements.txt` files in each part for detailed dependencies.

## License

This is a course assignment repository.


