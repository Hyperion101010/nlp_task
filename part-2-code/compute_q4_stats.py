"""
Script to compute Q4 data statistics (before and after preprocessing)
Computes statistics for Table 1 (before) and Table 2 (after preprocessing)

Table 1: Raw data (no preprocessing)
Table 2: Uses T5Dataset class to ensure it matches exactly what T5 sees during training
"""
from transformers import T5TokenizerFast
from load_data import load_lines, T5Dataset
import numpy as np

def compute_statistics(nl_path, sql_path, tokenizer, prefix=""):
    """
    Compute statistics for a dataset split.
    
    Args:
        nl_path: Path to natural language queries file
        sql_path: Path to SQL queries file
        tokenizer: T5 tokenizer
        prefix: Optional prefix to add to NL queries (for after preprocessing)
    
    Returns:
        Dictionary with statistics
    """
    # Load data
    nl_lines = load_lines(nl_path)
    sql_lines = load_lines(sql_path)
    
    assert len(nl_lines) == len(sql_lines), f"NL and SQL files must have same length: {len(nl_lines)} vs {len(sql_lines)}"
    
    num_examples = len(nl_lines)
    
    # Initialize lists and sets for statistics
    nl_lengths = []
    sql_lengths = []
    nl_vocab = set()
    sql_vocab = set()
    
    # Process each example
    for nl, sql in zip(nl_lines, sql_lines):
        # Add prefix if provided (for after preprocessing)
        nl_text = prefix + nl if prefix else nl
        
        # Tokenize with T5 tokenizer
        nl_tokens = tokenizer.encode(nl_text, add_special_tokens=False)
        sql_tokens = tokenizer.encode(sql, add_special_tokens=False)
        
        # Store lengths
        nl_lengths.append(len(nl_tokens))
        sql_lengths.append(len(sql_tokens))
        
        # Collect vocabulary (unique token IDs)
        nl_vocab.update(nl_tokens)
        sql_vocab.update(sql_tokens)
    
    # Compute statistics
    stats = {
        'num_examples': num_examples,
        'mean_nl_length': np.mean(nl_lengths),
        'mean_sql_length': np.mean(sql_lengths),
        'vocab_size_nl': len(nl_vocab),
        'vocab_size_sql': len(sql_vocab)
    }
    
    return stats

def compute_statistics_from_dataset(dataset, tokenizer):
    """
    Compute statistics using the actual T5Dataset class.
    This ensures Table 2 matches exactly what T5 sees during training.
    
    For Table 2, we compute statistics on:
    - encoder_ids: tokenized NL queries (after prefix "translate to SQL: " + tokenization)
    - decoder_target_ids: tokenized SQL queries (after tokenization)
    
    Args:
        dataset: T5Dataset instance (already preprocessed and tokenized)
        tokenizer: T5 tokenizer (not used, but kept for consistency)
    
    Returns:
        Dictionary with statistics
    """
    nl_lengths = []
    sql_lengths = []
    nl_vocab = set()
    sql_vocab = set()
    
    # Iterate through dataset to get tokenized examples
    for idx in range(len(dataset)):
        item = dataset[idx]
        
        # T5Dataset.__getitem__ returns:
        # - For train/dev: (encoder_ids, decoder_input_ids, decoder_target_ids, initial_decoder_token)
        # - For test: (encoder_ids, initial_decoder_token)
        
        if dataset.split == "test":
            # Test set: only encoder_ids available
            encoder_ids = item[0]
            # Convert tensor to list
            if hasattr(encoder_ids, 'tolist'):
                encoder_tokens = encoder_ids.tolist()
            else:
                encoder_tokens = list(encoder_ids)
            
            # For test, we only have encoder data
            nl_lengths.append(len(encoder_tokens))
            nl_vocab.update(encoder_tokens)
            # SQL stats not available for test set
        else:
            # Train/dev: both encoder and decoder targets available
            encoder_ids = item[0]  # encoder_ids (tokenized NL with prefix)
            decoder_target_ids = item[2]  # decoder_target_ids (tokenized SQL)
            
            # Convert tensors to lists
            if hasattr(encoder_ids, 'tolist'):
                encoder_tokens = encoder_ids.tolist()
            else:
                encoder_tokens = list(encoder_ids)
                
            if hasattr(decoder_target_ids, 'tolist'):
                decoder_target_tokens = decoder_target_ids.tolist()
            else:
                decoder_target_tokens = list(decoder_target_ids)
            
            # Store lengths (these are already tokenized, no padding at this stage)
            # Note: T5Dataset tokenizes with padding=False, so no padding to remove
            nl_lengths.append(len(encoder_tokens))
            sql_lengths.append(len(decoder_target_tokens))
            
            # Collect vocabulary (all token IDs, including special tokens)
            nl_vocab.update(encoder_tokens)
            sql_vocab.update(decoder_target_tokens)
    
    stats = {
        'num_examples': len(dataset),
        'mean_nl_length': np.mean(nl_lengths) if nl_lengths else 0,
        'mean_sql_length': np.mean(sql_lengths) if sql_lengths else 0,
        'vocab_size_nl': len(nl_vocab),
        'vocab_size_sql': len(sql_vocab)
    }
    
    return stats

def print_table(title, train_stats, dev_stats):
    """Print a formatted table with statistics"""
    print(f"\n{title}")
    print("=" * 70)
    print(f"{'Statistics Name':<40} {'Train':<15} {'Dev':<15}")
    print("-" * 70)
    print(f"{'Number of examples':<40} {train_stats['num_examples']:<15} {dev_stats['num_examples']:<15}")
    print(f"{'Mean sentence length':<40} {train_stats['mean_nl_length']:.2f} {dev_stats['mean_nl_length']:.2f}")
    print(f"{'Mean SQL query length':<40} {train_stats['mean_sql_length']:.2f} {dev_stats['mean_sql_length']:.2f}")
    print(f"{'Vocabulary size (natural language)':<40} {train_stats['vocab_size_nl']:<15} {dev_stats['vocab_size_nl']:<15}")
    print(f"{'Vocabulary size (SQL)':<40} {train_stats['vocab_size_sql']:<15} {dev_stats['vocab_size_sql']:<15}")
    print("=" * 70)

def main():
    # Initialize T5 tokenizer
    print("Loading T5 tokenizer...")
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    
    data_folder = "data"
    
    print("\n" + "=" * 70)
    print("Q4: Data Statistics")
    print("=" * 70)
    
    # ========== TABLE 1: BEFORE PREPROCESSING ==========
    print("\nComputing statistics BEFORE preprocessing...")
    train_stats_before = compute_statistics(
        f"{data_folder}/train.nl",
        f"{data_folder}/train.sql",
        tokenizer
    )
    
    dev_stats_before = compute_statistics(
        f"{data_folder}/dev.nl",
        f"{data_folder}/dev.sql",
        tokenizer
    )
    
    print_table("Table 1: Data statistics BEFORE preprocessing", 
                train_stats_before, dev_stats_before)
    
    # ========== TABLE 2: AFTER PREPROCESSING ==========
    print("\n\nComputing statistics AFTER preprocessing...")
    print("Using T5Dataset class to ensure Table 2 matches exactly what T5 sees during training")
    
    try:
        # Use the actual T5Dataset class to get preprocessed data
        train_dataset = T5Dataset(data_folder, "train")
        dev_dataset = T5Dataset(data_folder, "dev")
        
        train_stats_after = compute_statistics_from_dataset(train_dataset, tokenizer)
        dev_stats_after = compute_statistics_from_dataset(dev_dataset, tokenizer)
        
        print("✓ Successfully computed statistics using T5Dataset")
    except Exception as e:
        print(f"⚠ Warning: Could not use T5Dataset (may not be implemented yet): {e}")
        print("Falling back to manual preprocessing...")
        print("NOTE: You should implement T5Dataset first, then re-run this script")
        
        # Fallback: manual preprocessing (update this to match your preprocessing)
        prefix = ""  # Update this to match your T5Dataset preprocessing
        # Common options:
        # prefix = "translate to SQL: "
        # prefix = "sql: "
        
        train_stats_after = compute_statistics(
            f"{data_folder}/train.nl",
            f"{data_folder}/train.sql",
            tokenizer,
            prefix=prefix
        )
        
        dev_stats_after = compute_statistics(
            f"{data_folder}/dev.nl",
            f"{data_folder}/dev.sql",
            tokenizer,
            prefix=prefix
        )
    
    print_table("Table 2: Data statistics AFTER preprocessing", 
                train_stats_after, dev_stats_after)
    
    # Add model name row for Table 2
    print(f"\n{'Model name':<40} {'T5-small':<15} {'T5-small':<15}")
    
    # ========== SUMMARY FOR COPY-PASTE ==========
    print("\n" + "=" * 70)
    print("SUMMARY - Copy these values into your LaTeX tables:")
    print("=" * 70)
    
    print("\n--- TABLE 1 (Before preprocessing) ---")
    print(f"Train examples: {train_stats_before['num_examples']}")
    print(f"Dev examples: {dev_stats_before['num_examples']}")
    print(f"Train mean NL length: {train_stats_before['mean_nl_length']:.2f}")
    print(f"Dev mean NL length: {dev_stats_before['mean_nl_length']:.2f}")
    print(f"Train mean SQL length: {train_stats_before['mean_sql_length']:.2f}")
    print(f"Dev mean SQL length: {dev_stats_before['mean_sql_length']:.2f}")
    print(f"Train NL vocab size: {train_stats_before['vocab_size_nl']}")
    print(f"Dev NL vocab size: {dev_stats_before['vocab_size_nl']}")
    print(f"Train SQL vocab size: {train_stats_before['vocab_size_sql']}")
    print(f"Dev SQL vocab size: {dev_stats_before['vocab_size_sql']}")
    
    print("\n--- TABLE 2 (After preprocessing) ---")
    print(f"Model name: T5-small")
    print(f"Train mean NL length: {train_stats_after['mean_nl_length']:.2f}")
    print(f"Dev mean NL length: {dev_stats_after['mean_nl_length']:.2f}")
    print(f"Train mean SQL length: {train_stats_after['mean_sql_length']:.2f}")
    print(f"Dev mean SQL length: {dev_stats_after['mean_sql_length']:.2f}")
    print(f"Train NL vocab size: {train_stats_after['vocab_size_nl']}")
    print(f"Dev NL vocab size: {dev_stats_after['vocab_size_nl']}")
    print(f"Train SQL vocab size: {train_stats_after['vocab_size_sql']}")
    print(f"Dev SQL vocab size: {dev_stats_after['vocab_size_sql']}")
    print("=" * 70)

if __name__ == "__main__":
    main()
