"""
Script to generate Table 3 content for Q5 based on your actual implementation.
This extracts hyperparameters and model details from your code.
"""
import inspect
from load_data import T5Dataset, normal_collate_fn, test_collate_fn
from train_t5 import get_args
from t5_utils import initialize_optimizer_and_scheduler

def extract_hyperparameters():
    """Extract hyperparameters from train_t5.py defaults"""
    args = get_args()
    
    # Note: These are defaults. When you actually train, update these with your chosen values
    hyperparams = {
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'test_batch_size': args.test_batch_size,
        'optimizer_type': args.optimizer_type,
        'weight_decay': args.weight_decay,
        'scheduler_type': args.scheduler_type,
        'num_warmup_epochs': args.num_warmup_epochs,
        'max_n_epochs': args.max_n_epochs,  # You'll set this when training
        'patience_epochs': args.patience_epochs,  # You'll set this when training
    }
    return hyperparams

def extract_tokenization_details():
    """Extract tokenization details from load_data.py"""
    # Read the actual code to get max_length
    with open('load_data.py', 'r') as f:
        code = f.read()
        if 'max_length=512' in code:
            max_length = 512
        else:
            max_length = "CHECK_CODE"
    
    return {
        'tokenizer': 'google-t5/t5-small',
        'max_length': max_length,
        'padding': 'Dynamic padding in collate function',
        'truncation': True,
        'bos_token': '<extra_id_0>',
    }

def generate_table3_content():
    """Generate Table 3 content based on your implementation"""
    
    hyperparams = extract_hyperparameters()
    tokenization = extract_tokenization_details()
    
    print("=" * 80)
    print("TABLE 3: Details of the best-performing T5 model configurations")
    print("=" * 80)
    print()
    
    # Row 1: Data processing
    print("Row 1 - Data processing:")
    print("-" * 80)
    print("""For each split (train, dev, test), I read the .nl files as natural-language 
inputs and the .sql files as targets (only for train/dev). I removed only trailing 
newlines and kept the original casing and punctuation. For the encoder input, I prepend 
a fixed task prefix "translate to SQL: " to every natural-language query (e.g., 
"translate to SQL: show me all flights from Boston to Denver"). I do not modify or 
normalize the SQL strings except stripping whitespace. I do not remove stopwords or 
apply stemming/lemmatization; the model sees almost the raw text plus the prefix.""")
    print()
    
    # Row 2: Tokenization
    print("Row 2 - Tokenization:")
    print("-" * 80)
    print(f"""I use the HuggingFace {tokenization['tokenizer']} tokenizer (T5TokenizerFast) 
for both encoder and decoder.

Encoder side: For each example, I tokenize the string "translate to SQL: <natural_language_query>" 
with max_length={tokenization['max_length']}, truncation={tokenization['truncation']}, and 
padding=False in the dataset. Padding is handled later in the collate function via 
torch.nn.utils.rnn.pad_sequence, using 0 (the T5 pad token id) as the padding value. 
The encoder attention mask is a binary tensor where positions with non-pad tokens are 1 
and pad positions are 0.

Decoder targets: I tokenize the raw SQL string using the same tokenizer with 
max_length={tokenization['max_length']}, truncation={tokenization['truncation']}, and 
padding=False. These token ids are used as decoder targets (labels).

Decoder inputs: For teacher forcing, I create decoder inputs by shifting the target tokens 
to the right and adding a beginning-of-sequence token at the front. Concretely, I take 
decoder_input_ids = [BOS] + target_ids[:-1], where BOS is implemented as the T5 special 
token {tokenization['bos_token']}.

Dynamic padding: In the normal_collate_fn, I pad encoder ids, decoder inputs, and decoder 
targets to the maximum length in the batch with pad_sequence(..., batch_first=True, 
padding_value=0). For the test set, test_collate_fn pads only the encoder ids and returns 
the attention mask and an initial decoder token ({tokenization['bos_token']}) per example.""")
    print()
    
    # Row 3: Architecture
    print("Row 3 - Architecture:")
    print("-" * 80)
    print("""I fine-tune the pretrained google-t5/t5-small model, a standard encoder–decoder 
transformer with shared embeddings. I do not modify the architecture in any way: the encoder, 
decoder, and language modeling head are the same as in the pretrained T5.

I fine-tune all parameters of the model (no freezing). The encoder consumes the preprocessed 
input "translate to SQL: <NL query>", and the decoder is trained in a teacher-forcing setup. 
Decoder inputs are constructed by prepending the special token <extra_id_0> (used as BOS) and 
shifting the SQL target tokens to the right.

During generation, the model uses only the BOS token as the first decoder input, and outputs 
tokens autoregressively using its LM head.""")
    print()
    
    # Row 4: Hyperparameters
    print("Row 4 - Hyperparameters:")
    print("-" * 80)
    
    # Format hyperparameters
    lr_str = f"{hyperparams['learning_rate']:.0e}" if hyperparams['learning_rate'] < 1 else str(hyperparams['learning_rate'])
    
    scheduler_desc = ""
    if hyperparams['scheduler_type'] == "none":
        scheduler_desc = "No scheduler; fixed learning rate."
    elif hyperparams['scheduler_type'] == "cosine":
        warmup_steps = hyperparams['num_warmup_epochs']
        scheduler_desc = f"Cosine learning-rate schedule with {warmup_steps} warmup epochs."
    elif hyperparams['scheduler_type'] == "linear":
        warmup_steps = hyperparams['num_warmup_epochs']
        scheduler_desc = f"Linear learning-rate schedule with {warmup_steps} warmup epochs."
    
    weight_decay_str = f"{hyperparams['weight_decay']}" if hyperparams['weight_decay'] > 0 else "None"
    
    print(f"""I fine-tune the model using the {hyperparams['optimizer_type']} optimizer on all 
parameters of the T5 model. I use a learning rate of {lr_str}{' with no warmup and no LR scheduler' if hyperparams['scheduler_type'] == 'none' else f' with {scheduler_desc.lower()}'}. 
I train for {hyperparams['max_n_epochs'] if hyperparams['max_n_epochs'] > 0 else 'N'} epochs and select the 
final model based on development set performance.

The batch size is {hyperparams['batch_size']} for training and {hyperparams['test_batch_size']} for evaluation, 
limited by GPU memory. The encoder and decoder tokenization both use a maximum sequence length of 
{tokenization['max_length']} with truncation.

I use teacher forcing during training, with decoder input sequences created by shifting target SQL 
tokens and prepending <extra_id_0>. Padding tokens are masked out by setting their token IDs to 0 
in both encoder and decoder attention masks.

I do not apply label smoothing or gradient clipping. Weight decay is set to {weight_decay_str if hyperparams['weight_decay'] > 0 else '0 (no weight decay)'}. 
Stopping criteria: {'Early stopping with patience of ' + str(hyperparams['patience_epochs']) + ' epochs' if hyperparams['patience_epochs'] > 0 else 'Fixed number of epochs'} 
and select the best model by development set Record F1.""")
    
    print()
    print("=" * 80)
    print("NOTE: Update the following values after you actually train:")
    print(f"  - Learning rate: {hyperparams['learning_rate']} (default, you may want to change to 1e-4 or 5e-5)")
    print(f"  - Number of epochs: {hyperparams['max_n_epochs']} (currently 0, set when training)")
    print(f"  - Patience epochs: {hyperparams['patience_epochs']} (currently 0, set when training)")
    print("=" * 80)

if __name__ == "__main__":
    generate_table3_content()

