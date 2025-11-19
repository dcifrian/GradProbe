"""
Debug the histogram-based binary search for WANDA pruning.

This test focuses on a single layer to understand:
1. Whether the cached histogram matches the actual distribution
2. Whether bin calculations are correct
3. Why we get "jumping from bin 0 to bin 0" errors
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gradprobe import GradProbe, WANDAPruning
from gradprobe.logger import Logger, LogLevel

# Initialize logger with DEBUG level to see all the details
# Log to file so we can analyze the full output
logger = Logger(
    program_name='debug_histogram_search',
    level=LogLevel.DEBUG,  # Show everything on console
    file_log_level=LogLevel.DEBUG  # Log everything to disk
)

logger.info("="*70)
logger.info("HISTOGRAM BINARY SEARCH DEBUG TEST")
logger.info("="*70)

# Try to import transformers
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    logger.error("transformers library not found. Please install it:")
    logger.error("pip install transformers")
    sys.exit(1)

logger.info("\nLoading TinyStories-33M model...")
model_name = "roneneldan/TinyStories-33M"
try:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
except Exception as e:
    logger.error(f"Error loading model: {e}")
    logger.info("Trying to download...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

logger.info(f"Model loaded successfully")
logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Prepare test data
test_text = """Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the park. She ran to get it and kicked it very hard. The ball flew high into the sky and landed in a tree. Lily was sad because she couldn't reach it. Then, a kind boy came and helped her get the ball down. Lily was happy again and they played together all day.

Once there was a brave knight named Tom. He had a shiny sword and a big horse. One day, Tom went on an adventure to find a dragon. He rode his horse through the dark forest and over the tall mountains. When he found the dragon, it was sleeping in a cave. Tom was very quiet and sneaked past the dragon. He found a treasure chest full of gold coins. Tom took the treasure and rode back home to share it with his family.

There was a happy bunny who lived in the woods. The bunny liked to hop and play with his friends. One sunny morning, the bunny found a big carrot in the garden. He was so excited! He called all his friends to come and see. Together, they shared the carrot and had a wonderful picnic. The bunny and his friends played games until the sun went down. Then they all went home to sleep.

A little boy named Max loved to build things. He had many blocks of different colors. One day, Max decided to build a tall tower. He stacked the blocks very carefully, one on top of another. The tower grew taller and taller until it was as high as Max's head. Max was very proud of his tower. He showed it to his mom and dad, and they clapped their hands. That night, Max dreamed about building even bigger towers."""

# Tokenize
logger.info("\nPreparing data...")
tokens = tokenizer.encode(test_text, return_tensors='pt')
logger.info(f"Total tokens: {tokens.shape[1]}")

# Create dataset with sliding windows
seq_length = 128
stride = 64
input_sequences = []
target_sequences = []

for i in range(0, tokens.shape[1] - seq_length - 1, stride):
    input_seq = tokens[:, i:i+seq_length]
    target_seq = tokens[:, i+1:i+seq_length+1]
    if input_seq.shape[1] == seq_length and target_seq.shape[1] == seq_length:
        input_sequences.append(input_seq)
        target_sequences.append(target_seq)

logger.info(f"Created {len(input_sequences)} sequences of length {seq_length}")

# Create dataloaders
all_inputs = torch.cat(input_sequences, dim=0)
all_targets = torch.cat(target_sequences, dim=0)
dataset = TensorDataset(all_inputs, all_targets)
dataloader_pruning = DataLoader(dataset, batch_size=1, shuffle=False)

# Evaluation function
def eval_perplexity(m):
    m.eval()
    total_loss = 0
    total_tokens = 0
    device = next(m.parameters()).device
    eval_data = list(zip(input_sequences, target_sequences))

    with torch.no_grad():
        for input_seq, target_seq in eval_data[:min(10, len(eval_data))]:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            outputs = m(input_seq)
            logits = outputs.logits
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_seq.reshape(-1),
                reduction='sum'
            )
            total_loss += loss.item()
            total_tokens += target_seq.numel()

    perplexity = math.exp(total_loss / total_tokens)
    return perplexity

# Loss function
def loss_fn(outputs, targets):
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
    else:
        logits = outputs

    loss = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1)
    )
    return loss

# Eval function for iterative pruning
def eval_fn_adjusted(m):
    perplexity = eval_perplexity(m)
    return -perplexity

logger.info(f"\nInitial perplexity: {eval_perplexity(model):.2f}")

# Save model state
import copy
saved_state = copy.deepcopy(model.state_dict())

logger.info("\n" + "="*70)
logger.info("RUNNING WANDA PRUNING WITH LAYERWISE MODE")
logger.info("="*70)
logger.info("Target: 10% sparsity")
logger.info("This will log detailed information for the first few layers")
logger.info("="*70)

# WANDA pruning with layerwise mode
wanda_strategy = WANDAPruning(dataloader=dataloader_pruning, num_batches=50)
pruner_wanda = GradProbe(model, wanda_strategy, use_fp16=True, use_gradient_checkpointing=True)

results_wanda = pruner_wanda.iterative_prune(
    dataloader=dataloader_pruning,
    loss_fn=loss_fn,
    eval_fn=eval_fn_adjusted,
    initial_sparsity=0.1,  # Just 10% to see the issue right away
    sparsity_step=0.1,
    max_accuracy_drop=3.0,
    num_batches=100,
    gradient_threshold=1.0,
    layerwise=True,  # Enable layerwise to trigger the histogram caching
    verbose=True,
    compare_baseline=True,
    layer_order="size"  # Prune largest layers first
)

logger.info("\n" + "="*70)
logger.info("TEST COMPLETE")
logger.info("="*70)
logger.info(f"Final perplexity: {eval_perplexity(model):.2f}")
logger.info(f"Final sparsity: {results_wanda['final_sparsity']:.2%}")
logger.info(f"\nLog file saved to: {logger.log_file_path}")
logger.info("Check the log file for detailed histogram and binary search information")
logger.info("="*70)
