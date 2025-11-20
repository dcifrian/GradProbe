"""
Test GradProbe on TinyStories-33M language model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gradprobe import GradProbe, MagnitudePruning, WANDAPruning
from gradprobe.logger import Logger, LogLevel

# Initialize logger
logger = Logger(program_name='test_tinystories', level=LogLevel.INFO)

# Try to import transformers
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    logger.error("transformers library not found. Please install it:")
    logger.error("pip install transformers")
    sys.exit(1)

logger.info("Loading TinyStories-33M model...")
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

# Prepare test data - use multiple stories to get enough tokens
test_text = """Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the park. She ran to get it and kicked it very hard. The ball flew high into the sky and landed in a tree. Lily was sad because she couldn't reach it. Then, a kind boy came and helped her get the ball down. Lily was happy again and they played together all day.

Once there was a brave knight named Tom. He had a shiny sword and a big horse. One day, Tom went on an adventure to find a dragon. He rode his horse through the dark forest and over the tall mountains. When he found the dragon, it was sleeping in a cave. Tom was very quiet and sneaked past the dragon. He found a treasure chest full of gold coins. Tom took the treasure and rode back home to share it with his family.

There was a happy bunny who lived in the woods. The bunny liked to hop and play with his friends. One sunny morning, the bunny found a big carrot in the garden. He was so excited! He called all his friends to come and see. Together, they shared the carrot and had a wonderful picnic. The bunny and his friends played games until the sun went down. Then they all went home to sleep.

A little boy named Max loved to build things. He had many blocks of different colors. One day, Max decided to build a tall tower. He stacked the blocks very carefully, one on top of another. The tower grew taller and taller until it was as high as Max's head. Max was very proud of his tower. He showed it to his mom and dad, and they clapped their hands. That night, Max dreamed about building even bigger towers."""

# Tokenize
logger.info("\nPreparing data...")
tokens = tokenizer.encode(test_text, return_tensors='pt')
logger.info(f"Total tokens: {tokens.shape[1]}")

# Create dataset with sliding windows
# For language modeling, we need input and target sequences
seq_length = 128
stride = 64  # Overlapping windows
input_sequences = []
target_sequences = []

for i in range(0, tokens.shape[1] - seq_length - 1, stride):  # -1 for target shift
    input_seq = tokens[:, i:i+seq_length]
    target_seq = tokens[:, i+1:i+seq_length+1]  # Shifted by 1
    if input_seq.shape[1] == seq_length and target_seq.shape[1] == seq_length:
        input_sequences.append(input_seq)
        target_sequences.append(target_seq)

logger.info(f"Created {len(input_sequences)} sequences of length {seq_length}")

# Create dataloaders
all_inputs = torch.cat(input_sequences, dim=0)
all_targets = torch.cat(target_sequences, dim=0)
dataset = TensorDataset(all_inputs, all_targets)
dataloader_pruning = DataLoader(dataset, batch_size=1, shuffle=False)

# Also store for evaluation
eval_data = list(zip(input_sequences, target_sequences))

# Evaluation function - compute perplexity
def eval_perplexity(m):
    m.eval()
    total_loss = 0
    total_tokens = 0

    # Detect model device
    device = next(m.parameters()).device

    with torch.no_grad():
        # Use first few sequences for eval
        for input_seq, target_seq in eval_data[:min(10, len(eval_data))]:
            # Move tensors to model device
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

# Custom loss function for pruning
# The pruner will call this with loss_fn(model_output, target_tokens)
def loss_fn(outputs, targets):
    # outputs is the model output from model(input_tokens)
    # targets is the target tokens (shifted by 1 from inputs)
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
    else:
        logits = outputs

    loss = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1)
    )
    return loss

logger.info(f"\nInitial perplexity: {eval_perplexity(model):.2f}")

# Save model state
import copy
saved_state = copy.deepcopy(model.state_dict())

# Wrapper for perplexity as evaluation function
# iterative_prune expects a function that returns a percentage-like metric
# We'll return negative perplexity so lower is better, and scale it
def eval_fn(m):
    perplexity = eval_perplexity(m)
    # Return negative log perplexity scaled by 100 so it looks like accuracy
    # This way increases in perplexity = decreases in this metric
    # We want to stop when perplexity increases significantly
    # Let's just return perplexity directly and adjust the stopping logic
    return perplexity

# Actually, the iterative_prune uses accuracy_drop = initial_accuracy - accuracy
# So if we return perplexity, accuracy_drop = initial_perplexity - current_perplexity
# If perplexity goes up, this becomes negative
# We need perplexity_increase = current_perplexity - initial_perplexity
#
# Let me modify the approach: return -perplexity so that increases in perplexity
# show up as decreases in the metric

def eval_fn_adjusted(m):
    perplexity = eval_perplexity(m)
    # Return negative perplexity so higher perplexity = lower score
    # Then "accuracy_drop" will be initial_score - current_score
    # = -initial_perplexity - (-current_perplexity)
    # = current_perplexity - initial_perplexity
    # = perplexity increase (which is what we want!)
    return -perplexity

logger.info("\n" + "="*70)
logger.info("TEST 1: Regular Iterative Pruning")
logger.info("="*70)
model.load_state_dict(saved_state)
pruner = GradProbe(model, MagnitudePruning(),use_fp16=True,use_gradient_checkpointing=True)

results = pruner.iterative_prune(
    dataloader=dataloader_pruning,
    loss_fn=loss_fn,
    eval_fn=eval_fn_adjusted,
    initial_sparsity=0.1,
    sparsity_step=0.1,
    max_accuracy_drop=3.0,  # Stop at 3 point perplexity increase
    num_batches=100,  # Use 100 sequences for gradient computation
    gradient_threshold=2.0,  # Much lower threshold for LLMs
    layerwise=False,
    verbose=True,
    compare_baseline=True
)

logger.info("\n" + "="*70)
logger.info("FINAL RESULTS - MAGNITUDE")
logger.info("="*70)
logger.info(f"Final perplexity: {eval_perplexity(model):.2f}")
logger.info(f"Final sparsity: {results['final_sparsity']:.2%}")
logger.info("="*70)

# Test 2: WANDA pruning
logger.info("\n" + "="*70)
logger.info("TEST 2: WANDA Pruning")
logger.info("="*70)
model.load_state_dict(saved_state)

# WANDA needs the dataloader to collect activations
wanda_strategy = WANDAPruning(dataloader=dataloader_pruning, num_batches=50)
pruner_wanda = GradProbe(model, wanda_strategy, use_fp16=True, use_gradient_checkpointing=True)

results_wanda = pruner_wanda.iterative_prune(
    dataloader=dataloader_pruning,
    loss_fn=loss_fn,
    eval_fn=eval_fn_adjusted,
    initial_sparsity=0.1,
    sparsity_step=0.1,
    max_accuracy_drop=3.0,  # Stop at 3 point perplexity increase
    num_batches=100,  # Use 100 sequences for gradient computation
    gradient_threshold=1.0,  # Lower threshold for WANDA on LLMs
    layerwise=True,
    verbose=True,
    compare_baseline=True,  # Compare with WANDA-only (no gradient filtering)
    experimental_tune_both_steps=True,  # EXPERIMENTAL: Re-prune both steps together
    layer_order="size"  # NEW: Prune largest layers first
)

logger.info("\n" + "="*70)
logger.info("FINAL RESULTS - WANDA")
logger.info("="*70)
logger.info(f"Final perplexity: {eval_perplexity(model):.2f}")
logger.info(f"Final sparsity: {results_wanda['final_sparsity']:.2%}")
logger.info("="*70)

logger.info("\n" + "="*70)
logger.info("COMPARISON: MAGNITUDE vs WANDA")
logger.info("="*70)
logger.info(f"Magnitude pruning:")
logger.info(f"  Initial perplexity: {-results['initial_accuracy']:.2f}")
logger.info(f"  Final perplexity: {-results['final_accuracy']:.2f}")
logger.info(f"  Sparsity: {results['final_sparsity']:.2%}")
logger.info(f"\nWANDA pruning:")
logger.info(f"  Initial perplexity: {-results_wanda['initial_accuracy']:.2f}")
logger.info(f"  Final perplexity: {-results_wanda['final_accuracy']:.2f}")
logger.info(f"  Sparsity: {results_wanda['final_sparsity']:.2%}")
logger.info("="*70)

# Generate text samples to compare quality
logger.info("\n" + "="*70)
logger.info("TEXT GENERATION COMPARISON")
logger.info("="*70)

# Reload original model
model.load_state_dict(saved_state)
device = next(model.parameters()).device

prompt_text = "Once upon a time, there was a little girl who"
prompt_tokens = tokenizer.encode(prompt_text, return_tensors='pt').to(device)

logger.info(f"\nPrompt: \"{prompt_text}\"")
logger.info("\n" + "-"*70)
logger.info("ORIGINAL MODEL:")
logger.info("-"*70)
model.eval()
with torch.no_grad():
    output_tokens = model.generate(
        prompt_tokens,
        max_length=100,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
logger.info(generated_text)

logger.info("\n" + "-"*70)
logger.info(f"PRUNED MODEL (WANDA, {results_wanda['final_sparsity']:.1%} sparse):")
logger.info("-"*70)

# Apply WANDA pruning results
for name, param in model.named_parameters():
    if name in results_wanda['final_masks']:
        param.data[results_wanda['final_masks'][name]] = 0

model.eval()
with torch.no_grad():
    output_tokens = model.generate(
        prompt_tokens,
        max_length=100,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
logger.info(generated_text)
logger.info("="*70)

# Save the pruned models
logger.info("\n" + "="*70)
logger.info("SAVING PRUNED MODELS")
logger.info("="*70)

# Helper function to format numbers for filenames
def format_for_filename(num):
    """Format number by replacing '.' with '_'"""
    return f"{num:.2f}".replace('.', '_')

# Save most sparse checkpoint
final_sparsity = results_wanda['final_sparsity']
final_perplexity = eval_perplexity(model)
sparsity_str = format_for_filename(final_sparsity * 100)
perplexity_str = format_for_filename(final_perplexity)
output_dir = f"pruned_models/tinystories_wanda_sparse_s{sparsity_str}_p{perplexity_str}"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
logger.info(f"Most sparse checkpoint saved to: {output_dir}")
logger.info(f"  Sparsity: {final_sparsity:.2%}, Perplexity: {final_perplexity:.2f}")

# Save best accuracy checkpoint if different
if 'best_acc_masks' in results_wanda and results_wanda['best_acc_masks'] is not None:
    best_acc_sparsity = results_wanda['best_acc_sparsity']
    best_acc_accuracy = results_wanda['best_acc_accuracy']

    # Check if it's different from the final checkpoint
    if best_acc_sparsity != final_sparsity or abs(best_acc_accuracy - results_wanda['final_accuracy']) > 0.01:
        # Apply best accuracy masks
        model.load_state_dict(saved_state)
        for name, param in model.named_parameters():
            if name in results_wanda['best_acc_masks']:
                mask = results_wanda['best_acc_masks'][name].to(param.device)
                param.data[mask] = 0

        best_perplexity = eval_perplexity(model)
        sparsity_str = format_for_filename(best_acc_sparsity * 100)
        perplexity_str = format_for_filename(best_perplexity)
        output_dir = f"pruned_models/tinystories_wanda_bestacc_s{sparsity_str}_p{perplexity_str}"
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Best accuracy checkpoint saved to: {output_dir}")
        logger.info(f"  Sparsity: {best_acc_sparsity:.2%}, Perplexity: {best_perplexity:.2f}")

        # Restore most sparse checkpoint to model
        model.load_state_dict(saved_state)
        for name, param in model.named_parameters():
            if name in results_wanda['final_masks']:
                mask = results_wanda['final_masks'][name].to(param.device)
                param.data[mask] = 0

logger.info("="*70)
