"""
Test GradProbe on Mistral-7B-v0.3 language model.

This script prunes Mistral-7B using WANDA with gradient-based filtering.
Note: This requires significant RAM and is slow on CPU. GPU recommended.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import math
import copy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gradprobe import GradProbe, WANDAPruning

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("transformers library not found. Please install it:")
    print("pip install transformers")
    sys.exit(1)

# Configuration
MODEL_NAME = "mistralai/Mistral-7B-v0.3"
DEVICE = "auto"  # Will auto-detect CUDA
SEQ_LENGTH = 128  # Reduced from 512 to save memory
NUM_BATCHES_GRADIENT = 10  # Reduced from 50 to save memory and time
NUM_BATCHES_WANDA = 5  # Reduced from 20 to save memory
LOW_MEMORY_MODE = True  # CRITICAL for large models - disables gradient caching

# Calibration text - using more diverse content for larger model
CALIBRATION_TEXT = """The quick brown fox jumps over the lazy dog. This pangram contains every letter of the English alphabet at least once. It has been used for decades to test typewriters and computer keyboards.

In the realm of artificial intelligence, language models have revolutionized how we interact with computers. These models learn patterns from vast amounts of text data, enabling them to generate coherent and contextually relevant responses. The transformer architecture, introduced in 2017, marked a significant breakthrough in natural language processing.

Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models. These enable computer systems to improve their performance on specific tasks through experience, without being explicitly programmed. Deep learning, a subfield of machine learning, uses artificial neural networks with multiple layers to learn hierarchical representations of data.

The history of computing dates back to ancient times when humans first created tools for calculation. The abacus, one of the earliest computing devices, was invented thousands of years ago. In the 19th century, Charles Babbage designed the Analytical Engine, a mechanical general-purpose computer. Although never completed in his lifetime, Babbage's designs laid the groundwork for modern computers.

Climate change represents one of the most pressing challenges facing humanity in the 21st century. Rising global temperatures, melting ice caps, and increasingly frequent extreme weather events are clear indicators of our changing climate. Scientists worldwide are working to understand these changes and develop solutions to mitigate their impact.

The human brain is an extraordinarily complex organ, containing approximately 86 billion neurons. These neurons communicate through trillions of synaptic connections, forming intricate networks that enable thought, memory, emotion, and consciousness. Neuroscientists continue to unravel the mysteries of brain function, seeking to understand how this remarkable organ gives rise to the human mind.

Throughout history, explorers have ventured into the unknown, driven by curiosity and the desire to discover new lands and cultures. From ancient Polynesian navigators crossing vast oceans to modern astronauts venturing into space, the spirit of exploration has been a defining characteristic of human civilization.

Mathematics is often called the language of the universe. From the elegant simplicity of Euclidean geometry to the abstract complexities of modern algebra and topology, mathematics provides tools for understanding patterns, structures, and relationships in both the natural and abstract worlds. Many scientific breakthroughs have been made possible by mathematical insights.

The development of writing systems was a crucial milestone in human history. Early forms of writing, such as cuneiform in Mesopotamia and hieroglyphics in ancient Egypt, enabled the recording of information, laws, and stories for future generations. The invention of the printing press in the 15th century democratized access to written knowledge, sparking intellectual revolutions.

Biodiversity refers to the variety of life on Earth, encompassing millions of species of plants, animals, fungi, and microorganisms. Each species plays a unique role in its ecosystem, contributing to the complex web of life that sustains our planet. Conservation efforts aim to protect this diversity, recognizing that the loss of species can have far-reaching consequences."""

print("="*70)
print("GRADPROBE - MISTRAL-7B-v0.3 PRUNING TEST")
print("="*70)
print(f"Model: {MODEL_NAME}")
print(f"Device: {DEVICE}")
print(f"Sequence length: {SEQ_LENGTH}")
print("="*70)

# Memory monitoring helper
def print_gpu_memory(label):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[{label}] GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        torch.cuda.empty_cache()  # Clear cache after reporting
        after_clear = torch.cuda.memory_allocated() / 1024**3
        print(f"[{label}] After cache clear: {after_clear:.2f}GB")

# Load model
print(f"\nLoading Mistral-7B-v0.3...")
print("This may take a few minutes...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,  # Use float16 for memory efficiency
    low_cpu_mem_usage=True,
    device_map="auto" if torch.cuda.is_available() else None
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Enable gradient checkpointing to save memory
# This trades compute for memory by not storing intermediate activations
if hasattr(model, 'gradient_checkpointing_enable'):
    model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled")

print(f"Model loaded successfully")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Model dtype: {next(model.parameters()).dtype}")
print_gpu_memory("After model load")

# Prepare calibration data
print("\nPreparing calibration data...")
tokens = tokenizer.encode(CALIBRATION_TEXT, return_tensors='pt')
print(f"Total tokens: {tokens.shape[1]}")
print_gpu_memory("After tokenization")

# Create dataset with sliding windows
stride = SEQ_LENGTH // 2  # 50% overlap
input_sequences = []
target_sequences = []

for i in range(0, tokens.shape[1] - SEQ_LENGTH - 1, stride):
    input_seq = tokens[:, i:i+SEQ_LENGTH]
    target_seq = tokens[:, i+1:i+SEQ_LENGTH+1]
    if input_seq.shape[1] == SEQ_LENGTH and target_seq.shape[1] == SEQ_LENGTH:
        input_sequences.append(input_seq)
        target_sequences.append(target_seq)

print(f"Created {len(input_sequences)} sequences of length {SEQ_LENGTH}")

# Create dataloaders
all_inputs = torch.cat(input_sequences, dim=0)
all_targets = torch.cat(target_sequences, dim=0)
dataset = TensorDataset(all_inputs, all_targets)
dataloader_pruning = DataLoader(dataset, batch_size=1, shuffle=False)

# Store for evaluation
eval_data = list(zip(input_sequences, target_sequences))


# Evaluation function - compute perplexity
def eval_perplexity(m):
    m.eval()
    total_loss = 0
    total_tokens = 0
    device = next(m.parameters()).device

    with torch.no_grad():
        # Use first few sequences for eval
        for input_seq, target_seq in eval_data[:min(5, len(eval_data))]:
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


# Loss function for pruning
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


# Evaluation function returning negative perplexity
def eval_fn_adjusted(m):
    perplexity = eval_perplexity(m)
    return -perplexity


# Measure initial perplexity
print(f"\nMeasuring initial perplexity...")
initial_perplexity = eval_perplexity(model)
print(f"Initial perplexity: {initial_perplexity:.2f}")
print_gpu_memory("After initial perplexity")

# Save model state (do this on CPU to avoid GPU memory issues)
print("\nSaving model state...")
saved_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
print_gpu_memory("After saving model state")

# Create WANDA strategy
print(f"\nInitializing WANDA pruning strategy...")
print(f"Collecting activations from {NUM_BATCHES_WANDA} batches...")
print_gpu_memory("Before WANDA activation collection")
wanda_strategy = WANDAPruning(dataloader=dataloader_pruning, num_batches=NUM_BATCHES_WANDA)
print_gpu_memory("After WANDA strategy creation")

# Create pruner with low_memory_mode
print(f"\nInitializing GradProbe pruner...")
pruner = GradProbe(model, wanda_strategy, device=DEVICE, low_memory_mode=LOW_MEMORY_MODE, use_fp16=True,use_gradient_checkpointing=True)
print_gpu_memory("After pruner creation")

# Run pruning
print("\n" + "="*70)
print("PRUNING MISTRAL-7B WITH WANDA + GRADIENT FILTERING")
print("="*70)
print("Configuration:")
print(f"  Initial sparsity: 10%")
print(f"  Sparsity step: 10%")
print(f"  Max perplexity increase: 5.0")
print(f"  Gradient threshold: 0.5 (conservative for large model)")
print(f"  Layerwise: True")
print(f"  Layer order: size (largest first)")
print(f"  Low memory mode: {LOW_MEMORY_MODE}")
if LOW_MEMORY_MODE:
    print(f"  Threshold tuning: Disabled (low memory mode)")
    print(f"  Experimental two-step tuning: Disabled (low memory mode)")
else:
    print(f"  Threshold tuning: Enabled")
    print(f"  Experimental two-step tuning: Enabled")
print("="*70)

results = pruner.iterative_prune(
    dataloader=dataloader_pruning,
    loss_fn=loss_fn,
    eval_fn=eval_fn_adjusted,
    initial_sparsity=0.1,
    sparsity_step=0.1,
    max_accuracy_drop=5.0,  # Allow larger perplexity increase for initial test
    num_batches=NUM_BATCHES_GRADIENT,
    gradient_threshold=0.5,  # More conservative for large model
    layerwise=True,
    layer_order="size",  # Prune largest layers first
    verbose=True,
    compare_baseline=True,
    tune_threshold_on_fail=True,
    experimental_tune_both_steps=True
)

# Print final results
final_perplexity = eval_perplexity(model)
print("\n" + "="*70)
print("FINAL RESULTS - MISTRAL-7B PRUNING")
print("="*70)
print(f"Initial perplexity: {initial_perplexity:.2f}")
print(f"Final perplexity: {final_perplexity:.2f}")
print(f"Perplexity increase: {final_perplexity - initial_perplexity:.2f}")
print(f"Final sparsity: {results['final_sparsity']:.2%}")
total_params = sum(p.numel() for p in model.parameters())
pruned_params = int(results['final_sparsity'] * total_params)
print(f"Pruned parameters: {pruned_params:,} / {total_params:,}")
print("="*70)

# Generate text comparison
print("\n" + "="*70)
print("TEXT GENERATION COMPARISON")
print("="*70)

model.load_state_dict(saved_state)
device = next(model.parameters()).device

prompt_text = "The future of artificial intelligence is"
prompt_tokens = tokenizer.encode(prompt_text, return_tensors='pt').to(device)

print(f"\nPrompt: \"{prompt_text}\"")
print("\n" + "-"*70)
print("ORIGINAL MODEL:")
print("-"*70)
model.eval()
with torch.no_grad():
    output_tokens = model.generate(
        prompt_tokens,
        max_length=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print(generated_text)

print("\n" + "-"*70)
print(f"PRUNED MODEL ({results['final_sparsity']:.1%} sparse):")
print("-"*70)

# Apply pruning
for name, param in model.named_parameters():
    if name in results['final_masks']:
        param.data[results['final_masks'][name]] = 0

model.eval()
with torch.no_grad():
    output_tokens = model.generate(
        prompt_tokens,
        max_length=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print(generated_text)
print("="*70)

print("\n" + "="*70)
print("MISTRAL-7B PRUNING TEST COMPLETED")
print("="*70)
