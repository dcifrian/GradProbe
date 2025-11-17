"""
General-purpose script for pruning HuggingFace language models with GradProbe.

This script can prune any HuggingFace causal LM using WANDA or Magnitude pruning
with gradient-based filtering.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import math
import json
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gradprobe import GradProbe, MagnitudePruning, WANDAPruning

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("transformers library not found. Please install it:")
    print("pip install transformers")
    sys.exit(1)


DEFAULT_CALIBRATION_TEXT = """Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the park. She ran to get it and kicked it very hard. The ball flew high into the sky and landed in a tree. Lily was sad because she couldn't reach it. Then, a kind boy came and helped her get the ball down. Lily was happy again and they played together all day.

Once there was a brave knight named Tom. He had a shiny sword and a big horse. One day, Tom went on an adventure to find a dragon. He rode his horse through the dark forest and over the tall mountains. When he found the dragon, it was sleeping in a cave. Tom was very quiet and sneaked past the dragon. He found a treasure chest full of gold coins. Tom took the treasure and rode back home to share it with his family.

There was a happy bunny who lived in the woods. The bunny liked to hop and play with his friends. One sunny morning, the bunny found a big carrot in the garden. He was so excited! He called all his friends to come and see. Together, they shared the carrot and had a wonderful picnic. The bunny and his friends played games until the sun went down. Then they all went home to sleep.

A little boy named Max loved to build things. He had many blocks of different colors. One day, Max decided to build a tall tower. He stacked the blocks very carefully, one on top of another. The tower grew taller and taller until it was as high as Max's head. Max was very proud of his tower. He showed it to his mom and dad, and they clapped their hands. That night, Max dreamed about building even bigger towers.

In a peaceful village, there lived a wise old owl. The owl sat in a tall oak tree and watched over the village. One night, a little mouse got lost in the dark forest. The mouse was scared and didn't know how to get home. The wise owl heard the mouse crying and flew down to help. The owl guided the mouse through the forest with its bright eyes. Soon, the mouse was safely home with its family. The mouse thanked the owl and they became good friends.

A curious cat named Whiskers loved to explore. One day, Whiskers found a mysterious box in the attic. The box was old and dusty. Whiskers pawed at it until it opened. Inside was a beautiful golden bell. When Whiskers touched it, the bell made a soft, magical sound. From that day on, whenever Whiskers rang the bell, wonderful things would happen. Birds would come and sing, flowers would bloom, and everyone in the house would smile."""


def prepare_calibration_data(text, tokenizer, seq_length=128, stride=64, device='cpu'):
    """Prepare calibration data from text."""
    print("\nPreparing calibration data...")
    tokens = tokenizer.encode(text, return_tensors='pt')
    print(f"Total tokens: {tokens.shape[1]}")

    # Create dataset with sliding windows
    input_sequences = []
    target_sequences = []

    for i in range(0, tokens.shape[1] - seq_length - 1, stride):
        input_seq = tokens[:, i:i+seq_length]
        target_seq = tokens[:, i+1:i+seq_length+1]  # Shifted by 1
        if input_seq.shape[1] == seq_length and target_seq.shape[1] == seq_length:
            input_sequences.append(input_seq)
            target_sequences.append(target_seq)

    print(f"Created {len(input_sequences)} sequences of length {seq_length}")

    # Create dataloaders
    all_inputs = torch.cat(input_sequences, dim=0)
    all_targets = torch.cat(target_sequences, dim=0)
    dataset = TensorDataset(all_inputs, all_targets)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Store for evaluation
    eval_data = list(zip(input_sequences, target_sequences))

    return dataloader, eval_data


def eval_perplexity(model, eval_data, max_sequences=10):
    """Evaluate model perplexity."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    device = next(model.parameters()).device

    with torch.no_grad():
        for input_seq, target_seq in eval_data[:min(max_sequences, len(eval_data))]:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            outputs = model(input_seq)
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


def loss_fn(outputs, targets):
    """Loss function for pruning."""
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
    else:
        logits = outputs

    loss = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1)
    )
    return loss


def eval_fn_adjusted(model, eval_data):
    """Evaluation function returning negative perplexity for iterative_prune."""
    perplexity = eval_perplexity(model, eval_data)
    # Return negative perplexity so higher perplexity = lower score
    return -perplexity


def generate_text_comparison(model, tokenizer, prompt, original_state, pruned_masks,
                            max_length=100, temperature=0.8, top_p=0.9):
    """Generate text with original and pruned model for comparison."""
    device = next(model.parameters()).device
    prompt_tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)

    print("\n" + "="*70)
    print("TEXT GENERATION COMPARISON")
    print("="*70)
    print(f"\nPrompt: \"{prompt}\"")

    # Original model
    print("\n" + "-"*70)
    print("ORIGINAL MODEL:")
    print("-"*70)
    model.load_state_dict(original_state)
    model.eval()
    with torch.no_grad():
        output_tokens = model.generate(
            prompt_tokens,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print(generated_text)

    # Pruned model
    print("\n" + "-"*70)
    sparsity = sum(m.sum().item() for m in pruned_masks.values()) / sum(m.numel() for m in pruned_masks.values())
    print(f"PRUNED MODEL ({sparsity:.1%} sparse):")
    print("-"*70)
    for name, param in model.named_parameters():
        if name in pruned_masks:
            param.data[pruned_masks[name]] = 0

    model.eval()
    with torch.no_grad():
        output_tokens = model.generate(
            prompt_tokens,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print(generated_text)
    print("="*70)


def save_pruned_model(model, pruned_masks, output_path, tokenizer=None):
    """Save pruned model to disk."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Apply masks
    for name, param in model.named_parameters():
        if name in pruned_masks:
            param.data[pruned_masks[name]] = 0

    # Save model
    model.save_pretrained(output_path)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_path)

    # Save pruning info
    total_params = sum(p.numel() for p in model.parameters())
    pruned_params = sum(m.sum().item() for m in pruned_masks.values())
    sparsity = pruned_params / total_params

    info = {
        'total_parameters': total_params,
        'pruned_parameters': int(pruned_params),
        'sparsity': float(sparsity),
    }

    with open(output_path / 'pruning_info.json', 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\nPruned model saved to: {output_path}")
    print(f"Sparsity: {sparsity:.2%} ({pruned_params:,} / {total_params:,} parameters)")


def main():
    parser = argparse.ArgumentParser(description='Prune HuggingFace language models with GradProbe')

    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                       help='HuggingFace model name or path')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use (auto, cpu, cuda)')

    # Pruning strategy
    parser.add_argument('--strategy', type=str, default='wanda',
                       choices=['wanda', 'magnitude'],
                       help='Pruning strategy (wanda or magnitude)')

    # Pruning parameters
    parser.add_argument('--initial-sparsity', type=float, default=0.1,
                       help='Initial sparsity level (default: 0.1)')
    parser.add_argument('--sparsity-step', type=float, default=0.1,
                       help='Sparsity increase per iteration (default: 0.1)')
    parser.add_argument('--max-perplexity-increase', type=float, default=3.0,
                       help='Stop if perplexity increases by this much (default: 3.0)')
    parser.add_argument('--gradient-threshold', type=float, default=1.0,
                       help='Gradient threshold for filtering (default: 1.0)')

    # Advanced options
    parser.add_argument('--layerwise', action='store_true',
                       help='Use layer-by-layer pruning')
    parser.add_argument('--layer-order', type=str, default='reverse',
                       choices=['reverse', 'size', 'forward'],
                       help='Layer ordering for layerwise pruning (default: reverse)')
    parser.add_argument('--tune-threshold', action='store_true',
                       help='Enable threshold tuning when accuracy target is missed')
    parser.add_argument('--experimental-tune-both-steps', action='store_true',
                       help='EXPERIMENTAL: Re-prune both steps during threshold tuning')
    parser.add_argument('--num-batches', type=int, default=100,
                       help='Number of batches for gradient computation (default: 100)')
    parser.add_argument('--low-memory-mode', action='store_true',
                       help='Enable low memory mode for large models (disables gradient caching, '
                            'recommended for models > 1B parameters)')

    # Calibration data
    parser.add_argument('--calibration-text', type=str, default=None,
                       help='Path to calibration text file (default: use built-in text)')
    parser.add_argument('--seq-length', type=int, default=128,
                       help='Sequence length for calibration data (default: 128)')

    # Output
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for pruned model (optional)')
    parser.add_argument('--generate-text', action='store_true',
                       help='Generate comparison text samples')
    parser.add_argument('--generation-prompt', type=str,
                       default='Once upon a time, there was',
                       help='Prompt for text generation')

    # Comparison
    parser.add_argument('--compare-baseline', action='store_true',
                       help='Compare with strategy-only pruning (no gradient filtering)')

    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model}")
    print(f"Device: {args.device}")

    # Determine if we should use memory-efficient loading
    use_efficient_loading = args.low_memory_mode or args.device == 'auto'

    if use_efficient_loading:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto" if args.device == 'auto' else None
        )
        # Enable gradient checkpointing for large models
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model dtype: {next(model.parameters()).dtype}")

    # Prepare calibration data
    if args.calibration_text:
        with open(args.calibration_text, 'r') as f:
            calibration_text = f.read()
    else:
        print("Using default calibration text")
        calibration_text = DEFAULT_CALIBRATION_TEXT

    dataloader, eval_data = prepare_calibration_data(
        calibration_text,
        tokenizer,
        seq_length=args.seq_length
    )

    # Measure initial perplexity
    initial_perplexity = eval_perplexity(model, eval_data)
    print(f"\nInitial perplexity: {initial_perplexity:.2f}")

    # Save original state (on CPU to avoid GPU memory issues with large models)
    import copy
    if use_efficient_loading:
        saved_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        saved_state = copy.deepcopy(model.state_dict())

    # Create evaluation function
    eval_fn = lambda m: eval_fn_adjusted(m, eval_data)

    # Create pruning strategy
    if args.strategy == 'wanda':
        print("\nUsing WANDA pruning strategy")
        strategy = WANDAPruning(dataloader=dataloader, num_batches=min(50, len(dataloader)))
    else:
        print("\nUsing Magnitude pruning strategy")
        strategy = MagnitudePruning()

    # Create pruner
    pruner = GradProbe(model, strategy, device=args.device, low_memory_mode=args.low_memory_mode)

    # Run pruning
    print("\n" + "="*70)
    print(f"PRUNING WITH {args.strategy.upper()}")
    print("="*70)

    results = pruner.iterative_prune(
        dataloader=dataloader,
        loss_fn=loss_fn,
        eval_fn=eval_fn,
        initial_sparsity=args.initial_sparsity,
        sparsity_step=args.sparsity_step,
        max_accuracy_drop=args.max_perplexity_increase,
        num_batches=args.num_batches,
        gradient_threshold=args.gradient_threshold,
        layerwise=args.layerwise,
        layer_order=args.layer_order,
        verbose=True,
        compare_baseline=args.compare_baseline,
        tune_threshold_on_fail=args.tune_threshold,
        experimental_tune_both_steps=args.experimental_tune_both_steps
    )

    # Print final results
    final_perplexity = eval_perplexity(model, eval_data)
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Strategy: {args.strategy.upper()}")
    print(f"Initial perplexity: {initial_perplexity:.2f}")
    print(f"Final perplexity: {final_perplexity:.2f}")
    print(f"Perplexity increase: {final_perplexity - initial_perplexity:.2f}")
    print(f"Final sparsity: {results['final_sparsity']:.2%}")
    print("="*70)

    # Generate text comparison
    if args.generate_text:
        generate_text_comparison(
            model, tokenizer, args.generation_prompt,
            saved_state, results['final_masks']
        )

    # Save pruned model
    if args.output:
        save_pruned_model(model, results['final_masks'], args.output, tokenizer)

    return results


if __name__ == '__main__':
    main()
