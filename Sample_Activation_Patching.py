import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the fine-tuned model and tokenizer
model_dir = "./fine_tuned_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)
model.eval()  # Set the model to evaluation mode

# Disable gradient computation for inference
torch.set_grad_enabled(False)

# Define the question and answers
question = "What is the presence of interictal epileptiform discharges associated with?"
correct_answer = "Increased seizure risk in both focal and generalized epilepsy."
corrupted_answer = "Cognitive dysfunction risk in both focal and generalized epilepsy."


# Tokenize the inputs
inputs = tokenizer(question, return_tensors="pt")
correct_inputs = tokenizer(question + " " + correct_answer, return_tensors="pt")
corrupted_inputs = tokenizer(question + " " + corrupted_answer, return_tensors="pt")

# Get model outputs and cache activations
def run_with_cache(model, inputs):
    """Run the model and cache intermediate activations."""
    activations = {}
    def hook_fn(module, input, output):
        activations[module] = output
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):  # Example: hooking linear layers
            hooks.append(module.register_forward_hook(hook_fn))
    with torch.no_grad():
        outputs = model(**inputs)
    for hook in hooks:
        hook.remove()
    return outputs, activations

# Run the model on correct and corrupted inputs
correct_outputs, correct_cache = run_with_cache(model, correct_inputs)
corrupted_outputs, corrupted_cache = run_with_cache(model, corrupted_inputs)

# Define a metric to compare logits (e.g., logit difference for the correct answer)
def get_logit_diff(logits, correct_token_ids, incorrect_token_ids):
    """Calculate the difference in logits between correct and incorrect answers."""
    correct_logits = logits[:, -1, correct_token_ids].mean()
    incorrect_logits = logits[:, -1, incorrect_token_ids].mean()
    return correct_logits - incorrect_logits

# Tokenize the correct and incorrect answer tokens
correct_token_ids = tokenizer.encode("interictal epileptiform discharges", add_special_tokens=False)
incorrect_token_ids = tokenizer.encode("intracranial electrode diagnostics", add_special_tokens=False)

# Calculate logit differences
clean_logit_diff = get_logit_diff(correct_outputs.logits, correct_token_ids, incorrect_token_ids)
corrupted_logit_diff = get_logit_diff(corrupted_outputs.logits, correct_token_ids, incorrect_token_ids)

print(f"Clean logit difference: {clean_logit_diff:.4f}")
print(f"Corrupted logit difference: {corrupted_logit_diff:.4f}")

# Activation patching function
def activation_patching(model, corrupted_inputs, clean_cache, metric_fn):
    """Patch activations from the clean cache into the corrupted run."""
    patched_results = {}
    for module, clean_activation in clean_cache.items():
        def patch_hook(module, input, output):
            return clean_activation
        hook = module.register_forward_hook(patch_hook)
        with torch.no_grad():
            patched_outputs = model(**corrupted_inputs)
        patched_results[str(module)] = metric_fn(patched_outputs.logits)
        hook.remove()
        print(f"Patching layer: {module} | Logit difference: {patched_results[str(module)]:.4f}")
    return patched_results

# Define the metric function for patching
def patching_metric(logits):
    return get_logit_diff(logits, correct_token_ids, incorrect_token_ids)

# Perform activation patching
print("\nStarting activation patching...")
patched_results = activation_patching(model, corrupted_inputs, correct_cache, patching_metric)

# Print the final results
print("\nFinal patching results:")
for layer, logit_diff in patched_results.items():
    print(f"Layer: {layer} | Logit difference: {logit_diff:.4f}")