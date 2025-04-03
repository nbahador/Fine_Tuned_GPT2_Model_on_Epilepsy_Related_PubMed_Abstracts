import multiprocessing
import torch
import gc
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Clear GPU cache and run garbage collection
def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

# Function to run training in a subprocess
def train_in_subprocess():
    # Path to your Excel file
    file_path = r"replace_with_your_path_to\PubMed_resultsx.xlsx"

    # Load the Excel file
    df = pd.read_excel(file_path)

    # Extract the 'Abstract' column
    abstracts = df['Abstract'].dropna().tolist()  # Drop NaN values and convert to list

    # Split data into training and validation sets
    train_texts, val_texts = train_test_split(abstracts, test_size=0.2, random_state=42)

    # Create Hugging Face datasets
    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset = Dataset.from_dict({"text": val_texts})

    # Load GPT-2 tokenizer and model
    model_name = "gpt2"  # You can use "gpt2-medium", "gpt2-large", or "gpt2-xl" for larger models
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Set padding token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize datasets and add labels
    def tokenize_function(examples):
        # Tokenize the input text
        tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
        # For causal language modeling, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"]
        return tokenized

    # Tokenize datasets
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=4)
    tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True, num_proc=4)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./fine_tuned_gpt2",
        overwrite_output_dir=True,
        num_train_epochs=30,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=2,
        evaluation_strategy="epoch",  # Evaluate every epoch
        save_strategy="epoch",  # Save every epoch
        save_total_limit=2,  # Keep only the best 2 models
        logging_dir="./logs",
        logging_steps=100,  # Log every 100 steps
        fp16=True,  # Use mixed precision
        load_best_model_at_end=True,  # Load the best model at the end
        metric_for_best_model="eval_loss",  # Use evaluation loss to determine the best model
        greater_is_better=False,  # Lower evaluation loss is better
        report_to="none",  # Disable W&B
        learning_rate=5e-5,  # Use a smaller learning rate
        gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps
        warmup_steps=500,  # Warmup the learning rate
        weight_decay=0.01,  # Add weight decay for regularization
    )

    # Define Trainer with EarlyStoppingCallback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Early stopping with patience of 3 epochs
    )

    # Clear memory before training
    clear_memory()

    # Train the model
    train_result = trainer.train()

    # Print training and evaluation losses
    print("Training Loss:", train_result.metrics["train_loss"])
    eval_result = trainer.evaluate()
    print("Evaluation Loss:", eval_result["eval_loss"])

    # Save the final model and tokenizer
    trainer.save_model("./fine_tuned_gpt2")
    tokenizer.save_pretrained("./fine_tuned_gpt2")

    print("Model and tokenizer saved to ./fine_tuned_gpt2")

    # Clear memory after training
    clear_memory()

# Main function
def main():
    # Clear memory before starting
    clear_memory()

    # Run training in a subprocess
    process = multiprocessing.Process(target=train_in_subprocess)
    process.start()
    process.join()  # Wait for the subprocess to finish

    # Clear memory after training
    clear_memory()

if __name__ == "__main__":
    main()