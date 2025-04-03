import multiprocessing
import torch
import gc
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, EarlyStoppingCallback, DataCollatorForLanguageModeling
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Clear GPU cache and run garbage collection
def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

# Function to chunk long texts into smaller sequences and pad them
def chunk_text(text, tokenizer, max_length=512):
    tokens = tokenizer.encode(text, truncation=False)  # Tokenize without truncation
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]  # Split into chunks

    # Pad chunks to max_length
    for i in range(len(chunks)):
        if len(chunks[i]) < max_length:
            chunks[i] += [tokenizer.pad_token_id] * (max_length - len(chunks[i]))  # Pad with pad_token_id
    return chunks

# Function to tokenize and chunk the dataset
def tokenize_and_chunk(examples, tokenizer):
    chunks = []
    for text in examples["text"]:
        chunks.extend(chunk_text(text, tokenizer))  # Add all chunks for this text
    return {"input_ids": chunks, "labels": chunks}  # Labels are the same as input_ids for causal LM

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

    # Tokenize and chunk datasets
    tokenized_train_dataset = train_dataset.map(
        lambda x: tokenize_and_chunk(x, tokenizer),  # Pass tokenizer here
        batched=True,
        num_proc=4,
        remove_columns=["text"],
    )
    tokenized_val_dataset = val_dataset.map(
        lambda x: tokenize_and_chunk(x, tokenizer),  # Pass tokenizer here
        batched=True,
        num_proc=4,
        remove_columns=["text"],
    )

    # Use a data collator for dynamic padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Set to False for causal language modeling
    )

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./fine_tuned_gpt2",
        overwrite_output_dir=True,
        num_train_epochs=10,  # Reduced epochs
        per_device_train_batch_size=8,
        per_device_eval_batch_size=2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,
        fp16=False,  # Disabled mixed precision
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        learning_rate=2e-5,  # Reduced learning rate
        gradient_accumulation_steps=4,
        warmup_steps=500,
        weight_decay=0.1,  # Increased weight decay
    )

    # Define Trainer with dynamic padding
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,  # Add the data collator
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
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