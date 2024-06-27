import json
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

# Function to load JSONL file
def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line.strip()) for line in file]
    return data

# Load JSONL data
data = load_jsonl('data.jsonl')

# Prepare dataset
dataset = Dataset.from_dict({'input_text': [item['Concat'] for item in data],
                             'target_text': [item['MiddlePart'] for item in data]})

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-small")
model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-small")

# Tokenize function
def tokenize_function(examples):
    task_prefix = 'encode the following into logical unique codes: '
    inputs = [task_prefix + inp for inp in examples['input_text']]
    targets = examples['target_text']
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length')
    
    labels["input_ids"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels_seq]
        for labels_seq in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=4,   # batch size per device during training
    save_steps=1000,                 # number of updates steps before checkpoint saves
    save_total_limit=2,              # limit the total amount of checkpoints
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,               # log every x updates steps
    overwrite_output_dir=True,       # overwrite the content of the output directory
)

# Initialize Trainer
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=tokenized_dataset,     # training dataset
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')


