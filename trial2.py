import json
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

#Example of using the saved model for inference
def generate_text(input_text):
    model = T5ForConditionalGeneration.from_pretrained('./fine_tuned_model')
    tokenizer = T5Tokenizer.from_pretrained('./fine_tuned_model')

    # Tokenize the input
    task_prefix = "encode the following into logical unique codes: "
    inputs = tokenizer(task_prefix + input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate text
    outputs = model.generate(inputs['input_ids'], max_length=128, num_return_sequences=1, temperature=1.0, top_k=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test the inference function
print(generate_text("  oring at Battery room"))