import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    BertTokenizer, BertForSequenceClassification, 
     get_linear_schedule_with_warmup, BertConfig,
     Trainer, TrainingArguments
)


from datasets import load_dataset
import evaluate

from tqdm import tqdm
import os

from src.loss.distillation import DistillationLoss  # Custom loss function
from src.model.BERT import BERT as StudentBERT  # My BERT model
from src.model.myTrasformers import ExtractQK

import time
time_stamp = time.strftime("%m%d%H%M%S")

# ==========================
# Hyperparameters
# ==========================
TASK_NAME = "sst2"  # GLUE Task (e.g., 'sst2', 'mrpc', etc.)
BATCH_SIZE = 32  # Can be adjusted based on GPU capacity
LEARNING_RATE = 1e-5 # Peak learning rate for the 6-layer model in the paper
NUM_EPOCHS = 100  # Approximate number of epochs based on the paper
TEMPERATURE = 2.0  # Temperature for knowledge distillation
SAVE_EVERY = 10  # Save the model every 20 epochs (can adjust based on preference)
MODEL_SAVE_PATH = "./saved_models"  # Path to save the model
WRITER_PATH = "./runs"  # Path to save TensorBoard logs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Automatically use GPU if available


# ==========================
# Custom Dataset
# ==========================
class GLUEDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_ids = []
        self.attention_masks = []
        self.labels = []

        for item in dataset:
            # Tokenize and encode
            encoding = self.tokenizer(
                item['sentence'], 
                truncation=True, 
                padding='max_length', 
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            self.input_ids.append(encoding['input_ids'].squeeze())
            self.attention_masks.append(encoding['attention_mask'].squeeze())
            self.labels.append(torch.tensor(item['label'], dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }

# ==========================
# Data Preparation
# ==========================
def load_glue_data(task):
    """
    Load the GLUE benchmark dataset for the specified task.
    """
    dataset = load_dataset("glue", task)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Convert to custom dataset
    train_dataset = GLUEDataset(dataset['train'], tokenizer)
    val_dataset = GLUEDataset(dataset['validation'], tokenizer)

    return train_dataset, val_dataset, tokenizer

# ==========================
# Evaluation Function
# ==========================
def validate(model, dataloader, metric, device):
    """
    Evaluate the model on the validation set.
    """
    model.eval()
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            # Default to return_dict=False to get logits directly
            outputs = model(
                batch['input_ids'], 
                attention_mask=batch['attention_mask'].unsqueeze(1).unsqueeze(2),
                return_dict=False  # Explicitly set to get logits
            )
            
            # outputs will be logits if return_dict=False
            predictions = torch.argmax(outputs, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
    return metric.compute()

# ==========================
# Extract Self-Attention Features
# ==========================
def extract_self_attention_features(model, input_ids, attention_mask, return_attention=False):
    """
    Extract self-attention weights and values from the model.
    
    Args:
        model (nn.Module): BERT model (either teacher or student).
        input_ids (torch.Tensor): Input tensor with token IDs.
        attention_mask (torch.Tensor): Attention mask tensor.
        return_attention (bool): Whether to return attention weights.

    Returns:
        dict: Dictionary containing queries, keys, values, and optional attention.
    """
    model.eval()  # Ensure the model is in evaluation mode

    # Check if it's a HuggingFace model or custom model
    if hasattr(model, 'extract_attention_features'):
        # For custom StudentBERT
        outputs = model.extract_attention_features(
            input_ids, 
            attention_mask=attention_mask.unsqueeze(1).unsqueeze(2)
        )

        attention_features = outputs.attentions[-1]
        values = outputs.hidden_states[-1]


        return {
            'attention': [attention_features],  # Mimic list format of custom model
            'values': [values]
        }   
    else:
        with torch.no_grad():

            # For HuggingFace BertForSequenceClassification
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True
            )

            extractor = ExtractQK(model)
            attention_weights = outputs.attentions[-1]
            
            # Extract last layer's attention and hidden states
            values = outputs.hidden_states[-1]  # Last layer's hidden states as values
            
            return {
                'attention': [attention_weights],  # Mimic list format of custom model
                'values': [values]
            }

# ==========================
# Import pre-trained BERT model
# ==========================

# Load fine-tuned model and tokenizer
model = BertForSequenceClassification.from_pretrained('./saved_models/fine_tuned_bert')
tokenizer = BertTokenizer.from_pretrained('./saved_models/fine_tuned_bert')

# Load dataset
dataset = load_dataset('glue', 'sst2')

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Convert to PyTorch format
tokenized_datasets = tokenized_datasets.with_format("torch", columns=["input_ids", "attention_mask", "label"])

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # Directory to save results
    per_device_eval_batch_size=32,  # Batch size for evaluation
    logging_dir='./logs',           # Directory for logs
    eval_strategy="epoch",    # Evaluate every epoch
)

from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {'accuracy': accuracy_score(labels, predictions)}

# Set up Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_datasets['validation'],
    compute_metrics=compute_metrics
)

# Evaluate the model
results = trainer.evaluate()
print("Evaluation Results from Teacher Model:")
print(results)

# ==========================
# Main Training Loop
# ==========================
def train():
    # Ensure model save path exists
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    # Load configuration and initialize the modified BERT
    # config = BertConfig.from_pretrained("bert-base-uncased", num_labels=2)
    # teacher_model = BertForSequenceClassification.from_pretrained(
    #     "bert-base-uncased",
    #     config=config,
    #     attn_implementation="eager")
    # teacher_model.to(DEVICE)

    # Load fine-tuned model and tokenizer
    teacher_model = model

    student_model = StudentBERT(
        vocab_size=30522,  # BERT uncased vocab size 
        num_labels=2
    ).to(DEVICE)

    # Freeze the teacher model
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Load data
    train_dataset, val_dataset, tokenizer = load_glue_data(TASK_NAME)
    
    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize metric
    metric = evaluate.load("glue", TASK_NAME)

    # Define loss function
    distillation_loss = DistillationLoss(temperature=TEMPERATURE).to(DEVICE)

    # Optimizer and Learning Rate Scheduler
    optimizer = AdamW(student_model.parameters(), lr=LEARNING_RATE, 
                      betas=(0.9, 0.999), weight_decay=0.01)
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=2000, 
        num_training_steps=len(train_dataloader) * NUM_EPOCHS
    )

    print("Number of training steps:", len(train_dataloader) * NUM_EPOCHS)

    # Accumulation steps
    steps = 0


    # Criterion for classification loss
    classification_criterion = nn.CrossEntropyLoss()

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir="runs/knowledge_distillation_" + time_stamp)

   # Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        student_model.train()
        total_loss = 0.0 
        i = 0  # Initialize steps

        # Set up tqdm for progress tracking
        with tqdm(train_dataloader, desc=f"Epoch {epoch}/{NUM_EPOCHS}") as t:
            for batch in t:
                # Move batch to device
                batch = {k: v.to(DEVICE) for k, v in batch.items()}

                # Teacher forward pass (no gradients required)
                with torch.no_grad():
                    teacher_outputs = teacher_model(
                        input_ids=batch['input_ids'], 
                        attention_mask=batch['attention_mask']
                    )

                # Student forward pass
                student_outputs = student_model(
                    batch['input_ids'], 
                    attention_mask=batch['attention_mask'].unsqueeze(1).unsqueeze(2)
                )

                # Extract attention features
                teacher_attention_features = extract_self_attention_features(
                    teacher_model, 
                    batch['input_ids'], 
                    batch['attention_mask']
                )
                student_attention_features = extract_self_attention_features(
                    student_model, 
                    batch['input_ids'], 
                    batch['attention_mask']
                )

                # Compute distillation loss
                knowledge_loss = distillation_loss(
                    teacher_A=teacher_attention_features['attention'][0],
                    teacher_values=teacher_attention_features['values'][0],
                    student_A=student_attention_features['attention'][0],
                    student_values=student_attention_features['values'][0]
                )

                # Compute classification loss
                classification_loss = classification_criterion(
                    student_outputs, batch['labels']
                )

                # Combined loss
                total_batch_loss = knowledge_loss + classification_loss # Combine both losses


                # Backward pass (accumulate gradients)
                total_batch_loss.backward()


    
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


                # Update total loss tracking
                total_loss += total_batch_loss.item()

                # Log batch loss to TensorBoard
                writer.add_scalar("Loss/Batch", total_batch_loss.item(), epoch * len(train_dataloader) + i)
                i += 1  # Increment iteration counter
                steps += 1  # Increment global step counter

                # Update progress bar
                t.set_postfix({'loss': total_batch_loss.item(), 'step': steps})

        # End of epoch, print average loss
        avg_loss = total_loss / len(train_dataloader)
        writer.add_scalar("Loss/Epoch", avg_loss, epoch)
        print(f"Epoch {epoch}/{NUM_EPOCHS} - Average Loss: {avg_loss:.4f}")
        writer.add_scalar("Learning Rate", lr_scheduler.get_last_lr()[0], epoch)
        writer.add_scalar("Gradient Norm", torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0), epoch)
        writer.add_scalar("Knowledge Loss", knowledge_loss.item(), epoch)
        writer.add_scalar("Classification Loss", classification_loss.item(), epoch)

        



        # Save model periodically
        if epoch % SAVE_EVERY == 0:
            model_path = os.path.join(MODEL_SAVE_PATH, f"student_epoch_{epoch}_{time_stamp}.pt")
            torch.save(student_model.state_dict(), model_path)
            print(f"Model saved at {model_path}")

        # Validate accuracy on validation
        val_results = validate(student_model, val_dataloader, metric, DEVICE)
        writer.add_scalar("Accuracy/Validation", val_results['accuracy'], epoch)
        print(f"Validation Results (Epoch {epoch}): {val_results}")

    print("Training complete.")
    writer.close()

if __name__ == "__main__":
    train()