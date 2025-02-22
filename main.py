import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback, pipeline
import torch
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from scipy.special import softmax
import json, os, random
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

# Step 1: Load and prepare the dataset
file_path = 'symbipredict_2022.csv'  # Replace with your file path if different
data = pd.read_csv(file_path)

# Fill missing values with 0 (or you could remove rows with missing critical data)
data.fillna(0, inplace=True)

# Convert all symptom labels to lowercase (or any other necessary standardization)
data.columns = [col.replace('_', ' ').lower().strip() for col in data.columns]

# Load the symptom severity data
severity_file_path = 'Symptom-severity.csv'
severity_data = pd.read_csv(severity_file_path)

# Create a dictionary mapping each symptom to its severity weight
severity_dict = dict(zip(severity_data['Symptom'].str.lower(), severity_data['weight']))

# Define a default severity (mean of known severity scores)
mean_severity = severity_data['weight'].mean()

# Apply severity weight to the symptoms in the dataset
def apply_severity(row):
    for symptom in severity_dict.keys():
        if symptom in row.index and row[symptom] == 1:
            row[symptom] = severity_dict.get(symptom, mean_severity)  # Use severity if available, otherwise mean
    return row

# Apply the severity mapping to the dataset
symptom_columns = [col for col in data.columns if col != 'prognosis' and col != 'symptoms_text']
data[symptom_columns] = data[symptom_columns].apply(apply_severity, axis=1)

# Convert symptoms (binary columns) into text
data['symptoms_text'] = data.iloc[:, :-1].apply(lambda x: ', '.join(x.index[x == 1]), axis=1)

# Split into training and testing sets
# train_texts, test_texts, train_labels, test_labels = train_test_split(data['symptoms_text'], data['prognosis'], test_size=0.2)

# Add noise to input data
def add_noise_to_input(data, noise_level=0.1):
    """
    Add noise to input data by randomly toggling symptoms.
    """
    def toggle_symptoms(row):
        symptoms = row.split(', ')
        num_noisy = int(len(symptoms) * noise_level)
        for _ in range(num_noisy):
            symptom = random.choice(symptom_columns)
            if symptom in symptoms:
                symptoms.remove(symptom)  # Remove symptom
            else:
                symptoms.append(symptom)  # Add symptom
        return ', '.join(symptoms)
    data['symptoms_text'] = data['symptoms_text'].apply(toggle_symptoms)

# Apply noise
add_noise_to_input(data)

# First split: into train (70%) and temporary (30%) sets
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    data['symptoms_text'], data['prognosis'], test_size=0.3, random_state=42
)

# Second split: divide temporary set into validation (15%) and test (15%) sets
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42
)

# Step 2: Apply RandomOverSampler to balance the dataset
ros = RandomOverSampler(random_state=42)
train_texts_resampled, train_labels_resampled = ros.fit_resample(train_texts.values.reshape(-1, 1), train_labels)

# Reduce training data to 80%
train_texts_resampled = train_texts_resampled[:int(0.8 * len(train_texts_resampled))]
train_labels_resampled = train_labels_resampled[:int(0.8 * len(train_labels_resampled))]

# Flatten the resampled train_texts back into a 1D array
train_texts_resampled = train_texts_resampled.flatten()

# Step 2: Tokenize the input data using BioBERT tokenizer
tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.2')

# train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
# test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=128)

train_encodings = tokenizer(list(train_texts_resampled.flatten()), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=128)

# Convert labels to numerical format
label_dict = {label: i for i, label in enumerate(data['prognosis'].unique())}
train_labels = [label_dict[label] for label in train_labels_resampled]
val_labels = [label_dict[label] for label in val_labels]
test_labels = [label_dict[label] for label in test_labels]

# Ensure all unique labels are included in the classes argument
unique_labels = np.unique(train_labels)

# Step 4: Compute Class Weights for training
class_weights = compute_class_weight('balanced', classes=unique_labels, y=train_labels)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

# Step 3: Create a Dataset class for PyTorch
class SymptomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SymptomDataset(train_encodings, train_labels)
val_dataset = SymptomDataset(val_encodings, val_labels)
test_dataset = SymptomDataset(test_encodings, test_labels)

val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Step 4: Load the BioBERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('dmis-lab/biobert-base-cased-v1.2', num_labels=len(label_dict), ignore_mismatched_sizes=True)

# Function to save metrics and confusion matrix
def save_metrics_and_confusion_matrix(metrics, confusion_matrix, metrics_file, confusion_file, append=False):
    # Save metrics (append for epoch, overwrite for final test)
    if append:
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as file:
                metrics_list = json.load(file)
        else:
            metrics_list = []
        metrics_list.append(metrics)
        with open(metrics_file, 'w') as file:
            json.dump(metrics_list, file, indent=4)
    else:
        with open(metrics_file, 'w') as file:
            json.dump(metrics, file, indent=4)

    # Save confusion matrix (overwrite)
    df_confusion = pd.DataFrame(confusion_matrix)
    df_confusion.to_csv(confusion_file, index=False)

# Temperature scaling function
class TemperatureScaler(nn.Module):
    def __init__(self):
        super(TemperatureScaler, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature

# Fit temperature scaling on validation dataset
def fit_temperature_scaling(model, val_loader, device="cpu"):
    model.eval()
    logits_list, labels_list = [], []

    with torch.no_grad():
        for batch in val_loader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
            labels = batch["labels"].to(device)
            logits = model(**inputs).logits
            logits_list.append(logits)
            labels_list.append(labels)

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    temp_model = TemperatureScaler().to(device)
    optimizer = torch.optim.LBFGS([temp_model.temperature], lr=0.01, max_iter=50)

    def loss_fn():
        scaled_logits = temp_model(logits)
        probabilities = softmax(scaled_logits.detach().cpu().numpy(), axis=1)
        loss = -np.mean(np.log(probabilities[np.arange(len(labels)), labels.cpu().numpy()]))
        return torch.tensor(loss, requires_grad=True)

    optimizer.step(loss_fn)
    print(f"Optimal temperature: {temp_model.temperature.item()}")
    return temp_model.temperature.item()

# Function to calculate Expected Calibration Error (ECE)
def compute_ece(logits, labels, n_bins=10):
    logits = logits.cpu().numpy()
    labels = labels.cpu().numpy()
    probabilities = softmax(logits, axis=1)
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)

    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if np.sum(bin_mask) > 0:
            bin_accuracy = np.mean(predictions[bin_mask] == labels[bin_mask])
            bin_confidence = np.mean(confidences[bin_mask])
            ece += np.abs(bin_confidence - bin_accuracy) * np.mean(bin_mask)

    return ece

# Define the custom compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Apply temperature scaling
    scaled_logits = logits / optimal_temperature  # Scale logits using the optimal temperature
    predictions = np.argmax(scaled_logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    conf_matrix = confusion_matrix(labels, predictions)
    ece = compute_ece(torch.tensor(scaled_logits), torch.tensor(labels))
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'ece': ece
    }

    # Save metrics and confusion matrix for validation
    save_metrics_and_confusion_matrix(
        metrics,
        conf_matrix,
        metrics_file='validation_metrics.json',
        confusion_file='validation_confusion_matrix.csv',
        append=True
    )

    return metrics

# Step 5: Fine-tuning configuration using Hugging Face Trainer API
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch",  # Save at the end of each epoch
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    load_best_model_at_end=True,  # Save the best model
    metric_for_best_model="eval_loss",  # Use eval_loss as the metric
    greater_is_better=False,  # Lower eval_loss is better
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

device = "cuda" if torch.cuda.is_available() else "cpu"
optimal_temperature = fit_temperature_scaling(model, val_loader, device=device)

# Step 6: Train the model
trainer.train()

# Step 7: Evaluate the model on the test set
results = trainer.predict(test_dataset)
print("Test results:", results)

# Compute confusion matrix
test_predictions = np.argmax(results.predictions, axis=-1)
conf_matrix_test = confusion_matrix(results.label_ids, test_predictions)

save_metrics_and_confusion_matrix(
    results.metrics,
    conf_matrix_test,
    metrics_file='test_metrics.json',
    confusion_file='test_confusion_matrix.csv',
    append=False
)

# Save the fine-tuned model and tokenizer
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')