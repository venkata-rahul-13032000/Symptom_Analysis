import tkinter as tk
from tkinter import messagebox
from tkinter.ttk import Progressbar
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import re
import networkx as nx
import warnings
warnings.filterwarnings("ignore")
import os,csv

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained('./saved_model')
tokenizer = BertTokenizer.from_pretrained('./saved_model')

# Load symptoms and precaution data
symptom_severity = pd.read_csv('Symptom-severity.csv')
symptom_precaution = pd.read_csv('symptom_precaution.csv')
symptom_description = pd.read_csv('symptom_description.csv')

# Load the main dataset to recreate the label_dict
data = pd.read_csv('symbipredict_2022.csv')
symptom_columns = data.columns[:-1]  # All columns except the last "prognosis" column
symptoms_list = [symptom.replace('_', ' ') for symptom in symptom_columns]

# Normalize symptom names: replace underscores with spaces and make all lowercase
symptom_severity['Cleaned_Symptom'] = symptom_severity['Symptom'].str.replace(r'[^a-zA-Z]', '', regex=True).str.lower()
symptoms_list = [symptom.lower().strip() for symptom in symptoms_list]

# Create label_dict (mapping of diseases to numerical labels) as you did in training
label_dict = {label: i for i, label in enumerate(data['prognosis'].unique())}
label_dict_inv = {v: k for k, v in label_dict.items()}

# Load model performance metrics from a file (assumes metrics were saved during training)
try:
    with open("test_metrics.json", "r") as file:
        model_metrics = json.load(file)
except FileNotFoundError:
    model_metrics = {}

# Initialize the GUI window
root = tk.Tk()
root.title("AI-Assisted Healthcare Diagnosis")
root.state('zoomed')  # Full screen

# Clean a selected symptom for comparison
def clean_symptom(symptom):
    return re.sub(r'[^a-zA-Z]', '', symptom).lower()

# Function to filter symptoms based on search query
def filter_symptoms():
    query = search_var.get().lower()
    for symptom, var in symptoms_var.items():
        if query in symptom.lower():
            checkbuttons[symptom].pack(anchor='w')
        else:
            checkbuttons[symptom].pack_forget()

# Confidence-Based Feature Attribution
def confidence_based_attribution(predicted_diagnosis, selected_symptoms):
    # Get the binary symptom row for the predicted disease
    disease_row = data[data['prognosis'].str.lower() == predicted_diagnosis.lower()]
    if disease_row.empty:
        return "No attribution data available for this diagnosis."

    # List symptoms associated with the predicted disease
    disease_symptoms = [
        symptom.replace('_', ' ') for symptom in symptom_columns
        if disease_row.iloc[0][symptom] == 1
    ]

    # Find matched symptoms
    matched_symptoms = [symptom for symptom in selected_symptoms if symptom in disease_symptoms]
    return f"Contributing Symptoms: {', '.join(matched_symptoms) if matched_symptoms else 'None of the selected symptoms directly contributed.'}"


# Disease-Symptom Network Graph
def generate_network_graph(predicted_diagnosis, selected_symptoms):
    # Create a graph
    G = nx.Graph()

    # Add nodes for the disease and symptoms
    G.add_node(predicted_diagnosis, color='red')
    for symptom in selected_symptoms:
        G.add_node(symptom, color='blue')

    # Add edges between the disease and symptoms if they are linked in the dataset
    disease_row = data[data['prognosis'].str.lower() == predicted_diagnosis.lower()]
    if not disease_row.empty:
        for symptom in selected_symptoms:
            symptom_col = symptom.replace(' ', '_').lower()  # Match dataset format
            if symptom_col in disease_row.columns and disease_row.iloc[0][symptom_col] == 1:
                G.add_edge(predicted_diagnosis, symptom)

    # Draw the graph
    pos = nx.spring_layout(G)
    colors = [G.nodes[node]['color'] for node in G.nodes]
    nx.draw(G, pos, with_labels=True, node_color=colors, edge_color='gray', node_size=3000, font_size=10)
    plt.title("Disease-Symptom Network")
    plt.show()


# Manual Attribution with Dataset Insights
def manual_attribution(predicted_diagnosis, selected_symptoms):
    # Get the binary symptom row for the predicted disease
    disease_row = data[data['prognosis'].str.lower() == predicted_diagnosis.lower()]
    if disease_row.empty:
        return "No attribution data available for this diagnosis."

    # List symptoms associated with the predicted disease
    disease_symptoms = [
        symptom.replace('_', ' ') for symptom in symptom_columns
        if disease_row.iloc[0][symptom] == 1
    ]
    return f"Associated Symptoms for {predicted_diagnosis}: {', '.join(disease_symptoms)}"

match_rates = []

MAX_SYMPTOMS = 17

# Function to make prediction based on selected symptoms
def make_prediction():
    global predicted_diagnosis, selected_symptoms, match_rates  # Make match_rates accessible globally
    selected_symptoms = [symptom for symptom in symptoms_list if symptoms_var[symptom].get()]

    if len(selected_symptoms) > MAX_SYMPTOMS:
        messagebox.showerror("Error", f"Too many symptoms selected. Please select up to {MAX_SYMPTOMS} symptoms.")
        return

    if not selected_symptoms:
        messagebox.showwarning("No Symptoms Selected", "Please select at least one symptom.")
        return

    symptoms_text = ', '.join(selected_symptoms)
    inputs = tokenizer(symptoms_text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    # Get model prediction and probabilities
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()  # Keep as NumPy array
        predictions = np.argsort(probabilities, axis=1)[:, ::-1]  # Get sorted indices in descending order   

    # Get the top predictions
    top_predictions = [(label_dict_inv[i], probabilities[0][i]) for i in predictions[0][:3]]
    
    global confidence_score
    
    predicted_diagnosis, confidence_score = top_predictions[0]

    # Ensure `probabilities` is a 2D array for calibration
    plot_calibration.predictions = probabilities  # Already 2D after softmax
    plot_calibration.labels = np.array([label_dict[predicted_diagnosis]])  # True labels for calibration 

    # Check if selected symptoms contribute to the predicted diagnosis
    disease_row = data[data['prognosis'].str.lower() == predicted_diagnosis.lower()]
    contributing_symptoms = []
    global matched_symptoms
    matched_symptoms = []
    if not disease_row.empty:
        # List symptoms associated with the predicted disease
        contributing_symptoms = [
            symptom.replace('_', ' ') for symptom in symptom_columns
            if disease_row.iloc[0][symptom] == 1
        ]
        # Find matched symptoms
        matched_symptoms = [symptom for symptom in selected_symptoms if symptom in contributing_symptoms]

        # If no symptoms match, stop prediction and show a warning
        if not matched_symptoms:
            messagebox.showwarning(
                "No Symptom Match",
                f"None of the selected symptoms contribute to the predicted diagnosis."
            )
            return

    # Calculate match rate
    global match_rate
    match_rate = calculate_match_rate(selected_symptoms, matched_symptoms)
    match_rates.append((predicted_diagnosis, match_rate))  # Store match rate for visualization

    # Update GUI with the predicted diagnosis and confidence
    result_label.config(text=f"Predicted Diagnosis: {predicted_diagnosis}")
    confidence_label.config(text=f"Confidence: {confidence_score * 100:.2f}%")
    confidence_bar['value'] = confidence_score * 100

    # Show similar diagnoses and their scores
    similar_diagnoses = "\n".join([f"{disease}: {score * 100:.2f}%" for disease, score in top_predictions[1:]])
    similar_label.config(text=f"Other Possible Diagnoses:\n{similar_diagnoses}")

    # Display precautions and description
    precaution_row = symptom_precaution[symptom_precaution['Disease'].str.lower().str.strip() == predicted_diagnosis.lower().strip()]
    precautions = ', '.join(
        [precaution for precaution in precaution_row[['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values[0]
         if isinstance(precaution, str) and precaution.strip()]
    ) if not precaution_row.empty else "No precautions available."
    precautions_label.config(text=f"Precaution: {precautions}")

    description_row = symptom_description[symptom_description['Disease'].str.lower().str.strip() == predicted_diagnosis.lower().strip()]
    description_text = description_row['Description'].values[0] if not description_row.empty else "No description available."
    description_label.config(text=f"Description: {description_text}")

    # Confidence-based attribution
    confidence_text = f"Contributing Symptoms: {', '.join(matched_symptoms)}"
    confidence_label.config(text=f"Confidence-Based Attribution:\n{confidence_text}")

    # Manual attribution
    manual_text = f"Associated Symptoms for {predicted_diagnosis}: {', '.join(contributing_symptoms)}"
    attribution_label.config(text=manual_text)

    # Rule-based explanation for symptom severity
    relevant_symptoms = sorted(
        [
            (
                symptom,
                symptom_severity[symptom_severity['Cleaned_Symptom'] == clean_symptom(symptom)]['weight'].values[0]
                if not symptom_severity[symptom_severity['Cleaned_Symptom'] == clean_symptom(symptom)].empty else "Severity information not available"
            )
            for symptom in selected_symptoms
        ],
        key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True
    )
    explanation_text = ", ".join([f"{symptom}: {weight}" for symptom, weight in relevant_symptoms])
    explanation_label.config(text=explanation_text)

    network_graph_button.config(state=tk.NORMAL)
    calibration_button.config(state=tk.NORMAL)
    export_button.config(state=tk.NORMAL)
    match_rate_button.config(state=tk.NORMAL)



def reset_interface():
    # Reset symptom selection (checkboxes)
    for var in symptoms_var.values():
        var.set(0)

    # Clear search field
    search_var.set('')

    # Reset all labels and progress bar
    result_label.config(text="")
    confidence_label.config(text="")
    confidence_bar['value'] = 0
    similar_label.config(text="")
    precautions_label.config(text="")
    description_label.config(text="")
    explanation_label.config(text="")
    attribution_label.config(text="")

    # Reset clarity and relevance dropdowns
    clarity_var.set(0)  # Reset clarity rating
    relevance_var.set(0)  # Reset relevance rating

    # Clear any cached prediction results
    global predicted_diagnosis, selected_symptoms, match_rates
    predicted_diagnosis = ""
    selected_symptoms = []
    match_rates = []

    # Disable buttons that depend on predictions
    network_graph_button.config(state=tk.DISABLED)
    calibration_button.config(state=tk.DISABLED)
    export_button.config(state=tk.DISABLED)
    match_rate_button.config(state=tk.DISABLED)

    # Repack all symptom checkboxes (to make them visible again if filtered)
    for symptom in symptoms_var.keys():
        checkbuttons[symptom].pack(anchor='w')

    # Reset the scroll position to the top
    main_canvas.yview_moveto(0)


# Show model metrics
def show_model_metrics():
    metrics_text = (
        f"Model Metrics:\n"
        f"Accuracy: {model_metrics.get('test_accuracy', 'N/A') * 100:.2f}%\n"
        f"Precision: {model_metrics.get('test_precision', 'N/A') * 100:.2f}%\n"
        f"Recall: {model_metrics.get('test_recall', 'N/A') * 100:.2f}%\n"
        f"F1-Score: {model_metrics.get('test_f1', 'N/A') * 100:.2f}%\n"
        f"ECE: {model_metrics.get('test_ece', 'N/A') * 100:.2f}%\n"
    )
    messagebox.showinfo("Model Performance", metrics_text)

# Display metrics graph
def show_metrics_graph():
    if not model_metrics:
        messagebox.showerror("Error", "No metrics data available.")
        return

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ECE']
    values = [
        model_metrics.get('test_accuracy', 0),
        model_metrics.get('test_precision', 0),
        model_metrics.get('test_recall', 0),
        model_metrics.get('test_f1', 0),
        model_metrics.get('test_ece',0)
    ]

    plt.figure(figsize=(8, 6))
    plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
    plt.ylim(0, 1)
    plt.title("Model Metrics")
    plt.ylabel("Scores")
    plt.show()

def view_network_graph():
    try:
        generate_network_graph(predicted_diagnosis, selected_symptoms)
    except NameError:
        messagebox.showerror("Error", "Please make a prediction first to view the network graph.")

def submit_feedback():
    """
    Submit the feedback for the clarity and relevance of explanations.
    """
    clarity = clarity_var.get()
    relevance = relevance_var.get()

    # Ensure both clarity and relevance ratings are provided
    if clarity == 0 or relevance == 0:
        messagebox.showerror("Error", "Please rate both clarity and relevance.")
        return

    try:
        # Save feedback to a CSV file
        with open("feedback_log.csv", "a") as log_file:
            log_file.write(f"{predicted_diagnosis},{clarity},{relevance}\n")
        messagebox.showinfo("Feedback Submitted", "Thank you for your feedback!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save feedback: {e}")

def calculate_match_rate(selected_symptoms, contributing_symptoms):
    """
    Calculate the percentage of selected symptoms that match contributing symptoms.
    """
    if not selected_symptoms:
        return 0  # Avoid division by zero
    matches = set(selected_symptoms) & set(contributing_symptoms)
    return (len(matches) / len(selected_symptoms)) * 100

def plot_symptom_match_rate():
    """
    Visualize the symptom match rates for the predicted diseases.
    """
    if not match_rates:
        messagebox.showerror("Error", "No match rates available. Please make a prediction first.")
        return

    # Separate diseases and their match rates
    diseases, rates = zip(*match_rates)

    # Create the bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(diseases, rates, color="skyblue")
    plt.ylim(0, 100)
    plt.xlabel("Diseases")
    plt.ylabel("Symptom Match Rate (%)")
    plt.title("Symptom Match Rate by Disease")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def calculate_calibration_data(predictions, labels, n_bins=10):
    """
    Calculate confidence calibration data for the predictions.

    Args:
        predictions: Array of predicted probabilities (2D array).
        labels: True labels.
        n_bins: Number of bins for grouping confidence scores.

    Returns:
        Tuple of (bin_centers, bin_accuracies) for the calibration plot.
    """
    # Get maximum confidence scores and predicted classes
    confidences = np.max(predictions, axis=1)
    predicted_classes = np.argmax(predictions, axis=1)

    # Bin boundaries
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_accuracies = []
    for i in range(n_bins):
        bin_mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if np.any(bin_mask):  # Ensure the bin is not empty
            bin_accuracy = np.mean(predicted_classes[bin_mask] == labels[bin_mask])
        else:
            bin_accuracy = 0
        bin_accuracies.append(bin_accuracy)

    return bin_centers, bin_accuracies


def plot_calibration():
    """
    Generate and display a confidence calibration plot.
    """
    try:
        # Ensure predictions and labels exist
        if not hasattr(plot_calibration, "predictions") or not hasattr(plot_calibration, "labels"):
            messagebox.showerror("Error", "Please make predictions before viewing the calibration plot.")
            return

        predictions = plot_calibration.predictions
        labels = plot_calibration.labels

        # Calculate calibration data
        bin_centers, bin_accuracies = calculate_calibration_data(predictions, labels)

        # Plot the calibration curve
        plt.figure(figsize=(8, 5))
        plt.plot(bin_centers, bin_accuracies, marker="o", label="Model Calibration")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
        plt.xlabel("Confidence")
        plt.ylabel("Accuracy")
        plt.title("Confidence Calibration Plot")
        plt.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"Failed to generate calibration plot: {e}")

def export_results():
    """
    Export prediction results, confidence scores, and feedback to a CSV file.
    """
    # Check if there's a prediction available
    try:
        if not predicted_diagnosis or not selected_symptoms:
            messagebox.showerror("Error", "No prediction data to export.")
    except NameError:
        messagebox.showerror("Error", "No prediction data available. Please make a prediction first.")
        return

    # Prepare data to export
    export_data = {
        "Predicted Diagnosis": predicted_diagnosis,
        "Confidence Score (%)": f"{confidence_score * 100:.2f}",
        "Selected Symptoms": ", ".join(selected_symptoms),
        "Contributing Symptoms": ", ".join(matched_symptoms) if 'matched_symptoms' in globals() else "N/A",
        "Match Rate (%)": f"{match_rate:.2f}" if 'match_rate' in globals() else "N/A",
        "Clarity Feedback": clarity_var.get() if clarity_var.get() else "N/A",
        "Relevance Feedback": relevance_var.get() if relevance_var.get() else "N/A"
    }

    # File path to export results
    file_path = "exported_results.csv"

    try:
        # Check if the file exists, and append data if it does
        file_exists = os.path.isfile(file_path)
        with open(file_path, "a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=export_data.keys())

            # Write the header only if the file doesn't exist
            if not file_exists:
                writer.writeheader()

            # Write the prediction data
            writer.writerow(export_data)

        messagebox.showinfo("Success", "Results exported successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to export results: {e}")


# GUI layout
# Create a scrollable canvas
main_canvas = tk.Canvas(root)
main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Add a vertical scrollbar linked to the canvas
scrollbar = tk.Scrollbar(root, orient=tk.VERTICAL, command=main_canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
main_canvas.configure(yscrollcommand=scrollbar.set)

# Bind mouse scroll to the canvas
def on_mouse_wheel(event):
    main_canvas.yview_scroll(-1 * int(event.delta / 120), "units")

main_canvas.bind_all("<MouseWheel>", on_mouse_wheel)

# Create a frame inside the canvas
main_frame = tk.Frame(main_canvas)
main_canvas.create_window((0, 0), window=main_frame, anchor="nw")

instruction_label = tk.Label(main_frame, text="Search and select the symptoms you're experiencing:")
instruction_label.pack(pady=10)

search_var = tk.StringVar()
search_entry = tk.Entry(main_frame, textvariable=search_var)
search_entry.pack(pady=5)
search_var.trace_add("write", lambda name, index, mode: filter_symptoms())

checkbox_frame = tk.Frame(main_frame)
checkbox_frame.pack(fill=tk.BOTH, expand=True, pady=10)

# Update scroll region when the frame is resized
def update_scroll_region(event):
    main_canvas.configure(scrollregion=main_canvas.bbox("all"))

main_frame.bind("<Configure>", update_scroll_region)

symptoms_var = {}
checkbuttons = {}
for symptom in symptoms_list:
    var = tk.BooleanVar()
    symptom_check = tk.Checkbutton(checkbox_frame, text=symptom, variable=var)
    symptom_check.pack(anchor='w')
    symptoms_var[symptom] = var
    checkbuttons[symptom] = symptom_check

# Create a frame for the buttons to align them in a single line
button_frame = tk.Frame(main_frame)
button_frame.pack(pady=10)

# Add buttons to the frame
submit_button = tk.Button(button_frame, text="Submit", command=make_prediction)
submit_button.pack(side=tk.LEFT, padx=5)

reset_button = tk.Button(button_frame, text="Reset", command=reset_interface)
reset_button.pack(side=tk.LEFT, padx=5)

metrics_button = tk.Button(button_frame, text="View Model Metrics", command=show_model_metrics)
metrics_button.pack(side=tk.LEFT, padx=5)

# metrics_graph_button = tk.Button(button_frame, text="View Metrics Graph", command=show_metrics_graph)
# metrics_graph_button.pack(side=tk.LEFT, padx=5)

# Add a button to view the disease-symptom network graph
network_graph_button = tk.Button(button_frame, text="View Disease-Symptom Network", command=view_network_graph, state=tk.DISABLED)
network_graph_button.pack(side=tk.LEFT, padx=5)

match_rate_button = tk.Button(button_frame, text="View Symptom Match Rate", command=plot_symptom_match_rate, state=tk.DISABLED)
match_rate_button.pack(side=tk.LEFT, padx=5)

calibration_button = tk.Button(button_frame, text="View Calibration Plot", command=plot_calibration, state=tk.DISABLED)
calibration_button.pack(side=tk.LEFT, padx=5)

export_button = tk.Button(button_frame, text="Export Results", command=export_results, state=tk.DISABLED)
export_button.pack(side=tk.LEFT, padx=5)

result_label = tk.Label(main_frame, text="", font=("Helvetica", 14))
result_label.pack(pady=10)
confidence_label = tk.Label(main_frame, text="", font=("Helvetica", 12), wraplength=1500, justify="center")
confidence_label.pack(pady=5)

confidence_bar = Progressbar(main_frame, orient="horizontal", length=300, mode="determinate")
confidence_bar.pack(pady=5)

similar_label = tk.Label(main_frame, text="", font=("Helvetica", 12))
similar_label.pack(pady=5)
precautions_label = tk.Label(main_frame, text="", font=("Helvetica", 12), wraplength=1500, justify="center")
precautions_label.pack(pady=10)
description_label = tk.Label(main_frame, text="", font=("Helvetica", 12), wraplength=1500, justify="center")
description_label.pack(pady=10)

# Add a new label for explanations in the GUI layout
explanation_label = tk.Label(main_frame, text="", font=("Helvetica", 12), wraplength=1500, justify="center")
explanation_label.pack(pady=10)

# Add a label for manual attribution
attribution_label = tk.Label(main_frame, text="", font=("Helvetica", 12), wraplength=1500, justify="center")
attribution_label.pack(pady=5)

# Frame for user feedback
feedback_frame = tk.Frame(main_frame)
feedback_frame.pack(pady=10)

# Clarity rating
tk.Label(feedback_frame, text="Rate Clarity (1-5):").pack(side=tk.LEFT)
clarity_var = tk.IntVar()
clarity_dropdown = tk.OptionMenu(feedback_frame, clarity_var, *range(1, 6))
clarity_dropdown.pack(side=tk.LEFT, padx=5)

# Relevance rating
tk.Label(feedback_frame, text="Rate Relevance (1-5):").pack(side=tk.LEFT)
relevance_var = tk.IntVar()
relevance_dropdown = tk.OptionMenu(feedback_frame, relevance_var, *range(1, 6))
relevance_dropdown.pack(side=tk.LEFT, padx=5)

# Feedback submission button
feedback_button = tk.Button(feedback_frame, text="Submit Feedback", command=submit_feedback)
feedback_button.pack(side=tk.LEFT, padx=5)

root.mainloop()