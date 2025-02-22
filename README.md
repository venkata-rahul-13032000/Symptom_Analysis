# AI-Assisted Symptom Analysis for

# Healthcare Diagnostics

### Overview

This project extends the AI healthcare diagnostic tool developed in the midterm. It helps users
find potential diagnoses based on symptoms while improving trustworthiness, transparency,
and robustness. The system uses a fine-tuned BioBERT model to analyze symptoms and predict
potential health conditions. Enhanced features, such as Confidence Calibration, Disease-
Symptom Network Graphs, and Feedback Collection, provide reliability, fairness, and
interpretability to the system.

### Features

**Enhanced AI Diagnosis**

- **Symptom Selection** : Interactive dropdown search to select symptoms.
- **Disease Prediction** : Accurate predictions using the fine-tuned **BioBERT model**.
- **Confidence Calibration** : Temperature scaling for reliable confidence scores.
- **Attribution** :
    - **Confidence-Based Attribution** : Highlights key symptoms contributing to the
       diagnosis.
    - **Manual Attribution** : Lists associated symptoms from the dataset.

**Visual Interpretability**

- **Disease-Symptom Network Graph** : Illustrates the relationship between symptoms
    and predicted diseases.
- **Symptom Match Rate Plot** : Visualizes the alignment of selected symptoms with
    predictions.

**Trustworthiness Features**

- **Fairness** :
    - Symptom Match Rate ensures unbiased predictions.
    - Calibration plots validate reliable confidence scoring.
- **Reliability** :
    - Handles noisy symptom inputs for robustness.
    - Limits symptom input to avoid invalid predictions.
- **Transparency** :
    - Detailed explanations for each prediction with attribution methods.

**Feedback Integration**

- Users can rate the clarity and relevance of predictions.
- Feedback is stored locally for iterative improvements.


**GUI Enhancements**

- **Interactive and Scrollable Interface** : Handles large symptom datasets seamlessly.
- **Error Handling** : Provides clear messages for invalid actions.
- **Data Export** : Saves predictions, confidence scores, and feedback to CSV.

### Code Logic & Flow

The application logic is built around the interaction between the **user interface** , **symptom
processing** , and the **fine-tuned BERT model**.

**Data Preprocessing (main.py)**

- **Dataset Loading** : Reads and fills missing values in symbipredict_2022.csv.
- **Symptom Standardization** : Converts labels to lowercase and applies severity weights
    using Symptom-severity.csv.
- **Text Encoding** : Encodes symptoms into text for NLP processing.
- **Noise Injection** : Adds noise for robustness against real-world variability.
- **Data Splitting** :
    - 70% training, 15% validation, 15% testing.
- **Oversampling** : Balances classes using RandomOverSampler.

**Model Training**

- **Tokenizer** : Uses BioBERT (dmis-lab/biobert-base-cased-v1.2) for embeddings.
- **Architecture** :
    - BioBERT fine-tuned for sequence classification.
    - Outputs match unique diagnoses.
- **Class Weights** : Handles class imbalance.
- **Training Configuration** : 3 epochs, learning rate: 1×10<sup>−5</sup> , early stopping.
- **Metrics** : Accuracy, Precision, Recall, F1, and ECE.
- **Calibration** : Implements temperature scaling for confidence scores.
- **Output** : Saves fine-tuned model and evaluation results.

**Prediction and Evaluation**

- **Prediction Workflow** : Tokenizes symptoms and predicts top-3 diagnoses.
- **Confidence Calibration** : Ensures reliable confidence scores.
- **Evaluation Metrics** : Accuracy, Precision, Recall, F1, ECE, and symptom match rates.

**Graphical User Interface (main_gui.py)**

- **Symptom Input** : Dropdown and checkbox selection; limited to 17 symptoms.
- **Prediction Display** : Shows diagnosis, confidence, and key symptoms.
- **Visualization** :
    - **Disease-Symptom Network Graph** : Maps relationships.
    - **Symptom Match Rate Plot** : Aligns symptoms with predictions.
    - **Confidence Calibration Plot** : Reliability diagram.
- **Feedback** : Collects clarity/relevance ratings, stores feedback in feedback_log.csv.


### Technologies Used

- **Python:** Core programming language for implementing the project.
- **Transformers (Hugging Face):** Used for fine-tuning and sequence classification.
- **PyTorch:** Backend for model training, fine-tuning, and inference.
- **Scikit-learn:** Data splitting and class weight computation.
- **Matplotlib:** For plotting evaluation metrics, calibration plots, and visualizations.
- **Tkinter:** GUI library for developing the user interface.
- **Pandas:** For dataset processing and analysis.

### Installation & Setup

- **Install Dependencies (~ 2 minutes):**\
    Before running the application, install the necessary libraries:
  - pip install -r requirements.txt
- **Download the Repository:**\
    Ensure the datasets (Symptom-severity.csv, symptom_precaution.csv,
    symptom_description.csv, and symbipredict_2022.csv) are in the same directory as the
    code.
- **Model Training (~ 50 minutes):** **_Skip if you have saved model_**\
If you do not have the pre-trained model, follow these steps to fine-tune it:
  - Ensure datasets are in place (as listed above).
  - Run the model training script to fine-tune the BioBERT model using the following command:
    - _python main.py_\
    This script will train and save the model in the “./saved_model” directory.
  - Model Output: The fine-tuned model will be saved in the “./saved_model”
          directory, ready for use in the GUI.
- **Launch the Application:**
    To launch the application, run the following command:
  - _python main_gui.py_
- **Using the Interface:**\
    Search and Select Symptoms: Use the dropdown menu or search bar.\
    Submit for Diagnosis: Click "Submit" for diagnoses, precautions, and descriptions.\
    Reset Functionality: Use the "Reset" button to clear inputs for a new session.\
    Explore Visualizations: View calibration plots, symptom match rates, and disease-
    symptom networks.\
    Provide Feedback: Rate clarity and relevance, and export results if needed.

### Project Structure

- _main.py:_ Script for training and fine-tuning the BERT model.
- _gui.py:_ Tkinter-based GUI for interacting with the AI.
- _saved_model:_ The directory containing the trained model files.
- _Symptom-severity.csv:_ Symptom severity information.
- _symptom_precaution.csv:_ Precautionary measures for different diagnoses.
- _symptom_description.csv:_ Descriptions of conditions and diagnoses.
- _symbipredict_2022.csv:_ Main dataset used for symptom-to-diagnosis mapping.
- _requirements.txt:_ Libraries necessary for the application.
