# Greenhouse Plant Disease Classifier

A deep learning computer vision project to spot plant diseases. Built with a data pipeline, a CNN using transfer learning, and deployed on a Raspberry Pi for greenhouse monitoring.

## How It Works

### 1. Data Pipeline (ETL)
* **`download_dataset.py`**: Grabs the raw dataset.
* **`testing_dataframe.py`**: Sets up the data for training. 
* **Processing**: Uses Python (pandas, SciPy) to process over 87,000 images across 38 plant disease classes. Applies pixel normalization and geometric data augmentation to make sure the model can handle different lighting and angles.

### 2. Deep Learning Model
* **`main.py`**: The main script that builds and trains the Convolutional Neural Network (CNN) using PyTorch and transfer learning. 
* **`plant_cnn_best.keras`**: The final, saved model weights, optimized to run on low-power hardware.

### 3. Testing and Deployment
* **`Validating_model.py`**: Runs tests on the trained model to check performance.
* **`test_predictions.csv`**: Logs the validation results.
* **Edge Deployment**: The model runs locally on a Raspberry Pi. It hits **97% accuracy** on a mock dataset, proving it works well for real-time edge monitoring.

## Repository Structure

```text
├── download_dataset.py    # Fetches the dataset
├── testing_dataframe.py   # Formats and checks the data
├── main.py                # Runs the ETL pipeline and trains the CNN
├── Validating_model.py    # Tests the model against unseen data
├── plant_cnn_best.keras   # The trained model weights
├── test_predictions.csv   # Output logs from validation
├── requirements.txt       # Project dependencies
└── README.md              # This document
