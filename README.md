# Research Study: Greenhouse Plant Disease Classification via Convolutional Neural Networks (CNN)

## 1. Abstract
This project investigates the application of deep learning for the automated detection and classification of plant pathogens in controlled greenhouse environments. By engineering a Convolutional Neural Network (CNN) model optimized for computer vision tasks, this system analyzes leaf morphology and discoloration to accurately classify common agricultural diseases, providing a scalable solution for early-stage crop monitoring.

## 2. Introduction (Problem Statement)
In high-density greenhouse farming, the rapid spread of plant diseases can lead to catastrophic crop yield losses if not identified and isolated early. Traditional monitoring relies heavily on manual visual inspection by agricultural experts, a process that is time-consuming, expensive, and subject to human error. The objective of this project was to develop a machine vision classification model capable of consistently and rapidly identifying plant diseases from image data, thereby automating the preliminary diagnostic process.

## 3. Methodology & System Architecture
The classification engine was built utilizing a deep learning framework designed to extract hierarchical spatial features from plant imagery.

* **Data Preprocessing:** Input images undergo normalization, augmentation (rotation, zooming, and flipping to prevent overfitting), and resizing to match the input tensor requirements of the neural network.
* **Model Architecture (CNN):** The core system leverages a Convolutional Neural Network designed for image classification. It utilizes multiple convolutional layers for feature extraction (edge detection, texture analysis) followed by max-pooling layers for spatial dimension reduction.
* **Classification Head:** The fully connected dense layers at the end of the network use a Softmax activation function to output a probability distribution across multiple predefined plant disease classes.
* **Tech Stack:** Python, TensorFlow/Keras (or PyTorch), OpenCV (for image processing), NumPy, and Matplotlib.

## 4. Results & Core Functionality
When evaluated against the validation dataset, the current model architecture successfully demonstrates:
* **High-Fidelity Classification:** The model successfully categorizes images into distinct health states (e.g., healthy, blight, rust, mildew) with high statistical accuracy.
* **Batch Processing:** The inference script allows for the evaluation of multiple leaf images simultaneously, making it viable for bulk greenhouse data analysis.
* **Predictive Confidence:** Outputs include confidence intervals alongside the predicted class, allowing operators to flag low-confidence predictions for manual review.

## 5. Discussion & Future Scope
A primary technical challenge in developing this model involved standardizing the ambient lighting and shadow variations inherent to raw greenhouse imagery. Future iterations of this project will focus on:
* Implementing Transfer Learning (e.g., adapting a pre-trained ResNet or MobileNet model) to further improve accuracy and reduce training time.
* Developing a lightweight deployment model optimized for edge devices (such as a Raspberry Pi with a camera module) for real-time, in-situ greenhouse monitoring.

---

## Appendix: Study Replication (Local Setup)

To deploy the classification model locally for evaluation or further training, follow the replication procedure below.

### Prerequisites
* Python 3.8+
* Git
* (Optional but recommended) NVIDIA GPU with CUDA support for accelerated training

### Initialization Steps

1. **Clone the repository**
   ```bash
   git clone [https://github.com/AlanSanjit/greenhouse-plant-disease-classifier.git](https://github.com/AlanSanjit/greenhouse-plant-disease-classifier.git)
   cd greenhouse-plant-disease-classifier
2. **Establish a Virtual Environment (Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
4. **Run the Inference Script to test the model against a sample image, execute the prediction script:**
   ```bash
   python predict.py --image path/to/sample/leaf_image.jpg
5. **(Optional) Model Training. If you wish to retrain the model with a new dataset, ensure your data is sorted into categorical folders within the /data directory, then run:**
   ```bash
   python train.py --epochs 25 --batch_size 32
