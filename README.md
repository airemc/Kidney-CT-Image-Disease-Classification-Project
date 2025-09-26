# Kidney-CT-Image-Disease-Classification-Project
This repository presents a project focused on classifying Kidney CT scan images into four distinct pathological categories using a fine-tuned ResNet50 deep learning network structure. The project leverages Transfer Learning to achieve high classification reliability.


# Introduction
This project aims to detect abnormalities (Cyst, Stone, Tumor) in kidney CT images using a robust ResNet50 architecture. Due to the complexity of medical image data, the Transfer Learning methodology was adopted.

Feature	Detail
Dataset:	Kidney CT Scans (Cyst, Normal, Stone, Tumor Classes)
Algorithm:	Fine-Tuned ResNet50 (Convolutional Neural Network - CNN)
Goal:	Multi-class Classification (4 Classes)

All technical details regarding the architecture, layer configurations, and preprocessing steps are thoroughly documented within the project's main analysis file (notebook or script) using Markdown formatting.

# Network Architecture and Training Methodology
The network structure was trained using a two-phase fine-tuning strategy, built upon a powerful pre-trained base model.

1. Network Architecture
The architecture uses ResNet50 for feature extraction and adds a custom classification head:

Layer	Configuration	Purpose
Base Model:	ResNet50 (ImageNet pre-trained)	To extract deep, robust visual features from the images.
Custom Dense Layer:	Dense(512, activation='relu')	To adapt the high-level features for specific disease classification.
Regularization:	Dropout(0.4)	To prevent Overfitting and improve generalization capability.
Output Layer:	Dense(4, activation='softmax')	To produce the final probability distribution for the four classes.

2. Fine-Tuning Strategy
Head Training: Initially, the ResNet50 body was frozen (non-trainable), and only the newly added classification layers were trained.

End-to-End Fine-Tuning: Subsequently, the final layers of ResNet50 were unfrozen, and the entire network was retrained with a very low learning rate to finely tune the weights to the nuances of the kidney CT data.

# Metrics, Visualization, and Interpretation
The reliability of the network structure is comprehensively analyzed using both quantitative metrics and visual tools.

1. Core Performance Metrics
Reliability (Accuracy):	80.00%
Loss Function:	Categorical Crossentropy

2. Visual Metrics
The project utilizes the following visualizations to provide deeper insight into model and data performance:

Dataset Histogram (Class Distribution):

Purpose: Visualizes the balance of samples across the four classes (Cyst, Normal, Stone, Tumor) in the training and test sets.

Insight: Helps determine if class imbalance exists, which can bias the model towards majority classes.

Confusion Matrix:

Purpose: Visually maps the distribution of correct vs. incorrect predictions for each class.

Insight: The high values along the main diagonal confirm the 80% accuracy. More importantly, off-diagonal values pinpoint exactly which classes the network tends to confuse (e.g., mistaking a Tumor for a Cyst), guiding future model improvements.

3. Interpretation of Results
The 80.00% reliability demonstrates the network structure's strong capability in differentiating complex medical images. The visual metrics confirm the robustness of the predictions while identifying specific areas where performance can be optimized. This accuracy confirms the network structure's potential as a valuable pre-diagnostic tool.

# Addendum
The most critical technical achievement of this project was overcoming the persistent Keras/TensorFlow loading errors associated with fine-tuned models.

1. Functional API Reconstruction (Critical Solution)
Problem: Loading the fine-tuned .h5 file resulted in corrupted internal tensor references (e.g., AttributeError, ValueError), preventing further analysis like Grad-CAM.

Solution: The network architecture was rebuilt from scratch using the Functional API, while simultaneously preserving and copying the original trained weights.

Value: This method ensures that the final network structure used for prediction (model) is a clean, reliable copy of the original work, free from internal framework errors.

Conclusion and Future Work
This project successfully established a reliable deep learning pipeline for kidney image classification.

# Future Plans
To enhance the project's quality and practical utility, future work will focus on:

Dynamic Data Integration: Integrating the static dataset with a dynamic simulation or clinical data stream to make the project more reflective of real-time applications.

Architectural Exploration: Aiming to push reliability above 90% by experimenting with modern architectures like DenseNet or EfficientNet and performing rigorous comparative analysis.

Career Direction: The project serves as a foundation for exploring Medical Imaging Analysis (MIA) and related fields. Future efforts will involve studying technologies needed to integrate this network structure into clinical workflows.

# Links

https://www.kaggle.com/code/ayeiremolak/kidney-ct-image-disease-classification-project
