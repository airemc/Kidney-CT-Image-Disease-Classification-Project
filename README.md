# Kidney CT Image Disease Classification Project
This repository presents a project focused on classifying Kidney CT scan images into four distinct pathological categories using a fine-tuned ResNet50 deep learning network structure. The project successfully leverages Transfer Learning to achieve high classification reliability.

🔬 Introduction and Methodology
This project aims to detect abnormalities (Cyst, Normal, Stone, Tumor) in kidney CT images using a robust ResNet50 architecture. Due to the complexity and scarcity of medical image data, the Transfer Learning methodology was adopted, providing a strong foundation for image feature extraction.

Dataset: Kidney CT Scans (Cyst, Normal, Stone, Tumor Classes)

Algorithm: Fine-Tuned ResNet50 (Convolutional Neural Network - CNN)

Goal: Multi-class Classification (4 Classes)

All technical details regarding the architecture, layer configurations, and preprocessing steps are thoroughly documented within the project's main analysis file.

🧠 Network Architecture and Training Strategy
The network structure was meticulously trained using a two-phase fine-tuning strategy built upon a powerful pre-trained base model.

1. Network Architecture
The architecture uses ResNet50 (ImageNet pre-trained) for deep feature extraction, followed by a custom classification head designed for specific disease detection:

Custom Classification Head: Includes a Dense(512, activation='relu') layer to adapt high-level features, followed by a Dropout(0.4) layer for regularization to prevent overfitting.

Output Layer: A Dense(4, activation='softmax') layer produces the final probability distribution for the four classes.

2. Fine-Tuning Strategy
Head Training: Initially, the ResNet50 body was frozen (non-trainable), and only the newly added classification layers were trained.

End-to-End Fine-Tuning: Subsequently, the final 50 layers of ResNet50 were unfrozen. The entire network was then retrained with a very low learning rate using the Adam Optimizer to finely tune the weights to the nuances of the kidney CT data.

📊 Final Performance Metrics and Interpretation
The reliability of the network structure is comprehensively analyzed, demonstrating a strong, optimized performance that surpasses initial expectations.

1. Core Performance Metrics
The model achieved highly reliable results across the test set:

Final Test Accuracy (Güvenilirlik): 84.62%

Loss Function: Categorical Crossentropy

Key Classification Metrics: The model was rigorously evaluated based on Precision (Kesinlik), Recall (Duyarlılık/Hassasiyet), F1-Score, and Specificity (Özgüllük) across all classes.

2. Visual and Robustness Analysis
Visual Metrics: The Dataset Histogram confirmed manageable class distribution. The Confusion Matrix confirmed the strong accuracy while pinpointing specific areas of confusion (e.g., mistaking a Tumor for a Cyst), guiding future improvements.

Generalization Test: A critical test confirmed the model's ability to handle real-world variations, such as external grayscale (B&W) CT scans. This required a Robustness Fix involving an OpenCV B&W-to-RGB conversion and Normalization Fix in the preprocessing pipeline.

Test Outcome: The model successfully generalized with a high Prediction Confidence of 88.92%, validating its practical utility despite format shifts.

💻 Addendum: Overcoming Technical Barriers
The most critical technical achievement was overcoming the persistent Keras/TensorFlow loading errors associated with fine-tuned models, which complicated further analysis.

Functional API Reconstruction (Critical Solution): Loading the fine-tuned .h5 file resulted in corrupted internal tensor references (AttributeError, ValueError). The solution involved rebuilding the network architecture from scratch using the Functional API, ensuring the preservation and clean copying of all original trained weights. This method provides a stable and reliable network structure for deployment and further analysis.

🚀 Conclusion and Future Work
This project successfully established a reliable deep learning pipeline for kidney image classification. The strong accuracy and proven robustness confirm the network structure's potential as a valuable pre-diagnostic tool.

Future Plans
To enhance the project's quality and practical utility, future work will focus on:

Architectural Exploration: Aiming to push reliability above 90% by experimenting with modern, efficient architectures like DenseNet or EfficientNet.

Deeper Generalization: Integrating training with original grayscale CT/MRI data to better reflect actual clinical input.

Dynamic Data Integration: Integrating the model with a dynamic simulation or clinical data stream for real-time application scenarios.

Career Direction: The project serves as a foundation for exploring Medical Imaging Analysis (MIA) and related technologies needed to integrate this network structure into clinical workflows.

# Links

https://www.kaggle.com/code/ayeiremolak/kidney-ct-image-disease-classification-project
