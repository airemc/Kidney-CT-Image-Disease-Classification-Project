
# 🏆 Kidney CT Image Disease Classification Project
This repository documents a project that successfully classifies Kidney CT scan images into four distinct pathological categories using a Fine-Tuned ResNet50 deep learning network. By rigorously applying Class Weighting and a two-stage Transfer Learning strategy, the project achieved exceptionally high reliability, even for critical minority classes.

## 🔬 Introduction and Methodology
This project aimed to achieve highly accurate detection of abnormalities (Cyst, Normal, Stone, Tumor) in kidney CT images. Given the challenges of limited and imbalanced medical image data, a robust methodology was adopted to ensure the model did not overlook critical pathologies.

The solution is based on a Two-Stage Fine-Tuned ResNet50 architecture, leveraging its powerful feature extraction capabilities. A key focus was managing the data imbalance to prioritize high Recall for life-critical conditions like 'Stone' and 'Tumor'.

## 🧠 Network Architecture and Training Strategy
The network structure was meticulously trained using a strategy that adapts the model's pre-trained knowledge to the specific nuances of kidney CT data.

**Training Strategy Highlights**

1. Stage 1: Feature Extraction: The entire ResNet50 body was frozen (non-trainable), and only a newly added custom classification head was trained. This provided a rapid establishment of the core feature set.

2. Stage 2: End-to-End Fine-Tuning: The final 50 layers of ResNet50 were unfrozen and the entire network was trained again. A very low learning rate was used to ensure the model gently adapted the pre-trained weights to the new domain, leading to the final performance boost.

**Architecture Details**
The architecture uses ResNet50 as the backbone, topped with a custom head featuring a Dense(512,activation= ′relu ′) layer and a Dropout(0.3) layer for robust regularization, feeding into a Dense(4,activation= ′softmax ′) output layer.

## 🚀 Key Achievements and Robustness
The model's reliability is proven not just by its overall accuracy, but by its performance on medically critical metrics, directly demonstrating the success of the imbalance management techniques.

**Core Performance Metrics (Test Set)**
The model achieved outstanding results on the unseen test set:

Final Test Accuracy: 99.68%

Test Loss:	0.0122

Support:	1867 Samples


**Imbalance Management is Key--
The core success factor was the implementation of dynamically calculated Class Weights. This approach heavily penalized misclassifications of the minority classes (Stone, Tumor).

The resulting ≈0.99 Recall for these critical conditions proves the model's ability to minimize False Negatives (missed diagnoses), confirming its high clinical potential.

**Generalization Test**
The model's robustness was validated against real-world variations, including external grayscale CT scans. This required a Robustness Fix (OpenCV B&W-to-RGB conversion) in the preprocessing pipeline, which successfully ensured the model maintained its high performance despite format shifts.

## 🌟 Conclusion and Future Work
This project has successfully established a highly reliable deep learning pipeline for kidney image classification. The combination of high 99.68% accuracy and the near-perfect Recall on critical classes confirms the network's potential as a valuable pre-diagnostic tool.

**Future Plans**

Explainable AI (XAI): Integrate Grad-CAM analysis to visually demonstrate which image regions drive the model's prediction, providing essential clinical transparency.

Architectural Exploration: Experiment with newer, more compute-efficient architectures (like DenseNet or EfficientNet) to optimize for deployment speed.

Deployment Preparation: Convert the final model into a lightweight format (e.g., TensorFlow Lite or TensorFlow.js) and build a simple web interface for real-time application scenarios.

## Links

https://www.kaggle.com/code/ayeiremolak/kidney-ct-image-disease-classification
