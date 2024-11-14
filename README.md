---

# Binary Classification of Breast Cancer Using Feed Forward Neural Network

## Overview

This project focuses on classifying breast cancer data using a **deep learning model**. The dataset used for this project is the **Breast Cancer dataset**, which contains various features related to cell measurements from breast cancer biopsies. The goal is to predict whether a tumor is malignant or benign based on these features using a **neural network**.

The deep learning model used is a simple **feedforward neural network (FNN)** built with Keras, which is part of TensorFlow. The model utilizes multiple hidden layers and trains on the data to classify cancer tumors effectively.

## Methodology

### Data Preprocessing:
1. **Dataset:** The dataset is provided by the `sklearn.datasets` module and consists of 30 features describing the characteristics of cell nuclei present in breast cancer biopsies.
2. **Splitting Data:** The dataset is split into training and testing sets using a **75%-25% split**.

### Model Architecture:
1. **Input Layer:** The model accepts 30 input features corresponding to different measurements of cell nuclei.
2. **Hidden Layers:** The model contains two hidden layers:
   - **First hidden layer:** 18 neurons with the **ReLU** activation function.
   - **Second hidden layer:** 12 neurons with the **ReLU** activation function.
3. **Output Layer:** The output layer has 1 neuron with a **Sigmoid** activation function to output probabilities for binary classification (malignant or benign).
4. **Loss Function:** The **binary crossentropy** loss function is used since this is a binary classification problem.
5. **Optimizer:** **Adam optimizer** is used to minimize the loss and update weights.

### Training:
The model is trained for 5000 epochs with a batch size of 40. This allows the model to learn from the training data and optimize its parameters for classification.

### Evaluation:
After training, the model is evaluated on the test set to predict cancerous tumor labels (0 for benign, 1 for malignant). The following metrics are used for evaluation:
- **Accuracy:** The proportion of correct predictions made by the model.
- **Confusion Matrix:** A matrix used to describe the performance of a classification algorithm by comparing predicted and actual labels.

## Requirements

Before running the project, ensure that the following Python libraries are installed:

- `keras`
- `tensorflow`
- `pandas`
- `scikit-learn`

You can install them using the following commands:

```bash
pip install keras
pip install tensorflow
pip install pandas
pip install scikit-learn
```

## Results

The model's performance is evaluated using the following metrics:

- **Accuracy Score:** Measures how many predictions the model got correct compared to the total number of predictions.
- **Confusion Matrix:** Provides a detailed breakdown of the number of correct and incorrect classifications (True Positives, False Positives, True Negatives, False Negatives).

**Accuracy Score:** (This would show the value calculated)

**Confusion Matrix:** (This would show the confusion matrix output)

---
## Conclusion

The neural network model successfully classifies breast cancer tumors as malignant or benign based on various features of cell nuclei. The accuracy score provides a measure of the model's performance, which can be further improved by tuning hyperparameters or exploring different architectures.

Future improvements could include:
- Implementing **cross-validation** to better assess model performance.
- Trying **different neural network architectures** like Convolutional Neural Networks (CNNs) or more advanced deep learning techniques.
- Using more advanced preprocessing techniques such as **feature scaling** to enhance model training.

---
