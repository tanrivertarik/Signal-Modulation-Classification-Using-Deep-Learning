# Signal Modulation Classification Using Deep Learning

## Introduction

In the rapidly evolving field of wireless communications, accurate and efficient signal modulation classification is paramount. Modulation recognition plays a critical role in various applications such as cognitive radio, spectrum monitoring, and electronic warfare. Traditional methods for modulation classification often rely on handcrafted features and statistical analysis, which may not generalize well to complex and noisy environments.

This project explores the application of deep learning techniques, specifically Convolutional Neural Networks (CNN) combined with Long Short-Term Memory (LSTM) networks, to automatically classify signal modulation types from raw IQ (In-phase and Quadrature) data. Utilizing the RadioML 2018.01A dataset, we aim to build a robust model capable of recognizing various modulation schemes even in the presence of noise and interference.

## Objectives

- **Develop a deep learning model** that can accurately classify different signal modulation types from raw IQ data.
- **Implement data preprocessing** techniques to prepare the dataset for training.
- **Train and evaluate the model** to achieve high classification accuracy.
- **Provide a mechanism** to classify new signals and verify the model's predictions.
- **Document the process and results** for reproducibility and further research.

- 

## Dataset Overview

The RadioML 2018.01A dataset is a comprehensive collection of synthetic and over-the-air recordings of 24 different digital and analog modulation types. Key characteristics of the dataset include:

- **Modulation Types:** OOK, 4ASK, 8ASK, BPSK, QPSK, 8PSK, 16PSK, 32PSK, 16APSK, 32APSK, 64APSK, 128APSK, 16QAM, 32QAM, 64QAM, 128QAM, 256QAM, AM-SSB-WC, AM-SSB-SC, AM-DSB-WC, AM-DSB-SC, FM, GMSK, OQPSK.
- **Signal-to-Noise Ratios (SNR):** Ranging from -20 dB to +30 dB in 2 dB steps.
- **Samples per Modulation-SNR Combination:** 4096 frames.
- **Frame Size:** Each frame contains 1024 complex IQ samples, resulting in a shape of (1024, 2).

For this project, we focus on a subset of the modulation classes and SNR values to optimize model performance and reduce computational requirements.

## Methodology

### Data Preprocessing

1. **Modulation Class Selection:**
   - Selected 8 modulation types: `'4ASK', 'BPSK', 'QPSK', '16PSK', '16QAM', 'FM', 'AM-DSB-WC', '32APSK'`.
   - This selection reduces complexity and focuses on classes with distinct characteristics.

2. **SNR Range Limitation:**
   - Used high SNR values ranging from **22 dB to 30 dB**.
   - High-quality signals facilitate the model's learning process.

3. **Data Extraction:**
   - Extracted relevant samples from the dataset based on selected modulation classes and SNR values.
   - Reshaped the data to `(samples, 32, 32, 2)` for compatibility with CNN input requirements.

4. **Label Preparation:**
   - Converted one-hot encoded labels to a DataFrame.
   - Removed unused modulation class columns.
   - Ensured labels align with selected modulation classes.

5. **Train-Test Split:**
   - Split the dataset into training and testing sets using an 80-20 split.
   - Stratified sampling ensured balanced representation of each modulation class.

6. **Normalization:**
   - Applied normalization to the IQ components separately.
   - Ensured that data fed into the model during training and inference is on a similar scale.

### Model Architecture

The model combines CNN and LSTM layers to capture spatial and temporal features of the signals.

1. **Input Layers:**
   - Separate input layers for the **I** (In-phase) and **Q** (Quadrature) components.

2. **CNN Layers:**
   - Two parallel CNN paths for I and Q components.
   - Each path consists of convolutional layers with **LeakyReLU** activation and max-pooling layers.
   - These layers extract spatial features from the signal representations.

3. **Feature Concatenation:**
   - Flattened outputs from both CNN paths are concatenated.
   - This combines features from both I and Q components for comprehensive analysis.

4. **LSTM Layer:**
   - A single LSTM layer processes the combined features.
   - Captures temporal dependencies and sequential patterns in the data.

5. **Fully Connected Layers:**
   - Multiple dense layers with **LeakyReLU** activation.
   - Dropout layers with a rate of **0.5** are used for regularization.
   - Reduces the risk of overfitting by preventing co-adaptation of neurons.

6. **Output Layer:**
   - A final dense layer with **softmax** activation.
   - Outputs probability distributions over the modulation classes.

### Model Training

- **Optimizer:** Adam optimizer with a learning rate of **0.0001**.
- **Loss Function:** Categorical cross-entropy for multi-class classification.
- **Batch Size:** 64 samples per batch.
- **Epochs:** Set to 25 with early stopping based on accuracy improvements.
- **Callbacks:**
  - **EarlyStopping:** Monitors training to prevent overfitting.
  - **ModelCheckpoint:** Saves the best model weights during training.

### Evaluation Metrics

- **Accuracy:** Monitored during training and validation phases.

- **Confusion Matrix:** Visualized to assess model performance on individual classes.

- **Loss Curves:** Plotted to observe convergence and detect overfitting.


## Results

### Training and Validation Performance

- **Training Accuracy:** Achieved over **90%** accuracy on the training set.
- **Validation Accuracy:** Reached **93%** accuracy on the validation set.
- **Loss Reduction:** Consistent decrease in training and validation loss over epochs.

The model demonstrates strong learning capability and generalization to unseen data within the selected modulation classes and SNR range.

### Confusion Matrix Analysis

The confusion matrix indicates:

- **High True Positive Rates:** Most modulation classes are correctly classified.
- **Minimal Confusion:** Few misclassifications occur, primarily among modulation types with similar characteristics.
- **Strong Performance on Key Classes:** Critical modulation types like **QPSK** and **16QAM** show high recognition rates.

### Signal Classification

The model successfully classifies new signals by:

- **Preprocessing Input Signals:** Normalizing and reshaping them as during training.
- **Predicting Modulation Types:** Outputting probability distributions over the modulation classes.
- **Recognizing Modulations:** Correctly identifying the true modulation type in most cases.



Example output:

```
True Modulation Type: QPSK
Predicted Modulation Type: QPSK
Prediction Probabilities: [[0.0012, 0.0025, 0.9754, 0.0051, 0.0042, 0.0038, 0.0050, 0.0028]]
The model correctly recognized the modulation type!
```

## Conclusion

The project demonstrates that a deep learning model combining CNN and LSTM layers can effectively classify signal modulation types from raw IQ data. By focusing on high SNR signals and a subset of modulation classes, the model achieves high accuracy and robust performance.

Key takeaways:

- **Deep Learning Efficacy:** Neural networks can learn complex patterns in signal data without manual feature engineering.
- **Data Quality Matters:** High-quality, high SNR data significantly enhances model performance.
- **Model Architecture:** Combining spatial and temporal feature extraction is beneficial for signal classification tasks.
- **Generalization Capability:** The model generalizes well to new, unseen data within the trained modulation classes.

## Future Work

- **Expand Modulation Classes:** Incorporate more modulation types to increase the model's applicability.
- **SNR Diversity:** Train the model on a wider range of SNR values to improve robustness against noise.
- **Real-World Data:** Test and fine-tune the model using real-world signal data with varying conditions.
- **Optimize Model Efficiency:** Explore model compression techniques for deployment on resource-constrained devices.

## Repository and Usage

The complete code and instructions are available on GitHub:

**Repository:** [Signal Modulation Classification](https://github.com/tanrivertarik/Signal-Modulation-Classification-Using-Deep-Learning)

### How to Use

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/tanrivertarik/Signal-Modulation-Classification-Using-Deep-Learning.git
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the Dataset:**
   - Due to the dataset's size, download it directly from [RadioML Datasets](https://www.deepsig.ai/datasets).
   - Place the `GOLD_XYZ_OSC.0001_1024.hdf5` file in the project directory.

4. **Run the Training Script:**
   ```bash
   python train_model.py
   ```
5. **Classify a Signal:**
   - Use the `classify_signal.py` script to classify individual signals.
   ```bash
   python classify_signal.py --modulation QPSK --signal_path path_to_signal.npy
   ```

## Acknowledgements

- **DeepSig Inc.:** For providing the RadioML 2018.01A dataset.
- **OpenAI's ChatGPT:** For assistance in code development and problem-solving during the project.

## References

- O'Shea, T. J., & West, N. (2017). **"Radio Machine Learning Dataset Generation with GNU Radio."** Proceedings of the GNU Radio Conference, 1(1).
- **RadioML Datasets:** [DeepSig](https://www.deepsig.ai/datasets)
- **TensorFlow Documentation:** [https://www.tensorflow.org/](https://www.tensorflow.org/)
- **Keras Documentation:** [https://keras.io/](https://keras.io/)

