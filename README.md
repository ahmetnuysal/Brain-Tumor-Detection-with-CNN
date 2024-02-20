# Brain Tumor Detection with CNN

This project focuses on utilizing Convolutional Neural Networks (CNN) for the detection of brain tumors in MRI images. The developed model is implemented in Python using the TensorFlow and Keras libraries. The dataset consists of MRI images categorized into two classes: tumors (positive class) and non-tumors (negative class). The model is trained to classify these images based on their features.

## Project Structure
Data Collection
The dataset is organized into two main folders: 'yes' and 'no,' representing images with tumors and without tumors, respectively. The images are in JPEG format and are resized to 128x128 pixels during preprocessing.

Model Architecture
The CNN model is designed with the following layers:

Convolutional layers with 32 and 64 filters, respectively
Batch Normalization for better convergence
MaxPooling layers for downsampling
Dropout layers to prevent overfitting
Dense layers for classification with ReLU activation
Output layer with softmax activation for binary classification
The model is compiled using the Adamax optimizer and mean squared error as the loss function.

Training and Evaluation
The dataset is split into training and testing sets using an 80-20 ratio. The model is trained for 30 epochs with a batch size of 40. The training and validation loss are monitored during the training process. The evaluation includes calculating the error rate and root mean square error (RMSE).

Performance Visualization
Several visualizations are provided to assess the model's performance:

Loss Graph: Visualizing the training and validation loss over epochs.
Confusion Matrix: Displaying the true positive, true negative, false positive, and false negative predictions.
Correlation Matrix: Showing the correlation between true and predicted labels.
