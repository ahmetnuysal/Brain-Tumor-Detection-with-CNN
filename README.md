# Brain Tumor Detection with CNN

This project focuses on detecting brain tumors in MRI images using a Convolutional Neural Network (CNN). Developed in Python using the TensorFlow and Keras libraries, the model is trained on a dataset divided into two classes: images with tumors (positive class) and images without tumors (negative class). The model is trained to perform classification based on the features of these images.

## Project Structure

### Data Collection
The dataset has two main folders named 'yes' and 'no', representing images with and without tumors, respectively. The images are in JPEG format and resized to 128x128 pixels during pre-processing.

### Model Architecture
The CNN model consists of the following layers:

- Convolution layers with 32 and 64 filters
- Batch Normalization for better convergence
- MaxPooling layers for dimensional reduction
- Dropout layers to prevent over-learning
- Dense layers for classification with ReLU activation
- Output layer with softmax activation for binary classification

The model is compiled using the Adamax optimizer and the mean squared error is used as the loss function.

#### Training and Evaluation

The dataset is split with 80% training and 20% testing. The model is trained with a group size of 40 for 30 epochs. Training and validation losses are monitored during the training process. Evaluation includes error rate and root mean square error (RMSE) calculations.

#### Performance Visualization

Several visualizations are presented to evaluate the performance of the model:

- **Loss Graph:** Visualizing training and validation losses across epochs.
- Confusion Matrix:** Showing true positive, true negative, false positive and false negative predictions.
- Correlation Matrix:** Show the correlation between actual and predicted tags.

## Use Case

#### Single Image Classification
To classify a single image, specify the path to the image:

```
from matplotlib.pyplot import imshow
img = Image.open(r "C:/Users/ahmet/Desktop/data/all/degergiriniz.jpg")
img = img.convert("RGB")
x = np.array(img.resize((128, 128)))
x = x.reshape(1, 128, 128, 3)
res = model.predict_on_batch(x)
classification = np.where(res == np.amax(res))[1][0]
imshow(img)
print(str(res[0][classification] * 100) + '% Confidence This Is A ' + names(classification))
```
![]()
>[]()
