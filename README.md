# Fashion MNIST Image Classification using CNN and Deep Learning

This project involves the classification of Fashion MNIST images using a Convolutional Neural Network (CNN) model built with TensorFlow/Keras. Additionally, a Streamlit web application is provided to allow users to upload images and classify them into one of the fashion categories.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Web Application](#web-application)
- [Dockerization](#dockerization)
- [How to Run](#how-to-run)
- [Screenshots](#screenshots)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [Demo Link](#demolink)
- [License](#license)

## Project Overview
The goal of this project is to classify images from the Fashion MNIST dataset into one of 10 categories, such as T-shirt/top, Trouser, Pullover, Dress, etc. A Convolutional Neural Network (CNN) model is trained on the dataset, achieving high accuracy on both the training and test sets. The trained model is then used in a Streamlit web application, where users can upload their own fashion images for classification.

## Dataset
The Fashion MNIST dataset contains 70,000 grayscale images in 10 categories. The images are of size 28x28 pixels:
- 60,000 images are used for training.
- 10,000 images are used for testing.

The dataset is publicly available and is commonly used as a benchmark in machine learning.

## Model Architecture
The CNN model consists of the following layers:
1. **Conv2D Layer:** 32 filters, 3x3 kernel, ReLU activation
2. **MaxPooling2D Layer:** 2x2 pool size
3. **Conv2D Layer:** 64 filters, 3x3 kernel, ReLU activation
4. **MaxPooling2D Layer:** 2x2 pool size
5. **Conv2D Layer:** 64 filters, 3x3 kernel, ReLU activation
6. **Flatten Layer**
7. **Dense Layer:** 64 units, ReLU activation
8. **Dense Layer:** 10 units (output layer, one for each class)

The model is compiled with the Adam optimizer and Sparse Categorical Crossentropy loss function. It is trained for 5 epochs.

## Web Application
A Streamlit web application allows users to upload images and classify them using the trained model. The application includes a visually appealing UI with custom CSS for a gradient background and styled buttons.

## Dockerization
A `Dockerfile` is provided to containerize the web application. The Docker container is based on the Python 3.10-slim image, with necessary dependencies installed for TensorFlow, Streamlit, and image processing. The application is exposed on port 8501.

## How to Run
### Clone the Repository
```bash
git clone https://github.com/Pahinithi/Fashion-MNIST-Image-Classification-CNN-Deep-Learning
cd Fashion-MNIST-Image-Classification
```

### Run with Docker
1. **Build the Docker image:**
   ```bash
   docker build -t fashion-mnist-app .
   ```
2. **Run the Docker container:**
   ```bash
   docker run -p 8501:8501 fashion-mnist-app
   ```
3. Open your web browser and go to `http://localhost:8501` to access the app.

### Run Locally
1. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Streamlit app:**
   ```bash
   streamlit run main.py
   ```
3. Open your web browser and go to `http://localhost:8501` to access the app.

## Screenshots

<img width="1728" alt="DL08" src="https://github.com/user-attachments/assets/70f1d410-6eb8-4e5e-b071-d003e5df80fb">


You can view the model's accuracy and loss plots in the notebook provided.

## Technologies Used
- **TensorFlow/Keras** for building the CNN model.
- **Streamlit** for creating the web application.
- **Docker** for containerizing the application.
- **Python** for scripting and development.

## Contributing
If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are welcome.

## Demo 

## License
This project is licensed under the MIT License.
