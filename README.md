# 🎬 **Face Recognition Projects** 🎥

Welcome to my **Face Recognition Projects** repository! This repository contains multiple face recognition systems based on various techniques, including simple face detection, LBPH face recognition, and a deep learning-based model for recognizing characters from *The Simpsons*.

---

## 📚 **Projects Overview**

### 1. **Simple Face Detection 🏞**

This project demonstrates how to detect faces in images using OpenCV and Haar Cascades. The model is designed to locate faces within an image and highlight them with rectangles.

#### 📖 **How to Use**:
- **Install Dependencies**: 
  To run the face detection system, you need OpenCV. Install it using:
  ```bash
  pip install opencv-python
  ```
- **Haar Cascade File**:
  The `haar_face.xml` file, used for detecting faces, can be found in the project directory or you can download it from OpenCV's official repository.
- **Run the Code**:
  After setting up, simply run the Python script that will load an image, detect faces, and display the result with rectangles drawn around detected faces.

### 2. **LBPH Face Recognition 🤖**

This project is based on the **Local Binary Pattern Histogram (LBPH)** face recognizer to recognize faces. The system trains on a dataset of various people's images and then identifies the faces in new images with confidence scores.

#### 📖 **How to Use**:
- **Install Dependencies**:
  Install the required packages, such as OpenCV and NumPy, using:
  ```bash
  pip install opencv-python numpy
  ```
- **Training the Model**:
  To train the model, run the `recognizer.py` script, which processes the images in the `Faces` directory and trains the LBPH recognizer. The model is saved as `face_trained.yml`.
- **Recognizing Faces**:
  Once the model is trained, use the `recongize.py` script to predict the identity of faces in new images. The script will load the trained model and use it to identify the faces, printing out the name and confidence score of the prediction.

### 3. **Simpsons Character Recognition 🎬**

In this project, we use a deep learning model (CNN) to recognize characters from *The Simpsons* TV show. The model is trained on a dataset of labeled images for each character and is then used to predict the character in new images.

#### 📖 **How to Use**:
- **Install Dependencies**:
  You’ll need to install TensorFlow, Keras, OpenCV, and other dependencies before running the model:
  ```bash
  pip install tensorflow keras opencv-python numpy caer
  ```
- **Dataset**:
  The dataset for the Simpsons character recognition can be downloaded from [this link](https://github.com/chandraparsad3/Face-Recognition-Project/tree/master/Simpsons/Dataset). The dataset contains images of characters like Homer, Bart, and Lisa. Make sure it is properly organized with subfolders for each character.
- **Running the Notebook**:
  Open the `Simpsons.ipynb` file in a Jupyter notebook or compatible IDE. The notebook contains the following steps:
  - **Preprocessing the dataset**: This step resizes and normalizes the images.
  - **Building the CNN model**: A Convolutional Neural Network is used to classify the images.
  - **Training the model**: The CNN is trained on the dataset.
  - **Testing the model**: After training, the model can be used to predict Simpsons characters in new images.

---

## 🛠 **Technologies Used**:
- **Python** 🐍
- **OpenCV** 🖼 (for face detection and recognition)
- **TensorFlow/Keras** 📊 (for building the CNN)
- **NumPy** 🔢 (for array manipulation)
- **Haars Cascade Classifier** 🟢 (for face detection)

---

## 👨‍💻 **Contributors**:
- [Chandra Parsad](https://github.com/chandraparsad3) (Author)

---

## 💬 **Feedback & Issues**:
If you have any issues, feel free to open an issue or submit a pull request. Feedback is always welcome! 😊
