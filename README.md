# Waste Product Classification using Deep Learning (CNN)

## Project Overview
This project utilizes deep learning techniques to classify waste products using Convolutional Neural Networks (CNN). Two models were trained: a custom CNN and a pre-trained ResNet18. A user-friendly web app integrates the trained models for easy usage.

## Acknowledgments
* Original code adapted from Kaggle user [arbazkhan971](https://www.kaggle.com/code/arbazkhan971/image-classification-using-cnn-94-accuracy/notebook), licensed under Apache License 2.0.
* Modified and adapted by Animashaun Tomisin for waste product classification.

## Key Features
* Custom CNN model for waste classification
* Pre-trained ResNet18 model for comparison
* Web app interface for model interaction
* Achieved 85.1% accuracy with custom CNN and 96.1% accuracy with ResNet18

## Technical Details
* Programming Language: Python
* Frameworks/Libraries: FastAi, NumPy, PyTorch
* Web App: Streamlit
* Model Training: Jupyter Notebook

## Model Performance
* Custom CNN model: 85.1% accuracy (0.1 test size)
* ResNet18 model: 96.1% accuracy
* Best performing model weights (custom CNN, 0.1 test size): [exportCNN10.pkl](https://drive.google.com/file/d/1ELrjuiTUX5V3c1vFdqgD0NQkaCDU2p1_/view?usp=drive_link)

## Repository Contents
* Custom CNN model code: [model](/model.py)
* Model training notebook: [waste-classification](/waste-classification.ipynb)
* Web app code: [model_ui](/model_ui.py)
* ResNet18 model weights:
  * [export10.pkl (0.1 test size)](/export10.pkl)
  * [export20.pkl (0.2 test size)](export20.pkl)
  * [export.pkl (0.3 test size)](export.pkl)

## Getting Started
* Clone the repository.
* Install required dependencies
* Run the web app

## Contributions
Contributions are welcome! Feel free to fork and extend the project.

## License
This project is licensed under the MIT License.
