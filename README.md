# Image Segmentation Model

This repository contains the implementation of an image segmentation model using the U-Net architecture.

## Objective
The goal of this project is to accurately segment different regions within images using a machine learning or deep learning approach.

## Dataset
The COCO dataset was used for this project, focusing on various everyday objects.

## Model Development
A U-Net model was built and trained on the dataset. The model uses convolutional and upsampling layers to segment images effectively.

## Usage
1. **Clone the repository:**
    ```sh
    git clone <repository-url>
    cd <repository-name>
    ```

2. **Install dependencies:**
    ```sh
    pip install tensorflow numpy matplotlib scikit-learn
    ```

3. **Download and extract the dataset:**
    - Place the dataset in the appropriate folder as specified in the code (`C:\Users\shrey\Downloads\val2017\val2017`).

4. **Run the training script:**
    ```sh
    python shreya_pandey_image_segmentation.py
    ```

## Results
The model achieved a validation accuracy of approximately 50%, indicating room for improvement with real segmentation masks.

## Future Work
- Use actual segmentation masks for training.
- Experiment with more advanced architectures and hyperparameter tuning.

