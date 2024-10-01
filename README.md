# Brain MRI Metastasis Segmentation

## Overview

This project implements Nested U-Net and Attention U-Net architectures for brain metastasis segmentation using brain MRI images. The goal is to accurately identify and segment metastases in brain images to assist in diagnosis and treatment planning.

## Nested U-Net Architecture

The Nested U-Net architecture enhances the traditional U-Net by introducing a nested skip pathway structure, which allows the model to learn features at various resolutions. This design improves the propagation of context information throughout the network and helps in accurately segmenting complex structures like brain metastases.

### How Nested U-Net Applies to Metastasis Segmentation

In the context of brain MRI segmentation, Nested U-Net can effectively capture multi-scale features, which is critical for identifying varying sizes and shapes of metastases. The architecture's ability to integrate features from different layers aids in enhancing the segmentation quality, particularly in cases where metastases may appear faint or irregular.

## Attention U-Net Architecture

Attention U-Net extends the U-Net architecture by incorporating attention mechanisms, allowing the model to focus on relevant features while ignoring irrelevant ones. This is particularly beneficial in medical image segmentation where the foreground (tumor) and background (healthy tissue) can vary significantly.

### Challenges with Attention U-Net

During implementation, we encountered issues with shape mismatches when merging layers, which hindered the model from training successfully. As a result, we could not achieve a functional implementation of Attention U-Net for this project.

## Project Setup

### Prerequisites

- Python 3.7+
- Anaconda or virtual environment
- Required libraries (install with pip):

```bash
pip install numpy pandas tensorflow fastapi uvicorn streamlit Pillow ```

clone repo.

```bash
git clone https://github.com/your_username/your_repository.git
cd your_repository
```

Set up the environment:

```bash
conda create -n brain_mri_segmentation python=3.7
conda activate brain_mri_segmentation
pip install -r requirements.txt  # If you have a requirements.txt
```

run fast api

``` bash
uvicorn app:app --reload
```
run streamlit app
```bash
streamlit run streamlit_app.py
```


## Challenges in Brain Metastasis Segmentation
# Brain metastasis segmentation presents unique challenges, including:

Variability in Tumor Appearance: Metastases can vary significantly in size, shape, and intensity, making it difficult for standard segmentation techniques to be universally effective.
Class Imbalance: There may be a significant imbalance between healthy tissue and tumor regions, which can lead to model bias if not addressed.
Noisy Data: MRI scans may contain artifacts and noise, affecting the segmentation accuracy.

# Addressing Challenges
Data Augmentation: The training process incorporates data augmentation techniques to create a more robust dataset, helping the model generalize better to unseen data.
Nested U-Net Architecture: By utilizing the Nested U-Net, the model can capture multi-scale features, improving the segmentation performance on complex tumor structures.
Evaluation Metrics: The Dice Score is employed as the primary metric for assessing segmentation accuracy, which is particularly relevant for imbalanced datasets.
## Conclusion
This project demonstrates the application of advanced deep learning architectures for the segmentation of brain metastases in MRI images. Although the Attention U-Net implementation faced challenges, the Nested U-Net showed promise in accurately segmenting metastases.

Feel free to contribute to this project by reporting issues or suggesting improvements.
