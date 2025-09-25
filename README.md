# MammoFlow: End-to-End Mammogram Segmentation & Classification
---
## Overview

MammoFlow is a comprehensive deep learning pipeline for automated mammogram analysis that provides end-to-end solutions for breast cancer detection. The system combines advanced image segmentation techniques with deep learning classification models to assist in early breast cancer diagnosis.

## Key Features

- **Image Segmentation**: Used scikit-image `regionprops` techniques and `opening` morphological operations for tumor region extraction
- **Data Augmentation**: Comprehensive augmentation pipeline with rotation and flipping
- **Multi-Class Classification**: Support for 8-class mammogram classification using VGG16 architecture
- **Data Balancing**: `SMOTE` implementation for handling imbalanced datasets
- **Comprehensive Visualization**: Detailed visualization suite for results analysis
- **Performance Metrics**: Complete evaluation with confusion matrix, accuracy, precision, recall, and F1-score

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-enabled GPU (recommended for training)
- 8GB+ RAM

### Dependencies Installation

```bash
# Clone the repository
git clone https://github.com/shakifahmed/mammoflow-end-to-end-mammogram-segmentation-classification.git
cd mammoflow-end-to-end-mammogram-segmentation-classification

# Install required dependencies
pip install -r requirements.txt
```

### Key Dependencies

- **TensorFlow**: 2.20.0 (Deep learning framework)
- **OpenCV**: 4.12.0.88 (Image processing)
- **scikit-learn**: 1.7.2 (Machine learning utilities)
- **imbalanced-learn**: 0.14.0 (SMOTE for data balancing)
- **pandas**: 2.3.2 (Data manipulation)
- **numpy**: 2.3.3 (Numerical computing)
- **matplotlib & seaborn**: Visualization libraries
- **imutils**: 0.5.4 (Image processing utilities)

## Dataset Structure

Organize your mammogram dataset in the following structure:

```
images/
├── Density1Benign/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── Density2Benign/
│   └── ...
├── Density3Benign/
│   └── ...
├── Density4Benign/
│   └── ...
├── Density1Malignant/
│   └── ...
├── Density2Malignant/
│   └── ...
├── Density3Malignant/
│   └── ...
└── Density4Malignant/
    └── ...
```

The system supports 8 classes representing different breast density levels with benign/malignant classifications.

## Usage

### Quick Start

Run the complete pipeline with a single command:

```bash
python main.py
```

This will execute the entire workflow:
1. Directory processing and segmentation
2. Data augmentation
3. Data loading and preprocessing
4. Model training
5. Performance evaluation
6. Visualization generation
7. Model saving

## Project Structure

```
mammoflow-end-to-end-mammogram-segmentation-classification/
├── main.py                 # Main execution pipeline
├── segmentation.py         # Image segmentation module (not shown but referenced)
├── augmentation.py         # Data augmentation utilities
├── loader.py               # Data loading and preprocessing
├── model.py                # Deep learning model implementation
├── visualization.py        # Visualization and plotting tools
├── load_directory.py       # Directory processing utilities
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
├── .gitignore              # Git ignore rules
├── images                  # Datset
└── visualization/          # Generated visualization outputs
```

## Performance Metrics

The system evaluates model performance using:

- **Accuracy**: Overall classification accuracy
- **Precision**: Macro-averaged precision across all classes
- **Recall**: Macro-averaged recall (sensitivity)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed class-wise performance analysis

## Output Files

After running the pipeline, the following files will be generated:

- `visualization/`: Various plots and analysis images
- `segmentation/`: Processed segmented images
- `augmentation/`: Augmented dataset
- `model/brcan_model.keras`: Trained model file

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Shakif Ahmed**
- GitHub: [@shakifahmed](https://github.com/shakifahmed)

---

**If you find MammoFlow helpful, please consider giving it a star on GitHub!**