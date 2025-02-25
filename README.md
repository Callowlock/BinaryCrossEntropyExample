# Binary Cross-Entropy: Evaluating Model Performance on Balanced and Imbalanced Datasets

This project demonstrates the use of **binary cross-entropy (BCE)** for training neural networks on synthetic datasets, exploring how different model configurations and dataset imbalances impact performance. It also shows how weighted BCE can improve results.

## Key Techniques and Tools

- **Data Generation**: Created synthetic datasets using `sklearn.datasets.make_classification`.
- **Dimensionality Reduction**: Used **PCA** to visualize high-dimensional data in 2D.
- **Modeling**: Built neural network models using **TensorFlow** and **Keras** with varying architectures and batch sizes.
- **Evaluation**: Assessed model performance using:
  - **Confusion matrices**
  - **Classification reports** (precision, recall, F1-score)
- **Class Imbalance Handling**: Implemented **weighted BCE** to address issues with imbalanced datasets.

## Key Findings

1. Models performed well on balanced datasets but poorly on imbalanced datasets.
2. Weighted BCE significantly improved performance on imbalanced datasets by balancing precision and recall.
3. Smaller batch sizes improved model accuracy.

## Visualizations

- **PCA plots** to visualize class separation.
- **Confusion matrices** to illustrate model predictions.

## Dependencies

- Python 3.8+
- NumPy, Pandas, scikit-learn
- TensorFlow, Matplotlib, Seaborn

Install packages using:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn
