# Parallel CNN Aggregation (with unparalleled version) of CIFAR Dataset

A comprehensive framework for training convolutional neural networks (CNNs) on the CIFAR-10 dataset, featuring both a single-node standalone implementation for demonstration and a parallelized approach utilizing rank-based data processing and weight aggregation. The unparalleled version serves as an example to establish baseline performance, while the parallel version showcases efficient scaling and consistency across multiple nodes through distributed training.

## Features
- **Distributed Training**: Enables CNN training across multiple ranks using parallel data processing.
- **Rank-Based Data Splitting**: Allocates training data evenly among ranks for balanced workload distribution.
- **Global Weight Aggregation**: Synchronizes model weights across ranks to maintain consistency.
- **Scalability**: Optimized for high-performance computing environments with multiple nodes.

## Requirements
- Python 3.8+
- Required libraries for both unparalleled and parallel versions:
    - `mpi4py`: For MPI communication between nodes.
    - `tensorflow`: For building and training CNN models.
    - `keras`: For high-level neural network building blocks (part of TensorFlow).
    - `numpy`: For numerical operations and data manipulation.
    - `pickle`: For loading and saving the CIFAR-10 dataset.
    - `pandas`: For dataset handling (if used for data preprocessing).
    - `matplotlib`: For visualizing training results or datasets.
    - `sklearn`: Specifically, `shuffle` function from `sklearn.utils` for shuffling data.

## Dataset
The CIFAR-10 dataset is not included in this repository due to size constraints. You can download it [here](https://www.cs.toronto.edu/~kriz/cifar.html) and extract it in the project root.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/username/parallel-CNN-aggregation.git
   cd parallel-CNN-aggregation

2.	Create and activate a virtual environment (optional but recommended).

3. 	Install the required libraries:
    ```bash
    pip install mpi4py tensorflow pandas matplotlib scikit-learn

4.	Prepare the CIFAR-10 dataset:
	- Ensure the cifar-10-python.tar.gz file is in the root directory.
	- Extract the dataset:
    ```bash
    tar -xzf cifar-10-python.tar.gz

5.	Verify MPI installation:
	- Ensure mpirun or mpiexec is installed on your system. You can check by running:
    ```bash
    mpirun --version

6.	Run the program:
    - For the parallel version using MPI:
    ```bash
    mpirun -n <number_of_ranks> python parallel-CNN-aggregation.py
    ```
    - The unparallized version can be run in the Jupyter notebook.
    - The cnn.slurm file is provided to be run on a supercomputer. This option is recommended as the CNN aggregation was designed to be run using HPC.

## Known Issues
- **Parallel Version Accuracy**: While the training is distributed effectively across nodes, the final model accuracy may be lower than expected. This is due to limitations in weight aggregation techniques, which can lead to inconsistencies in the global model.
- The unparalleled version provides a more reliable accuracy metric and should be used as a benchmark for evaluation.

## Explanation of Model Behavior
The parallel version is designed to demonstrate distributed training capabilities. It effectively splits data across ranks, trains local models, and aggregates weights globally. However, the weight aggregation process may introduce inconsistencies, leading to suboptimal model accuracy. This is a common challenge in distributed deep learning frameworks and highlights the trade-off between scalability and accuracy.