from mpi4py import MPI
import pickle
import numpy as np
from keras import layers, models
from keras.utils import to_categorical
from sklearn.utils import shuffle

# initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# base path for CIFAR-10 dataset
base_path = "/lustre/home/user/final_project/cifar-10-batches-py"

# function to load a single batch
def load_batch(file_path):
    with open(file_path, 'rb') as file:
        batch = pickle.load(file, encoding='bytes')
        data = batch[b'data']
        labels = batch[b'labels']
        return data, labels

# function to load and preprocess the entire dataset
def load_and_prepare_data(base_path):
    x_train, y_train = [], []
    for i in range(1, 6):  # loads all 5 training batches
        file_path = f"{base_path}/data_batch_{i}"
        data, labels = load_batch(file_path)
        x_train.append(data)
        y_train += labels
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.array(y_train)
    
    # load testing data
    x_test, y_test = load_batch(f"{base_path}/test_batch")
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    # reshape and normalize
    x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype('float32') / 255.0
    
    # one-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return x_train, y_train, x_test, y_test

# this function shuffles the dataset and selects a random subset of data points, 
# and it ensures that the number of selected points doesn’t exceed the total available samples
def split_data_among_ranks(x_train, y_train, num_samples_per_rank):
    # shuffle the entire dataset
    # random_state makes sure it is reproducible if it run mroe than once with the rank
    x_train, y_train = shuffle(x_train, y_train, random_state=rank)

    # calculate the total number of samples in the training data set
    # x_train.shape[0] is the number of rows, or total data points

    total_samples = x_train.shape[0]

    # check if the requested number of samples per rank exceeds the total amount of sample
    if num_samples_per_rank > total_samples:
        num_samples_per_rank = total_samples  # prevent oversampling and avoid errors with cnn
    
    # randomly seleect the correct amount of samples from [0, total_samples - 1]
    # replace is false, so there is no multiple selections in one sampling
    indices = np.random.choice(total_samples, num_samples_per_rank, replace=False)
    
    # return the trainign data for that rank
    return x_train[indices], y_train[indices]

# define the CNN model
def define_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')  # 10 classes
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# aggregate model weights across ranks
def aggregate_weights(local_weights):
    # global weights list initialized to zeros with the same shape as the local weights
    global_weights = [np.zeros_like(w) for w in local_weights]

    # loop over each layer's weights
    for i, local_weight in enumerate(local_weights):
        # get weights from all ranks, sums them, and stores in global_weights[i]
        comm.Allreduce(local_weight, global_weights[i], op=MPI.SUM)
        
        # the sum of all the weights is divided by the number of ranks to compute the weight
        global_weights[i] = global_weights[i] / size

    # validate shapes to ensure consistency
    # does so by by comparing the shapes of local_weights to global_weights for each layer
    for i, (local_w, global_w) in enumerate(zip(local_weights, global_weights)):
        assert global_w.shape == local_w.shape, f"Shape mismatch at layer {i}"
    
    return global_weights 

# main script entry point
if __name__ == "__main__":
    # constants for the number of samples per rank and test samples
    NUM_SAMPLES_PER_RANK = 30000  # training samples per rank
    NUM_TEST_SAMPLES = 6000  # testing samples

    # load the dataset on rank 0
    if rank == 0:
        x_train_full, y_train_full, x_test_full, y_test_full = load_and_prepare_data(base_path)
        x_test, y_test = x_test_full[:NUM_TEST_SAMPLES], y_test_full[:NUM_TEST_SAMPLES]  # select test samples
    else:
        # ranks other than 0 do not need to load the full dataset
        # they will get the data via comm.bcast
        x_train_full, y_train_full, x_test, y_test = None, None, None, None  # initialize as none for other ranks

    # broadcast test data to all ranks to make sure each rank has access to the same test data for validation
    x_test = comm.bcast(x_test, root=0)
    y_test = comm.bcast(y_test, root=0)

    # broadcast training data to all ranks
    x_train_full = comm.bcast(x_train_full, root=0)
    y_train_full = comm.bcast(y_train_full, root=0)
    
    # splits the assigned portion of data to the ranks
    x_train, y_train = split_data_among_ranks(x_train_full, y_train_full, NUM_SAMPLES_PER_RANK)  # split training data among ranks

    # define the cnn model
    model = define_model()

    # check if rank has training data
    if x_train.shape[0] == 0:
        print(f"Rank {rank}: No training data. Skipping training.")
    else:
        # train the model locally on the assigned training data
        model.fit(x_train, y_train, epochs=4, batch_size=64, validation_data=(x_test, y_test), verbose=2)

    # evaluate the local model and log loss and accuracy for each rank
    local_loss, local_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Rank {rank}: Local Model - Loss: {local_loss:.4f}, Accuracy: {local_accuracy:.4f}")

    # gather local weights from the model 
    local_weights = model.get_weights()

    # set global model weights equal to the average weights of each locally trained model
    global_weights_1 = aggregate_weights(local_weights)
   
    # update the global model with the aggregated weights
    model.set_weights(global_weights_1)

    # evaluate the global model on rank 0
    if rank == 0:
        loss, accuracy = model.evaluate(x_test, y_test)
        print(f"Global Model with Regular Agregated Weights - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")




