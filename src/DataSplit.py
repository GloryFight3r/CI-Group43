import numpy as np

class DataSet:
    def createData(self, val_percentage=0.1, test_percentage=0.1):
        features = np.genfromtxt("../data/features.txt", delimiter=",")
        targets = np.genfromtxt("../data/targets.txt", delimiter=",")
        #unknowns = np.genfromtxt("../data/unknown.txt", delimiter=",")

        train_split = 1 - val_percentage - test_percentage
        val_split = val_percentage
        #test_split = 

        size = features.shape[0]

        size_train = int(size * train_split)
        size_val = int(size * val_split)
        #size_test = size - (size_train + size_val)

        indices = np.random.permutation(size)

        train_features = features[indices[:size_train]]
        train_targets = targets[indices[:size_train]]
        val_features = features[indices[size_train:size_train+size_val]]
        val_targets = targets[indices[size_train:size_train+size_val]]
        test_features = features[indices[size_train+size_val:]]
        test_targets = targets[indices[size_train+size_val:]]
        
        return(train_features, train_targets, val_features, val_targets, test_features, test_targets)
