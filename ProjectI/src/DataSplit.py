import numpy as np

class DataSet:

    def __init__(self, test_percentage):
        features = np.genfromtxt("../data/features.txt", delimiter=",")
        targets = np.genfromtxt("../data/targets.txt", delimiter=",")
        #unknowns = np.genfromtxt("../data/unknown.txt", delimiter=",")

        train_split = 1  - test_percentage

        size = features.shape[0]

        size_train = int(size * train_split)

        indices = np.random.permutation(size)

        self.train_features = features[indices[:size_train]]
        self.train_targets = targets[indices[:size_train]]
        self.test_features = features[indices[size_train:]]
        self.test_targets = targets[indices[size_train:]]


    def split_for_cross(self, k_folds):
        size_of_split = self.train_features.shape[0] / k_folds
        size_train = int(size_of_split)
        entry = []
        for i in range(k_folds):
            val_set_f = self.train_features[(size_train * i):(size_train * (i + 1))]
            val_set_t = self.train_targets[(size_train * i):(size_train * (i + 1))]
            train_set_f = np.concatenate((self.train_features[:(size_train * i)], self.train_features[(size_train * (i + 1)):]))
            train_set_t = np.concatenate((self.train_targets[:(size_train * i)], self.train_targets[(size_train * (i + 1)):]))
            entry.append((train_set_f, train_set_t, val_set_f, val_set_t))
        return entry
        

