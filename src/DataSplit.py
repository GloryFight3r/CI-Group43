import numpy as np

class DataSet:
    @staticmethod
    def createData(features, targets, val_percentage=0.1):
        train_split = 1 - val_percentage
        val_split = val_percentage

        size = features.shape[0]

        size_train = int(size * train_split)
        size_val = int(size * val_split)

        indices = np.random.permutation(size)

        train_features = features[indices[:size_train]]
        train_targets = targets[indices[:size_train]]
        val_features = features[indices[size_train:size_train+size_val]]
        val_targets = targets[indices[size_train:size_train+size_val]]
        
        return(train_features, train_targets, val_features, val_targets)
