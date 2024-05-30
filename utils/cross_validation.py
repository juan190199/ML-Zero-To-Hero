import numpy as np


class KFoldCV:
    """
    Provides train/test indices to split data in train/test sets.
    Split dataset into k consecutive folds;
    each fold is then used once as a validation while the k-1 remaining folds form the trraining set
    """

    def __init__(self, n_folds, shuffle=True, seed=12345):
        self.seed = seed
        self.shuffle = shuffle
        self.n_folds = n_folds

    def split(self, data_set):
        """
        Create mask for test set.
        :param data_set: data set
        :return:
        """
        # Shuffle modifies indices inplace
        n_samples = data_set.shape[0]
        indices = np.arange(n_samples)
        if self.shuffle:
            # A fixed seed and a fixed series of calls to ‘RandomState’ methods using the same parameters
            # will always produce the same results
            rstate = np.random.RandomState(self.seed)
            rstate.shuffle(indices)

        for test_mask in self._iter_test_masks(n_samples, indices):
            train_index = indices[np.logical_not(test_mask)]
            test_index = indices[test_mask]
            yield train_index, test_index

    def _iter_test_masks(self, n_samples, indices):
        """
        Create test mask
        :param n_samples:
        :param indices:
        :return:
        """
        # If n_samples cannot be evenly split,
        # rest of samples have to be distributed between folds beginning with the leading one
        fold_sizes = (n_samples // self.n_folds) * np.ones(self.n_folds, dtype=np.int_)
        fold_sizes[:n_samples % self.n_folds] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            test_mask = np.zeros(n_samples, dtype=np.bool_)
            test_mask[test_indices] = True
            yield test_mask
            current = stop
