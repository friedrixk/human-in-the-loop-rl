
from math import ceil
import random
import numpy as np
import pandas as pd


class Sampler():
    def systematic_random_sampling(self, dataset, sample_length):
     """
     method performs systematic random sampling on a given dataset.

     :param dataset: the data from which we want only a subset/sample.
     :param sample_length: number of items that the sample should contain at most.
     :return: the sample that contains only some data points of the original dataset.
     """
     sample = []
     # number of data points N
     N = len(dataset)

     #if there are less than sample_length data points, the visualization includes all (no sampling needed)
     if N < sample_length:
        return dataset

     # take every k-th element
     k = ceil(N / sample_length)
     i = 0
     while i < N:
        sample.append(dataset[i])
        i = i + k
     return sample



    def stratified_sampling_return_values(self, returns, sample_length):
     """
     method performs stratified random sampling on a given dataset of return values.
     method is necessary because there are often too many return values to fit in one visualization.

     :param returns: the return values from which we want only a subset/sample.
     :param sample_length: number of items that the sample should contain at most.
     :return: the sample that contains at most 30 return values of the original dataset.
     """
     # if there are less than the desired number of returns, the visualization includes all (no sampling necessary)
     if len(returns) < sample_length:
         return returns

     sample = []
     n = sample_length                                                   # how many returns are shown in the visualization
     zero_return = list(filter(lambda x: x <= 0, returns))    # sampling group 1
     nonzero_return = list(filter(lambda x: x > 0, returns))  # sampling group 2
     
     # calculate percentages to ensure that sample has the same percentage of zero/ non-zero returns as the given data
     x = 100 / len(returns)
     percentage_zero = len(zero_return) * x
     percentage_nonzero = len(nonzero_return) * x
    
     n1 = round(percentage_zero * (n / 100))
     n2 = round(percentage_nonzero * (n / 100))
     
     # sample each group 
     for i in range(0, n1):
         if zero_return:
          sample.append(zero_return[i])

     for j in range(0, n2):
         if nonzero_return:
          sample.append(nonzero_return[j])
     
     random.shuffle(sample)
     return sample



    def sample_batchwise(self, df, batch_size, max_sample_length):
        """
        method samples rows of a dataframe. Each row represents an episode. Multiple rows/episodes belong to the same batch.
        It samples rows and ensures that all rows corresponding to the same batch are included.

        :param df: data frame that contains training data.
        :param batch_size: batch size used in training.
        :param max_sample_length: maximum number of rows in the sample.
        :return: sample/ subset of original data frame.
        """
        
        no_batches_contained = df.shape[0] / batch_size
        no_batches_needed = max_sample_length / batch_size
        
        # no sampling necessary
        if no_batches_contained <= no_batches_needed:
            return df
        
        batch_starting_indices = np.arange(0, df.shape[0], batch_size)

        # exclude incomplete batch 
        if df.shape[0] - batch_starting_indices[-1] < batch_size:
            batch_starting_indices = batch_starting_indices[:-1]

        starting_indices_sample = self.systematic_random_sampling(batch_starting_indices, no_batches_needed)
        
        indices = []
        for i in starting_indices_sample:
            indices_per_batch = np.arange(i, i + batch_size, 1)
            for j in indices_per_batch:
                indices.append(j)
               
        sample = df[df.columns].loc[indices]
        return sample