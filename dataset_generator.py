import numpy as np
import pandas as pd

# setting seed using the last 3 digits of roll number 
np.random.seed(2) 

def make_low_noise_dataset():
    # minimum 1000 samples and 10 features rule 
    # we need at least 5 informative and 5 noisy features
    informative = np.random.normal(0, 1, (1000, 5))
    noise = np.random.normal(0, 1, (1000, 5))
    
    # putting them together
    x = np.hstack((informative, noise))
    
    # making labels based only on informative stuff
    y = (informative[:, 0] + informative[:, 1] > 0).astype(int)
    
    return x, y

def make_high_noise_imbalanced_dataset():
    # making it 5000 samples to pass the big dataset requirement
    informative = np.random.normal(0, 1, (5000, 5))
    noise = np.random.normal(0, 1, (5000, 10))
    
    x = np.hstack((informative, noise))
    
    # making labels but forcing imbalance where majority is >= 4x minority
    y = (informative[:, 0] > 1.5).astype(int) 
    
    # adding random noise to labels to make it a high noise dataset
    flip_idx = np.random.choice(len(y), size=1000, replace=False)
    y[flip_idx] = 1 - y[flip_idx]
    
    return x, y

def make_high_dimensional_dataset():
    # creating 1000 samples and 1000 noisy features like question 2 asks
    informative = np.random.normal(0, 1, (1000, 5))
    noise = np.random.normal(0, 1, (1000, 1000))
    
    x = np.hstack((informative, noise))
    y = (informative[:, 0] * informative[:, 1] > 0).astype(int)
    
    return x, y

if __name__ == "__main__":
    # saving them to test later
    x1, y1 = make_low_noise_dataset()
    x2, y2 = make_high_noise_imbalanced_dataset()
    x3, y3 = make_high_dimensional_dataset()
    print("datasets made successfully")
