''' Config Proto '''

import sys
import os


####### INPUT OUTPUT #######

# The name of the current model for output
log_base_dir = '/media/lenovo/SeagateExpansionDrive/deb/vloss/log/'
# name = 'sphere4_violate_m0.7_sc30'

# The folder to save log and model
name = 'face_classifier'

# The interval between writing summary
summary_interval = 100

# Training dataset path
train_dataset_path = os.environ["datasets"] + "/lfw/aligned"

# LFW dataset path
# IJB-A dataset folder
ijba_dataset_path = None #lfw_dataset_path # os.environ["DATABASES"] + "/FaceDatabases/Janus/ijba_mtcnnmat_aligned_sm"

# LFW standard protocol file
lfw_pairs_file = './proto/lfw_pairs.txt'

# Target image size for the input of network
image_size = [160, 160]

# 3 channels means RGB, 1 channel for grayscale
channels = 3

# Preprocess for training
preprocess_train = [
    # ['resize', (48,56)],
    # ['resize', (96,112)],
    ['random_flip'],
    # ['random_crop', (112,112)],
    # ['random_downsample', 0.5],
    ['standardize', 'mean_scale'],
]

# Preprocess for testing
preprocess_test = [
    # ['resize', (96,112)],
    # ['center_crop', (112, 112)],
    ['standardize', 'mean_scale'],
]

# Number of GPUs
num_gpus = 1


####### NETWORK #######

# Auto alignment network
localization_net = None

# The network architecture
network = "nets/sphere_net.py"

# Model version, only for some networks
model_version = "4"

# Number of dimensions in the embedding space
embedding_size = 512


####### TRAINING STRATEGY #######

# Optimizer
optimizer = ("MOM", {'momentum': 0.9})

# Number of samples per batch
batch_size = 256

# The structure of the batch
batch_format = 'random_samples'

# Number of batches per epoch
epoch_size = 100

# Number of epochs
num_epochs = 30

# learning rate strategy
learning_rate_strategy = 'step'

# learning rate schedule
lr = 0.001
learning_rate_schedule = {
    0:      1 * lr,
    16000:  0.1 * lr,
    24000:  0.01 * lr,
    28000:  0.001 * lr,
}

# Multiply the learning rate for variables that contain certain keywords
learning_rate_multipliers = {
    # 'ConditionalLoss/weights': ('MOM', 100.0)
    # 'SplitSoftmax/threshold_': 1.0,
    # 'BinaryLoss/weights': 100.,
    # 'LocalizationNet/': 1e-3,
}

# Restore model
restore_model = None

# Keywords to filter restore variables, set None for all
restore_scopes = ['SphereNet']

# Weight decay for model variables
weight_decay = 5e-4

# Keep probability for dropouts
keep_prob = 1.0


####### LOSS FUNCTION #######

# Scale for the logits
losses = {
    'softmax': {},
    # 'cosine': {'gamma': 'auto'},
    # 'angular': {'m': 4, 'lamb_min':5.0, 'lamb_max':1500.0},
    # 'split': {'gamma': 'auto', 'm': 0.7, "weight_decay": 5e-4},
    #'am_softmax': {'scale': 'auto', 'm': 5.0, 'alpha': 'auto'},
    # 'stochastic': {'coef_kl_loss': 1e-3},
    # 'norm': {'alpha': 1e-2},
    # 'triplet': {'margin': 1.0},
}

