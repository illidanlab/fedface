''' Config Proto '''

import sys
import os


####### INPUT OUTPUT #######

# The name of the current model for output
# name = 'triplet_m200_mu_eps1.0_lr1e-3_nodec'
# name = 'casia_uquadra64_m100_dim_avgbnrelubn_mom_bias-7_concat'
name = 'Baseline_64_layer'
# name = 'ms_rec_klstd1e-3'
# name = 'div_mom_lr1e-3'
# name = 'scratch_div_quadra_m100_samples_learnednorm'
# name = 'split_sigmoid'

# The folder to save log and model
log_base_dir = './log/'

# The interval between writing summary
summary_interval = 100

train_dataset_path = "./CASIA_WebFace/list_casia_mtcnncaffe_aligned_remove_lfw_ijbc_megaface.txt"
# Training dataset path
# train_dataset_path = os.environ["DATABASES"] + "/FaceDatabases/CASIA-Webface/list_casia_mtcnncaffe_aligned_train.txt"
# train_dataset_path = os.environ["DATABASES2"] + "/FaceDatabases/MsCeleb/list_msceleb5m_mtcnncaffe_aligned.txt"
# train_dataset_path = os.environ["DATABASES2"] + "/FaceDatabases/VGGFace2/vggface2_mtcnncaffe_aligned"
# train_dataset_path = os.environ["DATABASES2"] + "/FaceDatabases/cfp-dataset/list_cfp_aligned_cropped_labeled.txt"

# LFW dataset path
# lfw_dataset_path = os.environ["DATABASES"] + "/FaceDatabases/LFW/lfw_mtcnncaffe_aligned"

# CFP dataset path
# cfp_dataset_path = os.environ['DATABASES2'] + '/FaceDatabases/cfp-dataset/list_cfp_aligned_cropped.txt'
# cfp_proto_path = os.environ["DATABASES2"] + "/FaceDatabases/cfp-dataset/Protocol"

# IJB-A dataset folder
# ijba_dataset_path = None #lfw_dataset_path # os.environ["DATABASES"] + "/FaceDatabases/Janus/ijba_mtcnnmat_aligned_sm"

# Class stats file
# class_stats_path = None # './data/class_stats_casia_sphere4.npz'

# LFW standard protocol file
# lfw_pairs_file = './proto/lfw_pairs.txt'

# Target image size for the input of network
image_size = [96, 112]

# 3 channels means RGB, 1 channel for grayscale
channels = 3

# Preprocess for training
preprocess_train = [
    # ['resize', (48,56)],
    # ['resize', (112,112)],
    # ['center_crop', (112, 96)],
    # ['random_flip'],
    # ['random_blur', 'gaussian', 10],
    # ['random_blur', 'motion', 30],
    # ['random_crop', (112,112)],
    # ['random_downsample', 0.5],
    # ['random_noise', 0.3],
    ['standardize', 'mean_scale'],
]

# Preprocess for testing
preprocess_test = [
    # ['resize', (112,112)],
    # # ['center_crop', (112, 112)],
    ['center_crop', (112, 96)],
    ['standardize', 'mean_scale'],
]

# Number of GPUs
num_gpus = 1

####### NETWORK #######

# Auto alignment network
localization_net = None

# The network architecture
network = "nets/sphere_net_rec.py"

# Model version, only for some networks
model_version = '64'

# Number of dimensions in the embedding space
embedding_size = 512


####### TRAINING STRATEGY #######

# Optimizer
optimizer = ("MOM", {'momentum': 0.9})
# optimizer = ("ADAM", {'beta1': 0.5, 'beta2': 0.99})

# Base Random Seed
base_random_seed = 9


# learning rate strategy
learning_rate_strategy = 'step'

# learning rate schedule
lr = 0.01
learning_rate_schedule = {
    0:      1 * lr,
    # 8000:   0.1 * lr,
    # 12000:  0.01 * lr,
}

# Multiply the learning rate for variables that contain certain keywords
learning_rate_multipliers = None
# learning_rate_multipliers = {
#     'SphereNet': 0.0,
# }

# Restore model
# restore_model = './log_sphere64_ms_tune/ms_dpos_bn_sc_longer_seed9/20190225-234948/'

# Keywords to filter restore variables, set None for all
# restore_scopes = ['SphereNet'] #, 'SphereNet/Batch'] # ['ConditionalLoss'] # ['FaceResNet']

# Weight decay for model variables
weight_decay = 0e-4

# Keep probability for dropouts
keep_prob = 1.0


# Federated Learning Parameters

num_comm_rounds = 10000
num_local_epochs = 1
local_epoch_size = 1
local_batch_size = 256


# Baseline Learning Parameters
# batch_format = {
#     'size': 256,
#     'sampling': 'random_samples',
#     # 'num_classes': 64,
#     # 'cluster_size': 3,
# }
batch_format = 'random_samples'
batch_size = 128
epoch_size = 3232
num_epochs = 100000


####### LOSS FUNCTION #######

# Scale for the logits
losses = {
    # 'softmax': {},
    # 'cosine': {'gamma': 'auto'},
    'am_softmax': {'scale': 30.0, 'm':10.0},
    # 'angular': {'m': 4, 'lamb_min':5.0, 'lamb_max':1500.0},
    # 'split': {'gamma': 'auto', 'm': 2.5},
    # 'conditional': {'gamma': 'auto', 'm': 10.0, 'weight_decay': 0.0,
    #                 'alpha_decay': 0.0, 'coef_weights':1e-3},
    # 'utriplet': {'coef': 1.0, 'margin': 100.0},
    # 'divergence': {'coef': 1.00, 'alpha':1.0, 'margin':100.},
    # 'rec': {'coef': 10.},
    # 'dploss': {'m': 1.0},
    # 'kl_std': {'coef': 1e-3},
    # 'norm': {'alpha': 1e-4},
    # 'squared_hinge_loss' : {'scale':30.0, 'm':0.9}
}

