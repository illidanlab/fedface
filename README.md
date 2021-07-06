# FedFace
Experimental pipeline for FedFace.  

### Introduction
DNN-based face recognition models require large centrally aggregated face datasets for training. However, due to the growing data privacy concerns and legal restrictions, accessing and sharing face datasets has become exceedingly difficult. We propose FedFace, a federated learning (FL) framework for collaborative learning of face recognition models in a privacy aware manner. FedFace utilizes the face images available on multiple clients to learn an accurate and generalizable face recognition model where the face images stored at each client are neither shared with other clients nor the central host and each client is a mobile device containing face images pertaining to only the owner of the device (one identity per client). Our experiments show the effectiveness of FedFace in enhancing the verification performance of pre-trained face recognition system on standard face verification benchmarks namely LFW, IJB-A and IJB-C.

### Usage
