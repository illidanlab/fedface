"""
Main file for evaluation on IJB-A and IJB-B protocols.
More instructions can be found in README.md file.
2017 Yichun Shi
"""

import sys
import os
import numpy as np

from facepy import metric
from collections import namedtuple
import pdb
from tqdm import tqdm
from nntools.common.imageprocessing import preprocess
import facepy
# Configuration
target_FARs = [0.0001, 0.001, 0.01] # Only for verification
target_ranks = [1, 5] # Only for identification
target_FPIRs = [0.01, 0.1] # Only for identification


VerificationFold = namedtuple('VerificationFold', ['train_indices', 'test_indices', 'train_templates', 'templates1','templates2'])
IdentificationFold = namedtuple('IdentificationFold', ['train_indices', 'probe_indices', 'gallery_indices', 'train_templates', 'probe_templates', 'gallery_templates'])

class Template:
	def __init__(self, template_id, label, indices, medias):
		self.template_id = template_id
		self.label = label
		self.indices = indices
		self.medias = medias
		

def build_subject_dict(image_list):
	subject_dict = {}
	for i, line in enumerate(image_list):
		subject_id, image = tuple(line.split('/')[-2:])
		if subject_id == 'NaN': continue
		subject_id = int(subject_id)
		image, _ = os.path.splitext(image)
		image = image.replace('_','/',1) # Recover filenames 
		if not subject_id in subject_dict:
			subject_dict[subject_id] = {}
		subject_dict[subject_id][image] = i
	return subject_dict

def build_templates_old(subject_dict, meta_file):
	with open(meta_file, 'r') as f:
		meta_list = f.readlines()
		meta_list = [x.split('\n')[0] for x in meta_list]
		meta_list = meta_list[1:]

	templates = []
	template_id = None
	template_label = None
	template_indices = None
	template_medias = None
	count = 0
	for line in meta_list:
		temp_id, subject_id, image, media = tuple(line.split(',')[0:4])
		temp_id = int(temp_id)
		subject_id = int(subject_id)
		image, _ = os.path.splitext(image)
		if subject_id in subject_dict and image in subject_dict[subject_id]:
			index = subject_dict[subject_id][image]
			count += 1
		else:
			index = None

		if temp_id != template_id:
			if template_id is not None:
				templates.append(Template(template_id, template_label, template_indices, template_medias))
			template_id = temp_id
			template_label = subject_id
			template_indices = []
			template_medias = []

		if index is not None:
			template_indices.append(index)        
			template_medias.append(media)        

	# last template
	templates.append(Template(template_id, template_label, template_indices, template_medias))
	return templates


def build_templates(subject_dict, meta_file, template_dict):
	with open(meta_file, 'r') as f:
		meta_list = f.readlines()
		meta_list = [x.split('\n')[0] for x in meta_list]
		meta_list = meta_list[1:]

	template_id = None
	template_label = None
	template_indices = None
	template_medias = None
	count = 0
	for line in meta_list:
		temp_id, subject_id, image, media = tuple(line.split(',')[0:4])
		temp_id = int(temp_id)
		subject_id = int(subject_id)
		image = os.path.splitext(image)[0]
		if subject_id in subject_dict and image in subject_dict[subject_id]:
			index = subject_dict[subject_id][image]
			count += 1
		else:
			index = None

		if temp_id not in template_dict:
			template_dict[temp_id] = Template(temp_id, subject_id, [], [])

		if index is not None:
			template_dict[temp_id].indices.append(index)
			template_dict[temp_id].medias.append(media)
	 
	return template_dict

def read_pairs(pair_file):
	with open(pair_file, 'r') as f:
		pairs = f.readlines()
		pairs = [x.split('\n')[0] for x in pairs]
		pairs = [pair.split(',') for pair in pairs]
		pairs = np.array([(int(pair[0]), int(pair[1])) for pair in pairs])
	return pairs

class IJBCTest:

	def __init__(self, image_paths, config):
		self.image_paths = np.array(np.genfromtxt(image_paths, dtype=str)).astype(np.object).flatten()
		self.subject_dict = build_subject_dict(self.image_paths)
		self.verification_folds = None
		self.verification_templates = None
		self.verification_G1_templates = None
		self.verification_G2_templates = None
		self.config = config

	def init_verification_proto(self, protofolder):

		self.verification_folds = []
		self.verification_templates = []

		meta_gallery1 = os.path.join(protofolder,'ijbc_1N_gallery_G1.csv')
		meta_gallery2 = os.path.join(protofolder,'ijbc_1N_gallery_G2.csv')
		meta_probe = os.path.join(protofolder,'ijbc_1N_probe_mixed.csv')
		pair_file = os.path.join(protofolder,'ijbc_11_G1_G2_matches.csv')

		# all_templates = build_templates(self.subject_dict, meta_gallery1)
		# all_templates.extend(build_templates(self.subject_dict, meta_gallery2))
		# all_templates.extend(build_templates(self.subject_dict, meta_probe))
		# template_dict = {}
		# for t in all_templates:
		#     template_dict[t.template_id] = t
		
		template_dict = {}
		build_templates(self.subject_dict, meta_gallery1, template_dict)
		build_templates(self.subject_dict, meta_gallery2, template_dict)
		build_templates(self.subject_dict, meta_probe, template_dict)
		for t in template_dict.values():
			t.indices = np.array(t.indices)
			t.medias = np.array(t.medias)

		# Build pairs
		pairs = read_pairs(pair_file)
		unique_tids, indices_new = np.unique(pairs, return_inverse=True)
		self.verification_templates = np.array([template_dict[tid] for tid in unique_tids], dtype=np.object)
		self.verification_pairs = indices_new.reshape(-1,2)
		
		# self.verification_G1_templates = []
		# self.verification_G2_templates = []
		# for p in pairs:
		#     self.verification_G1_templates.append(template_dict[p[0]])
		#     self.verification_G2_templates.append(template_dict[p[1]])

		# indices = np.random.permutation(len(self.verification_G1_templates))[:10000]
		# self.verification_G1_templates = np.array(self.verification_G1_templates, dtype=np.object)
		# self.verification_G2_templates = np.array(self.verification_G2_templates, dtype=np.object)
	
		# self.verification_templates = np.concatenate([
		#     self.verification_G1_templates, self.verification_G2_templates])
		print('{} templates are initialized.'.format(len(self.verification_templates)))


	def init_identification_proto(self, protofolder):
		self.verification_folds = []
		self.verification_templates = []

		meta_gallery1 = os.path.join(protofolder,'ijbc_1N_gallery_G1.csv')
		meta_gallery2 = os.path.join(protofolder,'ijbc_1N_gallery_G2.csv')
		meta_probe = os.path.join(protofolder,'ijbc_1N_probe_mixed.csv')
		pair_file = os.path.join(protofolder,'ijbc_11_G1_G2_matches.csv')

		G1_dict = {}
		G2_dict = {}
		probe_dict = {}
		build_templates(self.subject_dict, meta_gallery1, G1_dict)
		build_templates(self.subject_dict, meta_gallery2, G2_dict)
		build_templates(self.subject_dict, meta_probe, probe_dict)
		template_dict = {**G1_dict, **G2_dict, **probe_dict}
		for t in template_dict.values():
			t.indices = np.array(t.indices)
			t.medias = np.array(t.medias)

		# Build pairs
		pairs = read_pairs(pair_file)
		unique_tids, indices_new = np.unique(pairs, return_inverse=True)
		self.verification_templates = np.array([template_dict[tid] for tid in unique_tids], dtype=np.object)
		self.verification_pairs = indices_new.reshape(-1,2)

		self.gallery_templates = np.array(list(G1_dict.values())+list(G2_dict.values()), dtype=np.object)
		self.probe_templates = np.array(list(probe_dict.values()), dtype=np.object)

	def init_proto(self, protofolder):
		self.init_identification_proto(protofolder)

	def update_feature(self, features):
		self.features = features

	def extract_features(self,proc_func, matcher=None, batch_size_preprocess=5000, batch_size_extract=1024):
		features = []
		n = len(self.image_paths)
		for start_idx in tqdm(range(0, n, batch_size_preprocess), desc='Extracting Features : '):
			end_idx = min(n, start_idx + batch_size_preprocess)
			temp_imgs = proc_func(self.image_paths[start_idx:end_idx])
			temp_feats = matcher.extract_feature(temp_imgs, batch_size_extract)
			features.extend(temp_feats)
		features = facepy.linalg.normalize(np.array(features), axis=-1)
		self.features = features

	def test_verification(self, compare_func, FARs=None, get_false_indices=False):

		FARs = [1e-5, 1e-4, 1e-3, 1e-2] if FARs is None else FARs

		indices1, indices2 = self.verification_pairs[:,0], self.verification_pairs[:,1]
		templates1 = self.verification_templates[indices1]
		templates2 = self.verification_templates[indices2]

		# features1 = [t.feature for t in templates1]
		# features2 = [t.feature for t in templates2]
		# score_vec = compare_func(features1, features2)
		features = np.array([facepy.linalg.normalize(np.mean(self.features[t.indices], axis=0), axis=-1) for t in self.verification_templates])
		score_mat = compare_func(features, features)
		score_vec = score_mat[indices1, indices2]


		# features1 = np.array([facepy.linalg.normalize(np.mean(self.features[t.indices], axis=0), axis=-1) for t in templates1])
		# features2 = np.array([facepy.linalg.normalize(np.mean(self.features[t.indices], axis=0), axis=-1) for t in templates2])

		# score_vec = compare_func(features1, features2)

		labels1 = np.array([t.label for t in templates1])
		labels2 = np.array([t.label for t in templates2])
		label_vec = labels1 == labels2

		temp = facepy.evaluation.ROC(score_vec, label_vec, 
				FARs=FARs, get_false_indices=get_false_indices)
		print('IJBC Verification Results : ', temp)
		return temp

	def extract_features(self, proc_func, matcher=None, batch_size_preprocess=5000, batch_size_extract=1024):
		features = []
		n = len(self.image_paths)
		for start_idx in tqdm(range(0, n, batch_size_preprocess), desc='Extracting Features : '):
			end_idx = min(n, start_idx + batch_size_preprocess)
			temp_imgs = proc_func(self.image_paths[start_idx:end_idx])
			temp_feats = matcher.extract_feature(temp_imgs, batch_size_extract)
			features.extend(temp_feats)
		self.features = facepy.linalg.normalize(np.array(features), axis=-1)
		return self.features

	def update_template_features(self,features):
		print("Updating Template Features")
		for i in range(self.verification_templates.shape[0]):
			self.verification_templates[i].feature = np.mean(features[self.verification_templates[i].indices], axis=0)

	def test_identification(self, compare_func, ranks=None, get_false_indices=False):

		ranks = [1, 5, 10] if ranks is None else ranks

		gallery_features = [t.feature for t in self.gallery_templates]
		probe_features = [t.feature for t in self.probe_templates]
		score_mat = compare_func(probe_features, gallery_features)

		labels1 = np.array([t.label for t in self.probe_templates])
		labels2 = np.array([t.label for t in self.gallery_templates])
		label_mat = labels1[:,None] == labels2[None]

		return metrics.DIR_FAR(score_mat, label_mat, ranks=ranks, get_false_indices=get_false_indices)

