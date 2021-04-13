"""
Main file for evaluation on IJB-A and IJB-B protocols.
More instructions can be found in README.md file.
2017 Yichun Shi
"""

import sys
import os
import numpy as np
import utils

import facepy
from collections import namedtuple
import pdb
from tqdm import tqdm

# Configuration
target_FARs = [0.0001, 0.001, 0.01] # Only for verification
target_ranks = [1, 5] # Only for identification
target_FPIRs = [0.01, 0.1] # Only for identification


VerificationFold = namedtuple('VerificationFold', ['train_indices', 'test_indices', 'train_templates', 'templates1','templates2'])
IdentificationFold = namedtuple('IdentificationFold', ['train_indices', 'probe_indices', 'gallery_indices', 'train_templates', 'probe_templates', 'gallery_templates'])

class Template:
	def __init__(self, subject_id, label, indices, medias):
		self.subject_id = subject_id
		self.label = label
		self.indices = np.array(indices)
		self.medias = np.array(medias)
		

def build_subject_dict(image_list):
	subject_dict = {}
	for i, line in enumerate(image_list):
		# pdb.set_trace()
		subject_id, image = tuple(line.split('/')[-2:])
		subject_id = int(subject_id)
		image, _ = os.path.splitext(image)
		image = image.replace('_','/',1) # Recover filenames 
		if not subject_id in subject_dict:
			subject_dict[subject_id] = {}
		subject_dict[subject_id][image] = i
	return subject_dict

def build_templates(subject_dict, meta_file):
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

def read_pairs(pair_file):
	with open(pair_file, 'r') as f:
		pairs = f.readlines()
		pairs = [x.split('\n')[0] for x in pairs]
		pairs = [pair.split(',') for pair in pairs]
		pairs = [(int(pair[0]), int(pair[1])) for pair in pairs]
	return pairs

class IJBATest:

	def __init__(self, image_paths):
		self.image_paths = np.array(np.genfromtxt(image_paths, dtype=str)).astype(np.object).flatten()
		self.subject_dict = build_subject_dict(self.image_paths)
		self.verification_folds = None
		self.verification_templates = None

	def init_verification_proto(self, protofolder):
		self.verification_folds = []
		self.verification_templates = []
		for split in range(10):
			splitfolder = os.path.join(protofolder,'split%d'%(split+1))
			train_file = os.path.join(splitfolder,'train_%d.csv'%(split+1))
			meta_file = os.path.join(splitfolder,'verify_metadata_%d.csv'%(split+1))
			pair_file = os.path.join(splitfolder,'verify_comparisons_%d.csv'%(split+1))

			train_templates = build_templates(self.subject_dict, train_file)
			train_indices = list(np.unique(np.concatenate([t.indices for t in train_templates])).astype(int))

			test_templates = build_templates(self.subject_dict, meta_file)
			test_indices = list(np.unique(np.concatenate([t.indices for t in test_templates])).astype(int))
			template_dict = {}
			for t in test_templates:
				template_dict[t.subject_id] = t
			pairs = read_pairs(pair_file)
			templates1 = []
			templates2 = []
			for p in pairs:
				templates1.append(template_dict[p[0]])
				templates2.append(template_dict[p[1]])

			train_templates = np.array(train_templates, dtype=np.object)
			templates1 = np.array(templates1, dtype=np.object)
			templates2 = np.array(templates2, dtype=np.object)

			self.verification_folds.append(VerificationFold(\
				train_indices=train_indices, test_indices=test_indices,
				train_templates=train_templates, templates1=templates1, templates2=templates2))

			self.verification_templates.extend(train_templates)
			self.verification_templates.extend(templates1)
			self.verification_templates.extend(templates2)


	def init_identification_proto(self, protofolder):
		self.identification_folds = []
		for split in range(10):
			splitfolder = os.path.join(protofolder,'split%d'%(split+1))
			train_file = os.path.join(splitfolder,'train_%d.csv'%(split+1))
			probe_meta_file = os.path.join(splitfolder,'search_probe_%d.csv'%(split+1))
			gallery_meta_file = os.path.join(splitfolder,'search_gallery_%d.csv'%(split+1))

			train_templates = build_templates(self.subject_dict, train_file)
			train_indices = list(np.unique(np.concatenate([t.indices for t in train_templates])).astype(int))

			probe_templates = build_templates(self.subject_dict, probe_meta_file)
			probe_indices = list(np.unique(np.concatenate([t.indices for t in probe_templates])).astype(int))

			gallery_templates = build_templates(self.subject_dict, gallery_meta_file)
			gallery_indices = list(np.unique(np.concatenate([t.indices for t in gallery_templates])).astype(int))

			train_templates = np.array(train_templates, dtype=np.object)
			probe_templates = np.array(probe_templates, dtype=np.object)
			gallery_templates = np.array(gallery_templates, dtype=np.object)

			self.identification_folds.append(IdentificationFold(\
				train_indices=train_indices, probe_indices=probe_indices, gallery_indices=gallery_indices, 
				train_templates=train_templates, probe_templates=probe_templates, gallery_templates=gallery_templates))

	def init_proto(self, protofolder):
		self.init_verification_proto(os.path.join(protofolder, 'IJB-A_11_sets'))
		self.init_identification_proto(os.path.join(protofolder, 'IJB-A_1N_sets'))

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

	def test_verification_fold(self, compare_func, fold_idx, FARs=None, get_false_indices=False):

		FARs = [0, 0.0001, 0.001, 0.01] if FARs is None else FARs

		fold = self.verification_folds[fold_idx]

		enrolled = [len(fold.templates1[i].indices)>0 \
			and len(fold.templates2[i].indices)>0 for i in range(len(fold.templates1))]

		# Only keep the enrolled templates
		templates1 =  fold.templates1 #[enrolled]
		templates2 = fold.templates2 #[enrolled]

		import pdb
		

		features1 = np.array([facepy.linalg.normalize(np.mean(self.features[t.indices], axis=0), axis=-1) for t in templates1])
		features2 = np.array([facepy.linalg.normalize(np.mean(self.features[t.indices], axis=0), axis=-1) for t in templates2])
		# features1 = [t.feature for t in templates1]
		# features2 = [t.feature for t in templates2]
		labels1 = np.array([t.label for t in templates1])
		labels2 = np.array([t.label for t in templates2])

		score_vec = compare_func(features1, features2)
		label_vec = labels1 == labels2

		score_neg = score_vec[~label_vec]     

		return facepy.evaluation.ROC(score_vec, label_vec, 
				FARs=FARs, get_false_indices=get_false_indices)

	def test_verification(self, compare_func, FARs=None):
		
		TARs_all = []
		FARs_all = []
		for i in range(10):
			TARs, FARs, thresholds = self.test_verification_fold(compare_func, i, FARs=FARs)
			TARs_all.append(TARs)
			FARs_all.append(FARs)

		TARs_all = np.stack(TARs_all)
		FARs_all = np.stack(FARs_all)


		return np.mean(TARs_all, axis=0), np.std(TARs_all, axis=0), np.mean(FARs_all, axis=0)
