#!/usr/bin/env python3
import pandas as pd
import numpy as np
import pickle
from sklearn.cross_decomposition import CCA

model_path = "pca_buc.pickle"
with open(model_path, 'rb') as f:
        pca_buc = pickle.load(f)

model_path = "pca_gpj.pickle"
with open(model_path, 'rb') as f:
        pca_gpj = pickle.load(f)

model_path = "pca_wsj.pickle"
with open(model_path, 'rb') as f:
        pca_wsj = pickle.load(f)

model_path = "pca_csj.pickle"
with open(model_path, 'rb') as f:
        pca_csj = pickle.load(f)



model_path = "transformed_data_buc.pickle"
with open(model_path, 'rb') as f:
        raw_data_buc = pickle.load(f)

model_path = "transformed_data_wsj.pickle"
with open(model_path, 'rb') as f:
        raw_data_wsj = pickle.load(f)

model_path = "transformed_data_csj.pickle"
with open(model_path, 'rb') as f:
        raw_data_csj = pickle.load(f)

model_path = "transformed_data_gpj.pickle"
with open(model_path, 'rb') as f:
        raw_data_gpj = pickle.load(f)

def select_dimensions(pca):
    exp_var_pca = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)
    last_component = min([i for i,v in enumerate(cum_sum_eigenvalues) if v >= 0.9])
    return last_component

data_buc= raw_data_buc[:, 1:select_dimensions(pca_buc)] #on enlève les composantes après celles qui expliquent 90% de la variance et la toute première qui représente le silence
data_csj= raw_data_csj[:, 1:select_dimensions(pca_csj)]
data_wsj= raw_data_wsj[:, 1:select_dimensions(pca_wsj)]
data_gpj= raw_data_gpj[:, 1:select_dimensions(pca_gpj)]


data_jp = np.concatenate([data_gpj, data_csj], axis=1)
data_en = np.concatenate([data_buc, data_wsj], axis=1)

cca = CCA(n_components=4)
en_scores, jp_scores = cca.fit_transform(data_en, data_jp) 

with open('cca_jp_scores.pickle', 'wb') as f:
    pickle.dump(jp_scores, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('cca_en_scores.pickle', 'wb') as f:
    pickle.dump(en_scores, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('cca.pickle', 'wb') as f:
    pickle.dump(cca, f, protocol=pickle.HIGHEST_PROTOCOL)

