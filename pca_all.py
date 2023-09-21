#!/usr/bin/env python3

#Script pour lire les fichiers de représentations des modèles GMM obtenus sur les audio des stimuli de l'expérience EEG, les concaténer et faire une PCA dessus.

import pandas as pd
import numpy as np
from os.path import join
import re
import glob
import h5features as h5f
from sklearn.decomposition import PCA
import pickle


def read_features_GMM(features_file_path):
    with h5f.Reader(features_file_path, 'features') as reader:
        data = reader.read()
    input_samples = data.labels() #timings des features
    input_features = data.features() #features concrets
    input_items=data.items() #nom des features ('trial_1'...)
    return input_samples, input_features, input_items

features_path = r"/baie/nfs-cluster-1/data1/home/ambre.balleroy/data/GMM_TRF/GMM_posteriors"
models = glob.glob(join(features_path, r'*final.features'))

data_dic = {}
features_names_dic = {}
for model in models:
   try:
       input_samples, input_features, input_items = read_features_GMM(model)
   except OSError:
       print("Model", model, "could not be opened")
   #avant cette étape faire les crops
   all_input_features = np.concatenate(input_features, axis=0)
   #sample_names = ["trial_{}_{}".format(i, y) for i in range(20)]
   model_name = re.search("[A-Z]{3}_\d+_\d+_", model).group()
   features_names = [model_name + str(i) for i in range(all_input_features.shape[1])]
   #print(len(features_names))
   data_dic[model_name]=all_input_features
   features_names_dic[model_name]=features_names

all_data = np.concatenate(list(data_dic.values()), axis=1)
all_feature_names = np.concatenate(list(features_names_dic.values()))
data_df=pd.DataFrame(all_data, columns=all_feature_names)

del all_data
del data_dic

# data_df = pd.read_csv("all_data.csv")
data_df.to_csv("all_data.csv", index=False)
#print("dataframe has been saved")

dataset_split_length = data_df.shape[0]/10

for i in range(0, data_df.shape[0], dataset_split_length):
    split = data_df[i, :]
    pca = PCA(n_components = 2000)
    print("Fitting is about to begin")
    model = pca.fit_transform(data_df)
    with open('pca_{}.pickle'.format(i), 'wb') as f:
        pickle.dump(pca, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('transformed_data_{}.pickle'.format(i), 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)







