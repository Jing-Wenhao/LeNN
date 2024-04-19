import numpy as np
from ase.io import read
from ase.db import connect
from hads import element_information
from hads import training
from hads import train_NN
from matplotlib import pyplot as plt
from hads.new_feature import Features_coord

db = connect(r'H.db')
db_pred = connect(r'H_pred.db')
# db_val = connect(r'H_new526.db')

index_max = 0
count = 0
nop_specie = []
X = []
Y = []
for atom in db.select():
    structure = atom.toatoms()
    a = []
    if atom.get('ads') != 0:
        symbols = structure.get_chemical_symbols()
        # print(atom.get('id'))
        id = atom.get('id')
        if set(symbols) != {'O', 'H'}:
            count_band = 0
            for i in list(set(symbols)):
                try:
                    element_information.data[i]['band_center']
                except KeyError:
                    pass
                else:
                    count_band += 1
            if count_band != 0:
                feature_test = Features_coord(structure)
            # if len(feature_test.index_3) > index_max and structure[feature_test.index_2].symbol == 'O':
            #     index_max = len(feature_test.index_3)
            #     print(atom.get('id'), atom.get('species'), index_max, feature_test.index_2, feature_test.index_3)
            # count += 1
                if feature_test.judge() :
                    id = atom.get('id')
                    x = feature_test.get_features()
                    y = atom.get('ads_e_single')
                    if np.any(np.isnan(x)) and set(symbols) != {'O', 'H'}:
                        X.append(x)
                        Y.append(y)
#
# # X_tocal = []
# # Y_tocal = []
# # for index in val:
# #     atom = db_val[index]
# #     structure = db_val[index].toatoms()
# #     nop_specie = ['In2O3_14387', 'O2S1_24645', 'Na2O1_60435']
# #     if atom.get('ads') != 0 and atom.get('species') not in nop_specie:
# #         feature_test = Features(structure, id=4, cut=1.05, max_coord=5)
# #         if feature_test.judge():
# #             id = atom.get('id')
# #             print(id)
# #             x = feature_test.get_features()
# #             y = atom.get('ads_e_single')
# #             if np.abs(y) <= 3 and not np.any(np.isnan(x)):
# #                 X_tocal.append(x)
# #                 Y_tocal.append(y)
# #
X = np.matrix(X)
Y = np.matrix(Y).T
# # #
MAE = []
pred = []
# for j in range(10):
#     mae, mae_train = training.train_data(X, Y, j)
#     MAE.append(mae)
#     MAE_train.append(mae_train)
    # importance_all.append(importance)
for j in range(10):
    pred.append(train_NN.train_nn(X, Y,))
