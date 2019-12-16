# import cPickle as pickle
import pickle
import numpy as np
import copy

def range_data():
    data_dict = {}

    with open("Data/data_dict","rb") as f:
        data_dict = pickle.load(f) 

    new_data_dict = {}
    new_data_dict["Adj"] = data_dict["Adj"][:15000]
    new_data_dict["D"] = data_dict["D"][:15000]
    new_data_dict["P"] = data_dict["P"][:15000]
    new_data_dict["accuracy"] = data_dict["accuracy"][:15000]

    with open("Data/new_data_dict","wb") as f:
        pickle.dump(new_data_dict,file=f)

def get_shuffled_data():
    data_dict = {}
    with open("Data/new_data_dict","rb") as f:
        data_dict = pickle.load(f) 
    
    A = data_dict["Adj"]
    D = data_dict["D"]
    feature = data_dict["P"]
    features = np.sum(feature,axis=2)

    one_hot_feature = np.zeros((len(features),len(features[0]),6))
    for i in range(len(features)):
        for j in range(len(features[0])):
            one_hot_feature[i][j][int(features[i][j])] = 1

    data = np.asarray(A)
    datashape = np.shape(data)

    
    A_bar = A + np.eye(datashape[1],datashape[2])
    D_bar = D + np.eye(datashape[1],datashape[2])

    for i in range(len(D_bar)):
        for j in range(len(D_bar[0])):
            D_bar[i][j][j] = 1/np.sqrt(D_bar[i][j][j])
    
    supports = []
    for i in range(len(A_bar)):
        support = np.matmul(np.matmul(D_bar[i],A_bar[i]),D_bar[i])
        supports.append(support)

    supports = np.asarray(supports)

    labels = data_dict["accuracy"]
    labels = np.reshape(labels,(len(labels),1))

    indice = [i for i in range(len(supports))]
    np.random.shuffle(indice)

    shuffled_supports = []
    shuffled_onehot_features = []
    shuffled_features = []
    shuffled_labels = []
    for i in indice:
        shuffled_supports.append(supports[i])
        shuffled_onehot_features.append(one_hot_feature[i])
        shuffled_features.append(np.reshape(features[i],[12,1]))
        shuffled_labels.append(labels[i])

    shuffled_supports = np.asarray(shuffled_supports)
    shuffled_features = np.asarray(shuffled_features)
    shuffled_labels = np.asarray(shuffled_labels)
    shuffled_onehot_features = np.asarray(shuffled_onehot_features)

    return [shuffled_supports,shuffled_features,shuffled_labels,shuffled_onehot_features]

if __name__ == "__main__":
    
    range_data()