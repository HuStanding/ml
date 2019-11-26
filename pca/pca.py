'''
@Author: huzhu
@Date: 2019-11-20 09:18:15
@Description: 
'''

import numpy as np
import matplotlib.pyplot as plt 

def load_data(file_name, delim='\t'):
    fr = open(file_name)
    str_arr = [line.strip().split(delim) for line in fr.readlines()]
    dat_arr = [list(map(float,line)) for line in str_arr]
    return np.mat(dat_arr)

def pca(data_mat, topNfeat = 999999):
    '''
    @description: PCA
    @return: low_data_mat, recon_mat
    '''
    mean_val = np.mean(data_mat, axis = 0)
    mean_removed = mean_val - data_mat
    # get the covrariance matrix
    cov_mat = np.cov(mean_removed, rowvar=0)
    # get the eigenvalue and eigenvector
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    # sort, sort goes smallest to largest
    eigen_val_ind = np.argsort(eigen_vals)
    # cut off unwanted dimensions
    eigen_val_ind = eigen_val_ind[:-(topNfeat+1):-1]
    print(eigen_val_ind)
    # reorganize eig vects largest to smallest
    red_eigen_vecs = eigen_vecs[:,eigen_val_ind] 
    print(red_eigen_vecs)
    # low dimension data
    low_data_mat = mean_removed * red_eigen_vecs
    # transfor low data to original dimension
    recon_mat = (low_data_mat * red_eigen_vecs.T) + mean_val
    return low_data_mat, recon_mat

if __name__ == '__main__':
    data_mat = load_data("testSet.txt")
    low_data_mat, recon_mat = pca(data_mat, 1)
    plt.figure()
    plt.scatter(data_mat[:,0].flatten().A[0], data_mat[:,1].flatten().A[0], marker='^', s = 90)
    plt.scatter(recon_mat[:,0].flatten().A[0], recon_mat[:,1].flatten().A[0], marker='o', s = 50, c = "red")
    plt.show()


