#!/usr/bin/env python3

from skimage import io
from matplotlib import pyplot as plt
import numpy as np
#from pixie_preprocessing import *
#from pixie_cluster_utils import *
from PIL import Image
from numpy import asarray
import os
import tifffile
from pathlib import Path

'''
Function to get all directories in a given path
'''
def get_directories(path):
    directories = []
    for entry in os.scandir(path):
        if entry.is_dir():
            directories.append(entry.name)
    return directories

def get_tiled_data(n):
    final_images = []
    similar_pairs = []
    dis_pairs = []
    fov = ["TIFs"]
    three_channels = ['159_NEFH','169_Iba1','174_GFAP']
    #three_channel_norm = channel_norm[['159_NEFH.tif','169_Iba1.tif','174_GFAP.tif']]
    for directory in directories:
        path = "Noise_rm_2/"+directory+"/"
        img_xr = load_utils.load_imgs_from_tree(Path(path), fovs=["TIFs"])
        misc_utils.verify_in_list(
            provided_chans=three_channels,
            pixel_mat_chans=img_xr.channels.values
        )
        img_data = img_xr.loc[fov, :, :, three_channels].values.astype(np.float32)

        # create vector for normalizing image data
        norm_vect = three_channel_norm.iloc[0].values
        norm_vect = np.array(norm_vect).reshape([1, 1, len(norm_vect)])

        # normalize image data
        img_data = img_data / norm_vect
        img_data = np.squeeze(img_data)
        M = 256
        N = 256
        tiles = [img_data[x:x+M,y:y+N] for x in range(0,img_data.shape[0],M) for y in range(0,img_data.shape[1],N)]
        #tile_coords = [(x,x+M,y,y+N) for x in range(0,img_data.shape[0],M) for y in range(0,img_data.shape[1],N)]
        #final_tiles = [[i+1, tile_coords[i]] for i in range(len(tiles))]
        '''
        for i in range(len(tiles)):
            final_images.append([directory, i+1, tile_coords[i], tiles[i]])
        '''

        adj_mappings = get_mappings(4)
        # make similar pairs
        for key in adj_mappings:
            for value in adj_mappings[key]:
                similar_pairs.append([tiles[key-1], tiles[value-1]])

        # make dissimilar pairs
        for key in adj_mappings:
            all_values = [i for i in range(1, 17)]
            key_values = adj_mappings[key] + [key]
            values = list(set(all_values) - set(key_values))
            for value in values:
                dis_pairs.append([tiles[key-1], tiles[value-1]])
    
    random.shuffle(similar_pairs)
    random.shuffle(dis_pairs)
    
    train_sim = similar_pairs[:3500]
    train_dissim = dis_pairs[:3500]
    
    eval_sim = similar_pairs[3500:4500]
    eval_dissim = dis_pairs[3500:4500]
    
    test_sim = similar_pairs[4500:]
    test_dissim = dis_pairs[4500:5088]
    
    
    np.savez_compressed('train_sim_pairs.npz', *train_sim)
    np.savez_compressed('train_dissim_pairs.npz', *train_dissim)
    np.savez_compressed('eval_sim_pairs.npz', *eval_sim)
    np.savez_compressed('eval_dissim_pairs.npz', *eval_dissim)
    np.savez_compressed('test_sim_pairs.npz', *test_sim)
    np.savez_compressed('test_dissim_pairs.npz', *test_dissim)


'''
Evaluate accuracy of model
'''
def accuracy(predictions, labels):
    binary_predictions = (predictions >= 0.5).float()

    # Compare binary predictions with true labels
    correct = (binary_predictions == labels).sum().item()
    total = labels.size(0)

    # Compute accuracy
    accuracy = correct / total

    return accuracy