#!/usr/bin/env python3

from skimage import io
from matplotlib import pyplot as plt
import numpy as np
from pixie_preprocessing import *
from pixie_cluster_utils import *
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

'''
Function to get all files in a given path
'''
def get_files(path):
    files = []
    for entry in os.scandir(path):
        if entry.is_file():
            files.append(entry.name)
    return files

'''
Function to get adjacent tile numbers in a given image
'''
def get_mappings(n):
    tiles = n * n
    adj_mappings = {}
    for i in range(1, tiles+1):
        values = []
        if (i % n != 0):
            values.append(i+1)
        if (i - n > 0):
            values.append(i - 4)
        if (i + n <= tiles):
            values.append(i + 4)
        if ((i - 1) % n != 0):
            values.append(i-1)
        adj_mappings[i] = values
    return adj_mappings


'''
Function to get tiled data for three channels --  NEFH, Iba1, GFAP
Creates similar and dissimilar pairs and returns two dataframes -- one for sim and one for dissim
'''
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
    np.savez_compressed('sim_pairs.npz', *similar_pairs)
    np.savez_compressed('dissim_pairs.npz', *dis_pairs)