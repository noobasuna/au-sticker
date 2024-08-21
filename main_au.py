import math
import numpy as np
import torchvision.models
import torch.utils.data as data
from torchvision import transforms
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import os, torch
import torch.nn as nn
import argparse, random
from functools import partial
from torchvision.transforms import Resize
torch.set_printoptions(precision=3, edgeitems=14, linewidth=350)
import csv
from PIL import Image
import random
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq, fisherz, kci, d_separation, mv_fisherz,gsq
import dlib
from imutils import face_utils

def get_au_region(au_path):
    ex = np.load(au_path)
    data = np.transpose(ex, (1, 0))
    cg = pc(data, 0.05, fisherz, show_progress=False)
    print(cg.G)
    G = cg.G.__str__()

    left_count = {}
    input_str = G

    # Split the input string into nodes and edges
    nodes_str, edges_str = input_str.split("\n\n")
    nodes = nodes_str.split(";")
    edges_list = edges_str.strip().split("\n")
    #
    for edge in edges_list:
        nodes = edge.split(" --> ")
        if len(nodes) == 2:
            left_node = nodes[0].split(" -- ")[-1]
            left_node = left_node.split("X")[-1]
            if left_node not in left_count:
                left_count[left_node] = 1
            else:
                left_count[left_node] += 1
    try:
        max_left_count_node = max(left_count, key=left_count.get)
        max_left_count = left_count[max_left_count_node]

    except ValueError:
        max_left_count_node = 1
        max_left_count = 1

    return max_left_count_node, max_left_count

def getmax_left_count_node(lm, max_left_count_node):
    max_left_count_node = int(max_left_count_node)
    if max_left_count_node == 1:
        y, x = (lm[17] + lm[18]) / 2
        y = int(y)
        x = int(x)
    elif max_left_count_node == 2:
        y, x = lm[20]
        y = int(y + 10)
        x = int(x)
    elif max_left_count_node == 3:
        y, x = lm[23]
        y = int(y + 10)
        x = int(x)
    elif max_left_count_node == 4:
        y, x = (lm[25] + lm[26]) / 2
        y = int(y)
        x = int(x)
    elif max_left_count_node == 5:
        y, x = (lm[37] + lm[38]) / 2
        y = int(y)
        x = int(x - 4)
    elif max_left_count_node == 6:
        y, x = (lm[21] + lm[22]) / 2
        y = int(y + 19)
        x = int(x + 2)
    elif max_left_count_node == 7:
        y, x = (lm[43] + lm[44]) / 2
        y = int(y)
        x = int(x + 10)
    elif max_left_count_node == 8:
        y, x = (lm[1] + lm[29]) / 2
        y = int(y + 10)
        x = int(x - 25)
    elif max_left_count_node == 9:
        y, x = (lm[15] + lm[29]) / 2
        y = int(y + 10)
        x = int(x + 25)
    elif max_left_count_node == 10:
        y, x = lm[60]
        y = int(y - 2)
        x = int(x)
    elif max_left_count_node == 11:
        y, x = (lm[33] + lm[51]) / 2
        y = int(y - 2)
        x = int(x)
    elif max_left_count_node == 12:
        y, x = lm[64]
        y = int(y - 2)
        x = int(x)
    elif max_left_count_node == 13:
        y, x = (lm[62] + lm[66]) / 2
        y = int(y + 2)
        x = int(x + 1)
    elif max_left_count_node == 14:
        y, x = (lm[57] + lm[8]) / 2
        y = int(y)
        x = int(x)
    else:
        print(max_left_count_node)
    return x, y

def get_landmarks(img_path):
    predictor = dlib.shape_predictor('../STSTNet/shape_predictor_68_face_landmarks.dat')
    img = cv2.imread(img_path)
    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(im_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    h, w, c = resized.shape
    bb = dlib.rectangle(left=0, top=0, right=w, bottom=h)
    shape = predictor(resized, bb)
    LM = face_utils.shape_to_np(shape)
    lm = np.zeros((len(LM), 2))
    for m, [x, y] in enumerate(LM):
        lm[m] = [y, x]
    return lm

def get_au_region(data):
    cg = pc(data, 0.05, fisherz, show_progress=False)
    print(cg.G)
    G = cg.G.__str__()

    left_count = {}
    input_str = G

    # Split the input string into nodes and edges
    nodes_str, edges_str = input_str.split("\n\n")
    nodes = nodes_str.split(";")
    edges_list = edges_str.strip().split("\n")
    #
    for edge in edges_list:
        nodes = edge.split(" --> ")
        if len(nodes) == 2:
            left_node = nodes[0].split(" -- ")[-1]
            left_node = left_node.split("X")[-1]
            if left_node not in left_count:
                left_count[left_node] = 1
            else:
                left_count[left_node] += 1
    try:
        max_left_count_node = max(left_count, key=left_count.get)
        max_left_count = left_count[max_left_count_node]

    except ValueError:
        max_left_count_node = 1
        max_left_count = 1

    return max_left_count_node, max_left_count

def getRoI(lm,RoI):
    if RoI == 1:
        y,x = (lm[17]+lm[18])/2
        y = int(y)
        x = int(x)
        margin_x = 12
        margin_y = 10
        start_point = (x-margin_x, y-margin_y)
        end_point = (x+margin_x, y+margin_y)
    elif RoI == 2:
        y,x = lm[20]
        y = int(y+10)
        x = int(x)
        margin_x = 15
        margin_y = 10
        start_point = (x-margin_x, y-margin_y)
        end_point = (x+margin_x, y+margin_y)
    elif RoI == 3:
        y,x = lm[23]
        y = int(y+10)
        x = int(x)
        margin_x = 15
        margin_y = 10
        start_point = (x-margin_x, y-margin_y)
        end_point = (x+margin_x, y+margin_y)
    elif RoI == 4:
        y,x = (lm[25]+lm[26])/2
        y = int(y)
        x = int(x)
        margin_x = 12
        margin_y = 10
        start_point = (x-margin_x, y-margin_y)
        end_point = (x+margin_x, y+margin_y)
    elif RoI == 5:
        y,x =  (lm[37]+lm[38])/2
        y = int(y)
        x = int(x-4)
        margin_x = 47
        margin_y = 21
        start_point = (x-margin_x, y-margin_y)
        end_point = (x+margin_x, y+margin_y)
    elif RoI == 6:
        y,x = (lm[21]+lm[22])/2
        y = int(y+19)
        x = int(x+2)
        margin_x = 30
        margin_y = 19
        start_point = (x-margin_x, y-margin_y)
        end_point = (x+margin_x, y+margin_y)
    elif RoI == 7:
        y,x = (lm[43]+lm[44])/2
        y = int(y)
        x = int(x+10)
        margin_x = 47
        margin_y = 21
        start_point = (x-margin_x, y-margin_y)
        end_point = (x+margin_x, y+margin_y)
    elif RoI == 8:
        y,x = (lm[1]+lm[29])/2
        y = int(y+10)
        x = int(x-25)
        margin_x = 35
        margin_y = 30
        start_point = (x-margin_x, y-margin_y)
        end_point = (x+margin_x, y+margin_y)
    elif RoI == 9:
        y,x = (lm[15]+lm[29])/2
        y = int(y+10)
        x = int(x+25)
        margin_x = 35
        margin_y = 30
        start_point = (x-margin_x, y-margin_y)
        end_point = (x+margin_x, y+margin_y)
    elif RoI == 10:
        y,x = lm[60]
        y = int(y-2)
        x = int(x)
        margin_x = 10
        margin_y = 12
        start_point = (x-margin_x, y-margin_y)
        end_point = (x+margin_x, y+margin_y)
    elif RoI == 11:
        y,x = (lm[33]+lm[51])/2
        y = int(y-2)
        x = int(x)
        margin_x = 30
        margin_y = 17
        start_point = (x-margin_x, y-margin_y)
        end_point = (x+margin_x, y+margin_y)
    elif RoI == 12:
        y,x = lm[64]
        y = int(y-2)
        x = int(x)
        margin_x = 10
        margin_y = 12
        start_point = (x-margin_x, y-margin_y)
        end_point = (x+margin_x, y+margin_y)
    elif RoI == 13:
        y,x = (lm[62]+lm[66])/2
        y = int(y+2)
        x = int(x+1)
        margin_x = 47
        margin_y = 20
        start_point = (x-margin_x, y-margin_y)
        end_point = (x+margin_x, y+margin_y)
    elif RoI == 14:
        y,x = (lm[57]+lm[8])/2
        y = int(y)
        x = int(x)
        margin_x = 35
        margin_y = 21
        start_point = (x-margin_x, y-margin_y)
        end_point = (x+margin_x, y+margin_y)
    coord = (y,x)
    return x, y, margin_x, margin_y

def get_14_region(img_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('../STSTNet/shape_predictor_68_face_landmarks.dat')

    img = cv2.imread(img_path)

    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(im_rgb, (256,256), interpolation = cv2.INTER_AREA)
    h, w, c = im_rgb.shape
    bb = dlib.rectangle(left=0, top=0, right=w, bottom=h)
    shape = predictor(im_rgb, bb)
    LM = face_utils.shape_to_np(shape)
    lm = np.zeros((len(LM),2))
    for m, [x,y] in enumerate(LM):
         lm[m] = [y,x]
    concatenated_rois = np.zeros_like(im_rgb, dtype=np.uint8)
    arrays = []
    for RoI in range(1, 15):
        x, y, margin_x, margin_y = getRoI(lm, RoI)
        mask = np.zeros((h, w), np.uint8)
        mask[y-margin_y:y+margin_y, x-margin_x:x+margin_x] = 1
        AU_img = cv2.bitwise_and(im_rgb, im_rgb, mask=mask)
#         AU_resized = cv2.resize(AU_img, (256,256), interpolation=cv2.INTER_AREA)
        #concatenated_rois = np.concatenate([concatenated_rois, AU_img], axis=2)
        a,b,c = AU_img.shape
        stack_array = np.reshape(AU_img, (a*b*c))
        arrays.append(stack_array)

    # Concatenate all the loaded arrays into a single 2D array
    concatenated_array = np.vstack(arrays)  # This will stack them vertically to create a (14, 1234) array
    data = np.transpose(concatenated_array, (1, 0))

    return data

def save_to_csv(file_path, data):
    with open(file_path, 'a', newline='') as csvfile:
        fieldnames = ['image_path', 'max_left_count_node', 'max_left_count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # If the file is empty, write the header
        if csvfile.tell() == 0:
            writer.writeheader()

        # Write the data to the CSV file
        writer.writerow({'image_path': data[0] ,'max_left_count_node': data[1], 'max_left_count': data[2]})

def process_images_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                file_path = os.path.join(root, file)
                print(file_path)
                data = get_14_region(file_path)
                max_left_count_node, max_left_count = get_au_region(data)
                save_to_csv('genderhqa_checkau.csv', (file_path, max_left_count_node, max_left_count))

if __name__ == "__main__":
    target_directory = "/home/tpei0009/RHDE/CelebA_HQ_face_gender_dataset_test"
    process_images_in_directory(target_directory)