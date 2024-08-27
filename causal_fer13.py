import os
import dlib
from PIL import Image
import numpy as np
import cv2
from imutils import face_utils
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz

def get_au_region(data):
    try:
        cg = pc(data, 0.05, fisherz, show_progress=False)
        G = cg.G.__str__()

        left_count = {}
        edges_list = G.split("\n\n")[1].strip().split("\n")

        for edge in edges_list:
            nodes = edge.split(" --> ")
            if len(nodes) == 2:
                left_node = nodes[0].split(" -- ")[-1]
                left_node = left_node.split("X")[-1]
                if left_node not in left_count:
                    left_count[left_node] = 1
                else:
                    left_count[left_node] += 1

        max_left_count_node = max(left_count, key=left_count.get)
        max_left_count = left_count[max_left_count_node]
    except ValueError:
        max_left_count_node = 5
        max_left_count = 5

    return max_left_count_node, max_left_count

def getRoI(lm, RoI):
    if RoI == 1: 
        y, x = (lm[17] + lm[18]) / 2
    elif RoI == 2: 
        y, x = lm[20]
        y += 10
    elif RoI == 3: 
        y, x = lm[23]
        y += 10
    elif RoI == 4: 
        y, x = (lm[25] + lm[26]) / 2
    elif RoI == 5: 
        y, x = (lm[37] + lm[38]) / 2
        x -= 4
    elif RoI == 6: 
        y, x = (lm[21] + lm[22]) / 2
        y += 19
        x += 2
    elif RoI == 7: 
        y, x = (lm[43] + lm[44]) / 2
        x += 10
    elif RoI == 8: 
        y, x = (lm[1] + lm[29]) / 2
        y += 10
        x -= 25
    elif RoI == 9: 
        y, x = (lm[15] + lm[29]) / 2
        y += 10
        x += 25
    elif RoI == 10: 
        y, x = lm[60]
        y -= 2
    elif RoI == 11: 
        y, x = (lm[33] + lm[51]) / 2
        y -= 2
    elif RoI == 12: 
        y, x = lm[64]
        y -= 2
    elif RoI == 13: 
        y, x = (lm[62] + lm[66]) / 2
        y += 2
        x += 1
    elif RoI == 14: 
        y, x = (lm[57] + lm[8]) / 2
    y = int(y)
    x = int(x)
    margin_x = 12 if RoI in [1, 4, 10] else 47 if RoI in [5, 7, 13] else 35 if RoI in [8, 9, 14] else 15
    margin_y = 10 if RoI in [1, 2, 3, 4, 10] else 21 if RoI in [5, 7, 13, 14] else 19 if RoI in [6] else 30
    return x, y, margin_x, margin_y

def get_14_region(img_path): 
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('../STSTNet/shape_predictor_68_face_landmarks.dat')
    img = cv2.imread(img_path)
    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = im_rgb.shape               
    bb = dlib.rectangle(left=0, top=0, right=w, bottom=h)
    shape = predictor(im_rgb, bb)
    LM = face_utils.shape_to_np(shape)   
    lm = np.zeros((len(LM), 2))
    for m, [x, y] in enumerate(LM):
        lm[m] = [y, x]
    
    arrays = []
    for RoI in range(1, 15):
        x, y, margin_x, margin_y = getRoI(lm, RoI)
        mask = np.zeros((h, w), np.uint8)
        mask[y-margin_y:y+margin_y, x-margin_x:x+margin_x] = 1  
        AU_img = cv2.bitwise_and(im_rgb, im_rgb, mask=mask)
        arrays.append(np.reshape(AU_img, (-1)))

    concatenated_array = np.vstack(arrays)
    return np.transpose(concatenated_array, (1, 0))

def get_node(roi_number):
    roi_coordinates = {
        1: (2, 3),
        2: (9, 11),
        3: (19, 11),
        4: (35, 3),
        5: (3, 6),
        6: (16, 21),
        7: (25, 5),
        8: (1, 21),
        9: (35, 21),
        10: (9, 18),
        11: (14, 14),
        12: (19, 18),
        13: (15, 21),
        14: (14, 25)
    }
    return int(roi_coordinates[roi_number][0]), int(roi_coordinates[roi_number][1])

def paste_sticker_causal(base_image_path, sticker_path, output_path, max_left_count_node):
    x, y = get_node(int(max_left_count_node))
    base_image = Image.open(base_image_path)
    sticker_image = Image.open(sticker_path).convert("RGBA")
    sticker_size = (48 * 2 // 10, 48 * 2 // 10)
    resized_sticker = sticker_image.resize(sticker_size, Image.Resampling.LANCZOS)
    y = 48 - y
    base_image.paste(resized_sticker, (x, y), resized_sticker)
    base_image.save(output_path)
    return output_path

def process_images_recursively(root_directory, sticker_image_paths, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    for current_directory, _, files in os.walk(root_directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_image_path = os.path.join(current_directory, file)
                emotion = os.path.split(current_directory)[1]

                try:
                    data = get_14_region(input_image_path)
                    max_edge, _ = get_au_region(data)

                    for i, sticker_image_path in enumerate(sticker_image_paths):
                        sticker_name = os.path.splitext(os.path.basename(sticker_image_path))[0]
                        sticker_folder = os.path.join(output_directory, sticker_name, emotion)

                        if not os.path.exists(sticker_folder):
                            os.makedirs(sticker_folder)
                        
                        output_image_path = os.path.join(sticker_folder, file)

                        paste_sticker_causal(input_image_path, sticker_image_path, output_image_path, max_edge)
                except IsADirectoryError:
                    print('Pass')

if __name__ == "__main__":
    root_directory = "/home/tpei0009/STSTNet/facial-recognition-dataset/test"
    sticker_image_paths = [
        "/home/tpei0009/MMNet/star_cartoon.png", 
        "/ibm/gpfs/home/tpei0009/STSTNet/sun_rm.png", 
        "/ibm/gpfs/home/tpei0009/sticker_test.png", 
        "/ibm/gpfs/home/tpei0009/STSTNet/heart_sticker_rm.png"
    ]
    output_directory = "./fer2013/test_causal/"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    process_images_recursively(root_directory, sticker_image_paths, output_directory)
