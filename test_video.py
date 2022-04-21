import sys
import time
import os
from numpy import save
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw
from utils import *
from darknet import *
from load_data import PatchTransformer, PatchApplier, InriaDataset
import json
from tqdm import tqdm
import cv2

import torchvision.models as models

def add_frame_number(frame):
    # text font parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2

    # Using cv2.putText() method
    image = cv2.putText(frame, str(frames), org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    
    return image

if __name__ == '__main__':
    print("Setting everything up")

    view_video = False
    save_video = True

    viddir = "../../data/videos/outline patch_v2"
    cfgfile = "cfg/yolo.cfg"
    weightfile = "weights/yolo.weights"

    savedir = "testing/videos/outline patch_v2" # change this
    prefix = "robust_downres"
    suffix = "yolo" # to append to output video filename
    conf_thresh = 0.4
    nms_thresh = 0.4
    class_names = open("coco-labels-2014_2017.txt", "r").readlines()

    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)
    darknet_model = darknet_model.eval().cuda()

    batch_size = 1
    img_width = darknet_model.width
    img_height = darknet_model.height

    if save_video:
        savedirs = [os.path.join(savedir)]
        # make saving directory if not exist
        for dir in savedirs:
            if not os.path.exists(dir):
                os.makedirs(dir)
    
    print("Done")
    #Loop over videos
    for videofile in tqdm(os.listdir(viddir)):
        # print("\nnew video", videofile)
        if videofile.endswith('.avi') or videofile.endswith('.mp4'):
            name = os.path.splitext(videofile)[0]    #image name w/o extension
            # txtname = name + '.txt'
            # txtpath = os.path.abspath(os.path.join(savedir, 'boxes/', txtname))
            # open beeld en pas aan naar yolo input size
            # pad image
            videofile = os.path.abspath(os.path.join(viddir, videofile))

            # read in videofile
            cap = cv2.VideoCapture(videofile)

            assert cap.isOpened(), 'Cannot capture source'

            # Default resolutions of the input frame are obtained.The default resolutions are system dependent.
            # We convert the resolutions from float to integer.
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if save_video:
                # Define the codec and create VideoWriter object.The output is stored in savename file.
                savename = os.path.join(savedir, f"{prefix}_{name}_{suffix}.avi")
                out = cv2.VideoWriter(savename,
                                    cv2.VideoWriter_fourcc('M','J','P','G'),
                                    fps, (frame_width,frame_height))

            frames = 0
            start = time.time()
            while cap.isOpened():

                ret, frame = cap.read()
                if ret:
                    # transforming the frame image
                    img = Image.fromarray(frame).convert('RGB')
                    w,h = img.size
                    if w==h:
                        padded_img = img
                    else:
                        dim_to_pad = 1 if w<h else 2
                        if dim_to_pad == 1:
                            padding = (h - w) / 2
                            padded_img = Image.new('RGB', (h,h), color=(127,127,127))
                            padded_img.paste(img, (int(padding), 0))
                        else:
                            padding = (w - h) / 2
                            padded_img = Image.new('RGB', (w, w), color=(127,127,127))
                            padded_img.paste(img, (0, int(padding)))
                    resize = transforms.Resize((img_width,img_height))
                    resized_img = resize(padded_img)
                    
                    # detecting with model
                    boxes = do_detect(darknet_model, resized_img, conf_thresh, nms_thresh, True)
                    # textfile = open(txtpath,'w+')
                    for box in boxes:
                        cls_id = box[6]
                        if(cls_id == 0):   #if person
                            x_center = box[0]
                            y_center = box[1]
                            width = box[2]
                            height = box[3]
                            # textfile.write(f'{cls_id} {x_center} {y_center} {width} {height}\n')
                    # textfile.close()

                    # plot bouding boxes
                    plotted_img = plot_boxes(padded_img, boxes, class_names=class_names)

                    # convert back to np.array for cv2
                    final_image = np.asarray(plotted_img)

                    # crop away padding
                    x = (final_image.shape[1] - frame_width) / 2
                    y = (final_image.shape[0] - frame_height) / 2
                    final_image = final_image[int(y):int(y+frame_height), int(x):int(x+frame_width)]

                    # # add frame number
                    # frames += 1
                    # final_image = add_frame_number(final_image)
                    
                    if view_video:
                        cv2.imshow(videofile, final_image)
                        # Press Q on keyboard to  exit
                        if cv2.waitKey(16) & 0xFF == ord('q'): # delay for 16ms
                            break

                    if save_video:
                        # Write the frame into output file
                        out.write(final_image)
                
                else: # not ret
                    break
            
            cap.release()
            if save_video:
                out.release()
                print("Saved video", savename)
            cv2.destroyAllWindows()