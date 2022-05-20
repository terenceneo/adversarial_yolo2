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
    fontScale = 1
    thickness = 2

    # Using cv2.putText() method
    image = cv2.putText(frame, str(frames), org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    
    return image

def add_text(img, txt, frame_width, frame_height, line, rgb=(255,0,0)):
    width = img.width
    font_multiplier = 60
    offset = 50
    if frame_width > frame_height:
        y = int((frame_width - frame_height)/2) + offset
    else:
        y = offset
    y += line * (round(width/font_multiplier) + 10) # spacing between lines
    pos = (offset, y)
    font=ImageFont.truetype("arialbd.ttf", size=round(width/font_multiplier))
    draw = ImageDraw.Draw(img)
    draw.text(pos, txt, fill=rgb, font=font)
    return img

if __name__ == '__main__':
    print("Setting everything up")

    # parameters to change
    view_video = True
    save_video = False

    detect_on_downsized_img = True
    downsized_cfgfile = "cfg/yolo.cfg"
    downsized_img_model_name = "DSO Model"

    detect_on_modelsized_img = True
    darknet_cfgfile = "cfg/yolo_HD.cfg"
    modelsized_img_model_name = "YOLOv2"

    conf_thresh_downsized = 0.6
    conf_thresh_modelsized = 0.7
    nms_thresh = 0.4

    # viddir = "../../data/videos/outline patch_v2"

    savedir = "testing/videos/outline patch_v2"
    prefix = "live"
    suffix = f"yolo_{conf_thresh_downsized}_{conf_thresh_modelsized}_{nms_thresh}" # to append to output video filename
    
    # fixed variables
    class_names = open("coco-labels-2014_2017.txt", "r").readlines()

    weightfile = "weights/yolo.weights"

    if detect_on_downsized_img:
        downsized_model = Darknet(downsized_cfgfile)
        downsized_model.load_weights(weightfile)
        downsized_model = downsized_model.eval().cuda()

        model_img_width = downsized_model.width
        model_img_height = downsized_model.height
        downsized_resize = transforms.Resize((model_img_width,model_img_height))
    if detect_on_modelsized_img:
        darknet_model = Darknet(darknet_cfgfile)
        darknet_model.load_weights(weightfile)
        darknet_model = darknet_model.eval().cuda()

        model_img_width = darknet_model.width
        model_img_height = darknet_model.height
        modelsized_resize = transforms.Resize((model_img_width,model_img_height))

    if save_video:
        savedirs = [os.path.join(savedir)]
        # make saving directory if not exist
        for dir in savedirs:
            if not os.path.exists(dir):
                os.makedirs(dir)
    
    print("Done")

    # read in video from webcam
    cap = cv2.VideoCapture(0)

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
            # transforming the frame image, padding into a square
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
            
            line = 0
            plotted_img = add_text(padded_img, "Legend", frame_width, frame_height, line, rgb=(0,0,0)) # black
            line += 1
            if detect_on_downsized_img:
                resized_img = downsized_resize(padded_img)
                
                # detecting with model
                boxes = do_detect(downsized_model, resized_img, conf_thresh_downsized, nms_thresh, True)

                # plot bouding boxes
                color = (255,0,0) # blue
                plotted_img = plot_boxes(plotted_img, boxes, class_names=class_names, color=color)
                plotted_img = add_text(plotted_img, downsized_img_model_name, frame_width, frame_height, line, rgb=color)
                line += 1
            if detect_on_modelsized_img:
                resized_img = modelsized_resize(padded_img)

                # detecting with model
                boxes = do_detect(darknet_model, resized_img, conf_thresh_modelsized, nms_thresh, True)

                # plot bouding boxes
                color = (0,0,255) # red
                plotted_img = plot_boxes(plotted_img, boxes, class_names=class_names, color=color) 
                plotted_img = add_text(plotted_img, modelsized_img_model_name, frame_width, frame_height, line, rgb=color)
                line += 1
            # writing boxes to text file
            # textfile = open(txtpath,'w+')
            # for box in boxes:
            #     cls_id = box[6]
            #     if(cls_id == 0):   #if person
            #         x_center = box[0]
            #         y_center = box[1]
            #         width = box[2]
            #         height = box[3]
                    # textfile.write(f'{cls_id} {x_center} {y_center} {width} {height}\n')
            # textfile.close()

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
                cv2.imshow("Live", final_image)
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