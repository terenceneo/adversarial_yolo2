from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import cv2 
from utils import *
from darknet import Darknet
# from preprocess import prep_image, inp_to_image, letterbox_image
import pandas as pd
import random 
import pickle as pkl
import argparse

video_file = "../../data/recording/WIN_20220317_10_41_16_Pro.mp4"
savedir = "../../data/video_out"

def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
    # img = cv2.resize(img, (input_dim, input_dim)) 
    # img_ =  img[:,:,::-1].transpose((2,0,1))
    # img_ = img_[np.newaxis,:,:,:]/255.0
    # img_ = torch.from_numpy(img_).float()
    # img_ = Variable(img_)
    
    padded_img, orig_im, dim = prep_image(img, input_dim)


    if CUDA:
        padded_img = padded_img.cuda()
    
    return padded_img

def prep_image(img, img_size):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    # orig_im = img
    # dim = orig_im.shape[1], orig_im.shape[0]
    # img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    # img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    # img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    # return img_, orig_im, dim
    w,h,c = img.shape
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
    resize = transforms.Resize((img_size,img_size))
    padded_img = resize(padded_img)
    return padded_img, img, img.size

def write(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')
   
    parser.add_argument("--video", dest = 'video', help = 
                        "Video to run detection upon",
                        default = video_file, type = str)
    parser.add_argument("--dataset", dest = "dataset", help = "Dataset on which the network has been trained", default = "pascal")
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.6) # original was 0.5
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolo.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "weights/yolo.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    classes = open("coco-labels-2014_2017.txt", "r").readlines()
    CUDA = torch.cuda.is_available()

    num_classes = 80

    CUDA = torch.cuda.is_available()
    
    bbox_attrs = 5 + num_classes
    
    print("Loading network.....", args.cfgfile)
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    # model.net_info["height"] = args.reso
    # inp_dim = int(model.net_info["height"])
    model.height = args.reso
    inp_dim = int(model.height)
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    if CUDA:
        model.cuda()
    
    # testing image
    # model(get_test_input(inp_dim, CUDA), CUDA)
    # model.eval()
    padded_img = get_test_input(inp_dim, CUDA)

    boxes = do_detect(model, padded_img, confidence, nms_thesh, True)
    plot_boxes(padded_img, boxes, savename=os.path.join(savedir, "test"), class_names=classes)
    
    videofile = args.video
    
    cap = cv2.VideoCapture(videofile)
    
    assert cap.isOpened(), 'Cannot capture source'
    
    frames = 0
    start = time.time()    
    while cap.isOpened():
        
        ret, frame = cap.read()
        if ret:
            

            img, orig_im, dim = prep_image(frame, inp_dim)
            
            im_dim = torch.FloatTensor(dim).repeat(1,2)                        
            
            
            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()
            
            # with torch.no_grad():   
            #     # output = model(Variable(img), CUDA)
            #     output = model(Variable(img))
            # output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

            boxes = do_detect(model, padded_img, confidence, nms_thesh, True)

            # if type(output) == int:
            #     frames += 1
            #     print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
            #     cv2.imshow("frame", orig_im)
            #     key = cv2.waitKey(1)
            #     if key & 0xFF == ord('q'):
            #         break
            #     continue
            
            

            
            # im_dim = im_dim.repeat(output.size(0), 1)
            # scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)
            
            # output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
            # output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
            
            # output[:,1:5] /= scaling_factor
    
            # for i in range(output.shape[0]):
            #     output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
            #     output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
            
            colors = pkl.load(open("pallete", "rb"))
            
            list(map(lambda x: write(x, orig_im), output))
            
            
            cv2.imshow("frame", orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))

            
        else:
            break
    

    
    

