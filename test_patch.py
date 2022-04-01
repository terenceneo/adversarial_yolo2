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

import torchvision.models as models


if __name__ == '__main__':
    print("Setting everything up")
    imgdir = "inria/Test/pos"
    # imgdir = "testing/testimages"
    # cfgfile = "cfg/yolo.cfg"
    # weightfile = "weights/yolo.weights"
    
    cfgfile = "cfg/resnet18.cfg"
    weightfile = "weights/resnet18.weights"

    patchdir = "pics/"
    # patchfiles = {"binoculars": "binoculars.jpg",
    # "cello": "cello__1_.jpg",
    # "class_detection": "class_detection.png",
    # "class_only": "class_only.png",
    # "electric_fan": "electric_fan.jpg",
    # "horns_dream": "horns_dream.jpg",
    # "lime_dream": "lime_dream.jpg",
    # "object_upper": "object_upper.png",
    # "tick": "tick.jpg"
    # }
    # patchfiles = {"2": "20220223-161301_ObjectOnlyPaper_2_1.106204867362976.jpg",
    # "150": "20220223-161301_ObjectOnlyPaper_150_0.7733585238456726.jpg",
    # "circle": "toast_270.png",
    # "original": "object_score.png"}
    # patchfiles = {"masked_scaled_white": "masked_patch_scaled_white.jpg",
    # "masked_inverse_scaled_white": "masked_patch_inverse_scaled_white.jpg"}
    # patchfiles = {"masked_trained_460": "masked/20220301-150155_ObjectOnlyPaper_460_0.7586309313774109.jpg",
    # "masked_trained_500": "masked/20220301-150155_ObjectOnlyPaper_500_0.7578548789024353.jpg",
    # "masked_trained_550": "masked/20220301-150155_ObjectOnlyPaper_550_0.7591909170150757.jpg",
    # "masked_trained_580": "masked/20220301-150155_ObjectOnlyPaper_580_0.7569643259048462.jpg",
    # "masked_trained_600": "masked/20220301-150155_ObjectOnlyPaper_600_0.7555060386657715.jpg",
    # "masked_trained_650": "masked/20220301-150155_ObjectOnlyPaper_650_0.7541428804397583.jpg",
    # "masked_trained_680": "masked/20220301-150155_ObjectOnlyPaper_680_0.7520384192466736.jpg",
    # "masked_trained_700": "masked/20220301-150155_ObjectOnlyPaper_700_0.75697922706604.jpg",
    # "masked_trained_1090": "masked/20220301-150155_ObjectOnlyPaper_1090_0.7551037073135376.jpg"
    # }

    # patchfiles = {"masked_trained_220": "masked/20220301-150155_ObjectOnlyPaper_220_0.9231265187263489.jpg",
    # "masked_trained_300": "masked/20220301-150155_ObjectOnlyPaper_300_0.7750753164291382.jpg",
    # "masked_trained_330": "masked/20220301-150155_ObjectOnlyPaper_330_0.7638962864875793.jpg",
    # "masked_trained_360": "masked/20220301-150155_ObjectOnlyPaper_360_0.7627858519554138.jpg",
    # "masked_trained_390": "masked/20220301-150155_ObjectOnlyPaper_390_0.7624249458312988.jpg"
    # }

    # patchfiles = {"masked_trained_1841": "masked/20220301-150155_ObjectOnlyPaper_1841_0.7546290159225464.jpg"}

    # patchfiles = {"merge_masked_550": "merge_masked_550.jpg",
    # "merge_both_550_382": "merge_both_550_382.jpg",
    # "merge_inverse_382": "merge_inverse_382.jpg"}
    patchfiles = {"masked_outline_582": "masked_outline/20220314-150408_ObjectOnlyPaper_582_0.8545225262641907.jpg"} # change this
    

    # patchfiles = {"object_class": "patches/adversarial_patches/1901.bmp"}

    # "masked_train_overlay": "masked/20220228-223836_ObjectOnlyPaper_3_0.8943374156951904.jpg"}
    # patchfiles = {2: "20220223-161301_ObjectOnlyPaper_2_1.106204867362976.jpg",
    # 100: "20220223-161301_ObjectOnlyPaper_100_0.7749974727630615.jpg", 
    # 150: "20220223-161301_ObjectOnlyPaper_150_0.7733585238456726.jpg"} # key is the epoch number
    # patchfile = "/home/wvr/Pictures/individualImage_upper_body.png"
    #patchfile = "/home/wvr/Pictures/class_only.png"
    #patchfile = "/home/wvr/Pictures/class_transfer.png"
    savedir = "testing/labelled_masked_outline" # change this
    conf_thresh = 0.6
    nms_thresh = 0.4
    class_names = open("coco-labels-2014_2017.txt", "r").readlines()

    # darknet_model = Darknet(cfgfile)
    # darknet_model.load_weights(weightfile)
    darknet_model = models.resnet18(pretrained=True)
    darknet_model = darknet_model.eval().cuda()
    patch_applier = PatchApplier().cuda()
    patch_transformer = PatchTransformer().cuda()

    batch_size = 1
    max_lab = 14
    # img_size = darknet_model.height
    img_size = 300

    patch_size = 300
    
    savedirs = [os.path.join(savedir, 'clean/', 'yolo-labels/'),
    os.path.join(savedir, 'random_patched/', 'yolo-labels/')]

    adv_patchs = {}

    clean_results = []
    noise_results = []
    patch_results = {}
    results_stats = {"clean":0}
    for e in patchfiles:
        results_stats[e] = 0
    
    for e in patchfiles:
        savedirs.append(os.path.join(savedir, f'proper_patched_{e}/', 'yolo-labels/'))
        
        # create patches
        patchfile = patchdir + patchfiles[e]
        patch_img = Image.open(patchfile).convert('RGB')
        tf = transforms.Resize((patch_size,patch_size))
        patch_img = tf(patch_img)
        tf = transforms.ToTensor()
        adv_patch_cpu = tf(patch_img)
        adv_patchs[e] = adv_patch_cpu.cuda()

        # for saving boxes
        patch_results[e] = []

    # make saving directory if not exist
    for dir in savedirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
    
    print("Done")
    #Loop over cleane beelden
    for imgfile in tqdm(os.listdir(imgdir)):
        # print("\nnew image")
        if imgfile.endswith('.jpg') or imgfile.endswith('.png'):
            name = os.path.splitext(imgfile)[0]    #image name w/o extension
            txtname = name + '.txt'
            txtpath = os.path.abspath(os.path.join(savedir, 'clean/', 'yolo-labels/', txtname))
            # open beeld en pas aan naar yolo input size
            # pad image
            imgfile = os.path.abspath(os.path.join(imgdir, imgfile))
            img = Image.open(imgfile).convert('RGB')
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
            resize = transforms.Resize((img_size,img_size))
            padded_img = resize(padded_img)
            cleanname = name + ".png"
            #sla dit beeld op
            # padded_img.save(os.path.join(savedir, 'clean/', cleanname))
            
            # clean
            #genereer een label file voor het gepadde beeld
            boxes = do_detect(darknet_model, padded_img, conf_thresh, nms_thresh, True)
            textfile = open(txtpath,'w+')
            for box in boxes:
                cls_id = box[6]
                if(cls_id == 0):   #if person
                    x_center = box[0]
                    y_center = box[1]
                    width = box[2]
                    height = box[3]
                    textfile.write(f'{cls_id} {x_center} {y_center} {width} {height}\n')
                    clean_results.append({'image_id': name, 'bbox': [x_center.item() - width.item() / 2,
                                                                    y_center.item() - height.item() / 2,
                                                                    width.item(),
                                                                    height.item()],
                                        'score': box[4].item(),
                                        'category_id': 1})
                    results_stats["clean"] += 1
            textfile.close()

            # plot bouding boxes
            print("\nClean \t\t\timage:", cleanname, "\tBoxes:", len(boxes))
            plot_boxes(padded_img, boxes, savename=os.path.join(savedir, 'clean/', cleanname), class_names=class_names)

            #lees deze labelfile terug in als tensor            
            if os.path.getsize(txtpath):       #check to see if label file contains data. 
                label = np.loadtxt(txtpath)
            else:
                label = np.ones([5])
            label = torch.from_numpy(label).float()
            if label.dim() == 1:
                label = label.unsqueeze(0)

            
            transform = transforms.ToTensor()
            padded_img = transform(padded_img).cuda()
            img_fake_batch = padded_img.unsqueeze(0)
            lab_fake_batch = label.unsqueeze(0).cuda()

            # proper patched
            for e in patchfiles:
                #transformeer patch en voeg hem toe aan beeld
                adv_patch = adv_patchs[e]
                adv_batch_t = patch_transformer(adv_patch, lab_fake_batch, img_size, do_rotate=True, rand_loc=False)
                p_img_batch = patch_applier(img_fake_batch, adv_batch_t)
                p_img = p_img_batch.squeeze(0)
                p_img_pil = transforms.ToPILImage('RGB')(p_img.cuda())
                properpatchedname = name + "_p.png"
                # p_img_pil.save(os.path.join(savedir, f'proper_patched_{e}/', properpatchedname))
                
                #genereer een label file voor het beeld met sticker
                txtname = properpatchedname.replace('.png', '.txt')
                txtpath = os.path.abspath(os.path.join(savedir, f'proper_patched_{e}/', 'yolo-labels/', txtname))
                boxes = do_detect(darknet_model, p_img_pil, conf_thresh, nms_thresh, True)
                textfile = open(txtpath,'w+')
                for box in boxes:
                    cls_id = box[6]
                    if(cls_id == 0):   #if person
                        x_center = box[0]
                        y_center = box[1]
                        width = box[2]
                        height = box[3]
                        textfile.write(f'{cls_id} {x_center} {y_center} {width} {height}\n')
                        patch_results[e].append({'image_id': name, 'bbox': [x_center.item() - width.item() / 2, y_center.item() - height.item() / 2, width.item(), height.item()], 'score': box[4].item(), 'category_id': 1})
                        results_stats[e] += 1
                textfile.close()
                
                # plot bouding boxes
                print("Patch:", e, "\timage:", properpatchedname, "\tBoxes:", len(boxes))
                plot_boxes(p_img_pil, boxes, savename=os.path.join(savedir, f'proper_patched_{e}/', properpatchedname), class_names=class_names)

            # Random patch
            #maak een random patch, transformeer hem en voeg hem toe aan beeld
            random_patch = torch.rand(adv_patch_cpu.size()).cuda()
            adv_batch_t = patch_transformer(random_patch, lab_fake_batch, img_size, do_rotate=True, rand_loc=False)
            p_img_batch = patch_applier(img_fake_batch, adv_batch_t)
            p_img = p_img_batch.squeeze(0)
            p_img_pil = transforms.ToPILImage('RGB')(p_img.cuda())
            properpatchedname = name + "_rdp.png"
            # p_img_pil.save(os.path.join(savedir, 'random_patched/', properpatchedname))
            
            #genereer een label file voor het beeld met random patch
            txtname = properpatchedname.replace('.png', '.txt')
            txtpath = os.path.abspath(os.path.join(savedir, 'random_patched/', 'yolo-labels/', txtname))
            boxes = do_detect(darknet_model, p_img_pil, conf_thresh, nms_thresh, True)
            textfile = open(txtpath,'w+')
            for box in boxes:
                cls_id = box[6]
                if(cls_id == 0):   #if person
                    x_center = box[0]
                    y_center = box[1]
                    width = box[2]
                    height = box[3]
                    textfile.write(f'{cls_id} {x_center} {y_center} {width} {height}\n')
                    noise_results.append({'image_id': name, 'bbox': [x_center.item() - width.item() / 2, y_center.item() - height.item() / 2, width.item(), height.item()], 'score': box[4].item(), 'category_id': 1})
            textfile.close()

            # plot bouding boxes
            print("Patch: random", "\t\timage:", properpatchedname, "\tBoxes:", len(boxes))
            plot_boxes(p_img_pil, boxes, savename=os.path.join(savedir, 'random_patched/', properpatchedname), class_names=class_names)

    with open(os.path.join(savedir,'clean_results.json'), 'w') as fp:
        json.dump(clean_results, fp)
    with open(os.path.join(savedir,'noise_results.json'), 'w') as fp:
        json.dump(noise_results, fp)
    for e in patch_results:
        with open(os.path.join(savedir,f'patch_results_{e}.json'), 'w') as fp:
            json.dump(patch_results[e], fp)

    for key in patchfiles:
        if key == "clean":
            print(key, results_stats[key])
        else:
            print(f"{key}\tpeople detected:{results_stats[key]}\t% people hidden: {(1-(results_stats[key]/results_stats['clean']))*100} %")
            

