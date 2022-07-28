
# Adversarial YOLO

This repository is based on the marvis YOLOv2 inplementation: https://github.com/marvis/pytorch-yolo2

This work corresponds to the following paper: https://arxiv.org/abs/1904.08653:
```
@inproceedings{thysvanranst2019,
    title={Fooling automated surveillance cameras: adversarial patches to attack person detection},
    author={Thys, Simen and Van Ranst, Wiebe and Goedem\'e, Toon},
    booktitle={CVPRW: Workshop on The Bright and Dark Sides of Computer Vision: Challenges and Opportunities for Privacy and Security},
    year={2019}
}
```

If you use this work, please cite this paper.

## Setting up

We use Python 3.6.
Make sure that you have a working implementation of PyTorch installed, to do this see: https://pytorch.org/
To visualise progress we use tensorboardX which can be installed using pip:

```sh
pip install tensorboardX tensorboard
```

No installation is necessary, you can simply run the python code straight from this directory.

Make sure you have the YOLOv2 MS COCO weights:

```sh
mkdir weights; curl https://pjreddie.com/media/files/yolov2.weights -o weights/yolo.weights
```

Get the INRIA dataset:

```sh
curl ftp://ftp.inrialpes.fr/pub/lear/douze/data/INRIAPerson.tar -o inria.tar
tar xf inria.tar
mv INRIAPerson inria
cp -r yolo-labels inria/Train/pos/
```

## Generating a patch

`patch_config.py` contains configuration of different experiments. You can design your own experiment by inheriting from the base `BaseConfig` class or an existing experiment. `ReproducePaperObj` reproduces the patch that minimizes object score from the paper (With a lower batch size to fit on a desktop GPU).

Generating a patch can be done in an unsupervised manner. Requirements:

- Dataset containing the target class you want to mask. This can be a video as well, and the dataset does not need to be labelled as "labelling" is done by the model.
- Access to the model you want to attack.

The interim patch generated at the end of each epoch will be saved to [/pics](/pics/). Once you have tested and are satisfied with the performance of the generated patch, you can make a copy of the working patches into [/patches](/patches/).

You can generate this patch by running:

```sh
python train_patch.py paper_obj
```

You also have the option to selecting the model to train a patch for by selecting the model's config and weights in `self.darknet_model`. There is a list of available model config files [here](./cfg/) and model weight files [here](./weights/). I have had various degrees of success with the various models. This should work with other darknet models.

## Testing

Testing of the patch is done without the need for supervision. Similar to how patch generation works, first, the model is ran on the test images to plot the bounding boxes. The patch is then transformed and applied onto the bounding boxes of the target class. The model is then ran on the new images which contains the patches, and the percentage of the target classes which have been masked is displayed.

**For testing the performance of the patch on a test dataset of images, run:**

```sh
python test_patch.py
```

Note that multiple patches can be tested in the same run by adding their file path to the `patchdir` and `patchfiles` variables.

There is also the option of testing on different models by modifying `cfgfile` and `weightfile` variables.

The method of transforming and applying the patch can be modified by modifying `patch_applier` and `patch_transformer` respectively.

Summary of testing results on various patches can be found [here](./testing/results.xlsx).

**For testing on live video feed from a webcam:**

```sh
python test_video_live.py
```

**For testing on a folder containing video files of the patch**, modify `viddir` and `savedir` variables in `test_video.py`, and run:

```sh
python test_video.py
```
