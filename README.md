# HackerEarth--Identify-the-dance-form

This is my solution to the deeplearning challenge on HackerEarth for Indentify the dance form. [Link](https://www.hackerearth.com/challenges/competitive/hackerearth-deep-learning-challenge-identify-dance-form/) to the challenge.

This solution uses fastaiv1 library available at: https://github.com/fastai/fastai
The library relies on pytorch 1.0 

List of applied transforms (data augmentations) on images-
1. Flip the images horizontally (randomly) with a probability of 0.5 (default)
2. Rotate the images randomly from -30 degrees to +30 degrees with probability of 0.75
3. Apply random zoom to images from 1.0 to 0.3 with probability of 0.75
4. Randomly change brightness of images. On a scale of 0 to 1, apply 0.2 with probability 0.75
   (0 means fully dark and 1 means fully white) (default)
5. Do not apply any perspective transformation since all images are front facing.
6. Apply affine transforms and symmetrix warps with probabilty 0.75 (default)
7. Resize images to (100,100)

All of the above transforms are applied to train data and only 7th transformation is applied to valid data.

Steps carried out-
Default mom is Point=(0.95, 0.85).
1. First train later layers of resnet 50 (5 epochs with fit one cycle policy)
2. Unfreeze whole model and train all layers again (5 epochs with fit one cycle policy)
3. Resize the image from 100 to 244 and start learning again on the model obtained from step 2 (train only later layers)
   (5 epochs with fit one cycle policy).
4. Unfreeze whole model and train all layers again (5 epochs and fit one cycle policy).
5. Step 5 and 6 same as 3 and 4 respectively with epoch count 3 and 1 respectively (resize from 244 to 299).

Managed to achieve F1 score of 87.25834 (offline score) and 85.19924(online score). Finally secured 79th spot out of 793 in online run.

Complete source in jupyter notebook.

