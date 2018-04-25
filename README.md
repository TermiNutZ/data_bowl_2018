# Data Science Bowl 2018
This is selvar medal solution for Data Science Bowl 2018.
The task was to segment nuclei on images.

NN Framework: Keras

I used U-Net architechture to approach this problem. Instead of encoder part I used transfer learning with VGG16 pretrained model. (model.py) 

I predicted 3 chanell mask: [mask of nuclei, contour of nuclei, inner of nuclei]. (train.py)
For training I used different augmentation as: RandomCrop, Rotation, Contrast, Brightness. (augment.py)

Then I used watershed transform with predicted inner part as a marker. (postprocess.py)
