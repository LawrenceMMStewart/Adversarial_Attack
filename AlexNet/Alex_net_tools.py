
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import models

import copy
import cv2



#The mean and standard deviation values for the imagenet dataset:  https://github.com/facebook/fb.resnet.torch/projects

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


#inverse mean and standard deviaton used to go back from normalised to origional images
mean_inverse = -np.array(mean)
std_inverse = 1/np.array(std)


#Here once can control the input images, feel free to add your own 224x224 images in the following form
list_input_images = [["../Alex_net_input_images/radiator.JPEG",753 ],
                ['../Alex_net_input_images/beach.JPEG', 978],
                ['../Alex_net_input_images/tabbycat.JPEG', 281],
                ["../Alex_net_input_images/car.JPEG",817],
                ["../Alex_net_input_images/scubadiving.JPEG",983]
                ]



#  _____                       _               _______          _     
# |_   _|                     (_)             |__   __|        | |    
#   | |  _ __ ___   __ _  __ _ _ _ __   __ _     | | ___   ___ | |___ 
#   | | | '_ ` _ \ / _` |/ _` | | '_ \ / _` |    | |/ _ \ / _ \| / __|
#  _| |_| | | | | | (_| | (_| | | | | | (_| |    | | (_) | (_) | \__ \
# |_____|_| |_| |_|\__,_|\__, |_|_| |_|\__, |    |_|\___/ \___/|_|___/
#                         __/ |         __/ |                         
#                        |___/         |___/                          




#Note all imagenet pics for our attack are 244x244 dimension




#image_to_cnn_input takes an image to needs to be processed, the input_image and if fit_dim_to_224 is true resizes the image to dimension
#224 by 224 and then normalises the image and puts it into tensor format for alexnet to use.


def image_to_cnn_input(input_image, fit_dim_to_224=True):

	#if resize is true then resize image
    if fit_dim_to_224:
        input_image = cv2.resize(input_image, (224, 224))


    #make the image into an array:
    array_image = np.ascontiguousarray(np.float32(input_image)[..., ::-1])

    #convert the array to Depth, Width , Height
    array_image = array_image.transpose(2, 0, 1)  


    # Normalise each of the channels:
    for channel, _ in enumerate(array_image):

    	#x= (x/255-µ)/sigma
        array_image[channel] = ((array_image[channel]/255)-mean[channel])/std[channel]

    # Convert to a tensor
    tensor_image = torch.from_numpy(array_image).float()

    # We need image with tensor shape 1,3,224,224 so add one more channel as current shape is 3,224,224
    tensor_image.unsqueeze_(0)

    # Convert to Pytorch variable
    im_as_var = Variable(tensor_image, requires_grad=True)
    return im_as_var




# takes the outputted torch tensor and returns it back to its origional image format
def net_output_to_image(im_as_var):


    reciprocal_im = copy.copy(im_as_var.data.numpy()[0])
    for i in range(3):
        reciprocal_im[i] = (reciprocal_im[i]/std_inverse[i])-mean_inverse[i]

    #clip everything to between 0 and 1
    reciprocal_im[reciprocal_im > 1] = 1
    reciprocal_im[reciprocal_im < 0] = 0
    reciprocal_im = np.round(reciprocal_im * 255)


    #reshape image
    reciprocal_im = np.uint8(reciprocal_im).transpose(1, 2, 0)

    # Convert RBG to GBR
    reciprocal_im = reciprocal_im[..., ::-1]

    return reciprocal_im


"""
fetch info on a selected example returns the following items in a tuple:

1) the origional image in form as a numpy array
2) the image processed in the input form for alex net
3) The target label for the image (see the image net list to choose)
4)  String of file name to save images to
5) The loaded alexnet model:
"""
def fetch_info(selected_example):

    # Load Alexnet 
    pretrained_model = models.alexnet(pretrained=True)
 
    img_path = list_input_images[selected_example][0]
    target_label = list_input_images[selected_example][1]
    file_name_to_export = img_path[img_path.rfind('/')+1:img_path.rfind('.')]
    
    # Read image
    og_im = cv2.imread(img_path, 1)

    # switch image format to that compatible with alex net
    preprocessed_image = image_to_cnn_input(og_im)

    #return a tuple of all information needed
    return (og_im,preprocessed_image, target_label, file_name_to_export,pretrained_model)
