# Adversarial Pertubations via FGSM - Lenet5 and Alexnet

The following project implements FGSM attacks on the neural networks Lenet5 and Alexnet in Pytorch.


Example: An image originally classified as a beach by Alexnet:

<img src="https://github.com/LawrenceMMStewart/Adversarial_Attack/blob/master/Alex_net_input_images/beach.JPEG" width="250">

Which is attacked by the following adversarial pertubation:

<img src="https://github.com/LawrenceMMStewart/Adversarial_Attack/blob/master/AlexNet_Adversarial_Images/targeted_adv_noise_from_'seashore%2C%20coast%2C%20seacoast%2C%20sea-coast'%2C_to_'harvestman%2C%20daddy%20longlegs%2C%20Phalangium%20opilio'%2C.jpg" width="250">

will now be classified as a Golden Retriever:

<img src="https://github.com/LawrenceMMStewart/Adversarial_Attack/blob/master/AlexNet_Adversarial_Images/targeted_adv_img_from_'seashore%2C%20coast%2C%20seacoast%2C%20sea-coast'%2C_%20to_'harvestman%2C%20daddy%20longlegs%2C%20Phalangium%20opilio'%2C.jpg" width="250">


# Contents 

* **Alexnet** - Contains the untargeted and targeted FGSM attack on Alexnet - **FGSM_Alexnet.py** 
* **FGSM_Alexnet.py**  - An implementation of targeted and untargeted FGSM attack on Alexnet. Select an input image (and optional target class) and run whilst in the directory **Alexnet**. The code outputs pertubated image and noise will be stored in the folder **AlexNet_Adversarial_Images**
* **AlexNet_Adversarial_Images** - A folder that stores the outputted pertubated images generated by the FGSM attack
* **Alex_net_input_images** - Input images for alexnet and FGSM attack, all size 224x224. Feel free to add more images and experiement.
* **LeNet5.py** - Implementation of LeNet5 which when ran will store the weights after training in the folder **Stored_Weights**
* **FGSM_LeNet5.py** - FGSM attack on LeNet5 which generates images and plots in the folder **LenetImages**
* **LenetImages** - Folder of generated outputs from **FGSM_LeNet5.py** 
* **Stored_Weights** - Folder containing pretrained weights of **LeNet5.py**

# References:

* __Explaining and Harnessing Adversarial Examples - I. Goodfellow et. al__  https://arxiv.org/pdf/1412.6572.pdf
* __Pytorch Adversarial Example generation__ https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
* __Facebook imagenet (for channel mean and std)__ https://github.com/facebook/fb.resnet.torch/projects
* __Misc_functions__ image pre-processing tools taken from https://github.com/utkuozbulak/pytorch-cnn-adversarial-attacks/blob/master/src/misc_functions.py
* __Image-net labels__ https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
* __Adversarial Attacks and Defences for CNN's - Joao Jones blog__ https://medium.com/onfido-tech/adversarial-attacks-and-defences-for-convolutional-neural-networks-66915ece52e7

With thanks to Marc Lelarge and Andrei Bursuc.


