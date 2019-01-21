
import numpy as np
import torch
from torch import nn
from torchvision import models
from torch.autograd import Variable
import cv2
import os
import copy





#The mean and standard deviation values for the imagenet dataset:  

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

	#if resize is true then resize image ---> (224,224)
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
    torch_im = Variable(tensor_image, requires_grad=True)
    return torch_im


# Inverts the image using the defined mean and std
def inverter(in_image):

    for i in range(3):
  	  in_image[i] = (in_image[i]/std_inverse[i])-mean_inverse[i]

    #clip everything to between 0 and 1
    in_image[in_image > 1] = 1
    in_image[in_image < 0] = 0
    in_image = np.round(in_image * 255)

    return in_image




# takes the outputted torch tensor and returns it back to its origional image format
def net_output_to_image(im_as_var):

    reciprocal_im=inverter(copy.copy(im_as_var.data.numpy()[0]))

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




#  ______ _____  _____ __  __           _   _             _    
# |  ____/ ____|/ ____|  \/  |     /\  | | | |           | |   
# | |__ | |  __| (___ | \  / |    /  \ | |_| |_ __ _  ___| | __
# |  __|| | |_ |\___ \| |\/| |   / /\ \| __| __/ _` |/ __| |/ /
# | |   | |__| |____) | |  | |  / ____ \ |_| || (_| | (__|   < 
# |_|    \_____|_____/|_|  |_| /_/    \_\__|\__\__,_|\___|_|\_\
                                                             
                                                             




#open the dictionary of all the labels for imagenet:

with open('../imagenet_labels.txt') as f:
  label_dict = dict(x.rstrip().split(None, 1) for x in f)

 #to access a labels meaning one would run the following ; (for label i)	
 # label_dict["i:"]


class FGSM():
	"""
		FGSM attack;  maximizes the target class activation
		with iterative grad sign updates - option to work in targeted or untargeted framework
	"""

	def __init__(self, model, alpha):
		self.model = model
		self.alpha = alpha
		self.model.eval()
		


	#is targeted is the choice between random class or targeted class attack
	def create_adversarial(self, original_image, origin_label, target_label,is_targeted,epochs=20,is_save=True): 
 

     #----------------------- [Targeted ----> Assign target_label ] ----------------

		#targeted 
		if is_targeted:
			im_label_as_var = Variable(torch.from_numpy(np.asarray([target_label]))) #change name of im_label_as_var


	#----------------------- [Untargeted ----> Assign as normal ] -----------------
		#untargeted
		else:
			im_label_as_var = Variable(torch.from_numpy(np.asarray([origin_label])))


		#load cross entropy
		cross_entr = nn.CrossEntropyLoss()

		# put image into the form for the cnn
		processed_image = image_to_cnn_input(original_image)
		


		# Create the folder to export images if not exists
		if not os.path.exists('../AlexNet_Adversarial_Images'):
			os.makedirs('../AlexNet_Adversarial_Images')

		#---- Begin Training-------

		for i in range(epochs):
	
			#reset gradients
			processed_image.grad = None
			
			#foward pass & calculate the loss
			foward_out = self.model(processed_image)
			orig_loss = cross_entr(foward_out, im_label_as_var)

			#print loss
			print('Iteration:', str(i)," origional_loss = " , orig_loss.item())


			# Compute backprogation
			orig_loss.backward()

			#create noise using update rule given in Paper
			adv_noise = self.alpha * torch.sign(processed_image.grad.data)


			# Perform the gradient step -- i.e Add noise to the image 
			if targeted_op:
				processed_image.data = processed_image.data - adv_noise
			else:
				processed_image.data = processed_image.data + adv_noise


			#ensure that image is indeed adversarial after recreating the image:

			# re-generated the image
			out_image = net_output_to_image(processed_image)
			prep_confirmation_image = image_to_cnn_input(out_image)

			# Pass the adversarial image through the net
			adv_out = self.model(prep_confirmation_image)

			# Predict class of adversarial image
			_, adv_prediction = adv_out.data.max(1)

			# Calculate probability of correctness
			adv_pred_confidence = \
			nn.functional.softmax(adv_out,0)[0][adv_prediction].data.numpy()[0]

			# Make tensor into an integer value
			adv_prediction = adv_prediction.numpy()[0]

			



#   ____        _               _       _______             _             
#  / __ \      | |             | |     / / ____|           (_)            
# | |  | |_   _| |_ _ __  _   _| |_   / / (___   __ ___   ___ _ __   __ _ 
# | |  | | | | | __| '_ \| | | | __| / / \___ \ / _` \ \ / / | '_ \ / _` |
# | |__| | |_| | |_| |_) | |_| | |_ / /  ____) | (_| |\ V /| | | | | (_| |
#  \____/ \__,_|\__| .__/ \__,_|\__/_/  |_____/ \__,_| \_/ |_|_| |_|\__, |
#                  | |                                               __/ |
#                  |_|                                              |___/ 




			#now it reamains to confirm that the predictions are indeed different

			#Not targeted
			if not targeted_op:
				if adv_prediction != origin_label:
					print('Alex net normally predicts image as:', origin_label," which represents ",label_dict["%d:" %origin_label],
						  'which after untargeted FGSM attack is predicted as:', adv_prediction,
						  " which represents ",label_dict["%d:" %adv_prediction],
						  'with a confidence value of:', adv_pred_confidence)
					# Create the image for noise as: Original image - generated image
					noise_image = original_image - out_image

					if is_save:
						cv2.imwrite('../AlexNet_Adversarial_Images/untargeted_adv_noise_from_' + str(label_dict["%d:" %origin_label]) + '_to_' +
									str(label_dict["%d:" %adv_prediction]) + '.jpg', noise_image)
						# Write image
						cv2.imwrite('../AlexNet_Adversarial_Images/untargeted_adv_img_from_' + str(label_dict["%d:" %origin_label]) + '_to_' +
									str(label_dict["%d:" %adv_prediction]) + '.jpg', out_image)
					break
				else:
					if i==epochs-1:
					
						print("adversarial training terminated without breaking prediction class")


			else: #Targeted -

				if adv_prediction == target_label:
					print('Alex net normally predicts image as :',  origin_label," which represents ",label_dict["%d:" %origin_label],
						  'which after targeted FGSM attack is predicted as:', adv_prediction,
						  " which represents ",label_dict["%d:" %adv_prediction],
						  'with a confidence value of:', adv_pred_confidence)

					# Noise= Original image - Pertubated image
					noise_image = original_image - out_image

					#write pertubated image and noise to file if option selected
					if is_save:

						cv2.imwrite('../AlexNet_Adversarial_Images/targeted_adv_noise_from_' + str(label_dict["%d:" %origin_label]) + '_to_' +
									str(label_dict["%d:" %adv_prediction]) + '.jpg', noise_image)
						# Write image
						cv2.imwrite('../AlexNet_Adversarial_Images/targeted_adv_img_from_' + str(label_dict["%d:" %origin_label]) + '_ to_' +
									str(label_dict["%d:" %adv_prediction]) + '.jpg', out_image)
						break
				
				#output failure message if does not reach target:
				else:	
	
					if i==epochs-1:

			
					
						print("adversarial training terminated without arriving at target class")

			
		return None






#  __  __       _       
# |  \/  |     (_)      
# | \  / | __ _ _ _ __  
# | |\/| |/ _` | | '_ \ 
# | |  | | (_| | | | | |
# |_|  |_|\__,_|_|_| |_|
                      
                      

if __name__ == '__main__':

	#Flags:

	#1) targeted (1/True) vs untargeted (0/False)
	targeted_op=True

	#2) Option to save image and noise generation in the adversarial folder
	save_op=True

	#) Number of epochs
	training_epochs=50


	#if we are doing targeted attack then select the class we wish to attack:
	target_label = 206 # choose from list https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a


	target_example = 4 #scuba
	(original_image, prep_img, origin_label, _, pretrained_model) = fetch_info(target_example)

	FGS = FGSM(pretrained_model, 0.01)
	FGS.create_adversarial(original_image, origin_label, target_label,targeted_op,training_epochs,save_op)


	#interestingly tabby cat when attacked naturally gradients into the egyptian cat category
	#however when we set an attack class to end up in we can force; 

