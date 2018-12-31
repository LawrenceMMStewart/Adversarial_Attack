
from torch import nn
from torch.autograd import Variable

import numpy as np
import torch
import cv2
import os


from Alex_net_tools import image_to_cnn_input
from Alex_net_tools import net_output_to_image
from Alex_net_tools import  fetch_info





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
		
		# Create the folder to export images if not exists
		if not os.path.exists('../AlexNet_Adversarial_Images'):
			os.makedirs('../AlexNet_Adversarial_Images')

	#is targeted is the choice between random class or targeted class attack
	def create_adversarial(self, original_image, origin_label, target_label,is_targeted,epochs=20,is_save=True): 
 

		#targeted 
		if is_targeted:
			im_label_as_var = Variable(torch.from_numpy(np.asarray([target_label])))

		#untargeted
		else:
			im_label_as_var = Variable(torch.from_numpy(np.asarray([origin_label])))


		# Define loss functions
		cross_entr = nn.CrossEntropyLoss()
		# Process image
		processed_image = image_to_cnn_input(original_image)
		# Start iteration
		for i in range(epochs):
	
			#reset gradients
			processed_image.grad = None
			
			#foward pass
			out = self.model(processed_image)

			# Calculate CE loss
			orig_loss = cross_entr(out, im_label_as_var)

			#print loss
			print('Iteration:', str(i)," origional_loss = " , orig_loss.item())


			# Compute backprogation
			orig_loss.backward()

			#create noise using update rule given in Paper
			adv_noise = self.alpha * torch.sign(processed_image.grad.data)


			# Add noise to processed image
			if targeted_op:
				processed_image.data = processed_image.data - adv_noise
			else:
				processed_image.data = processed_image.data + adv_noise


			#ensure that image is indeed adversarial after recreating the image:

			# Generate confirmation image
			recreated_image = net_output_to_image(processed_image)

			# Process confirmation image
			prep_confirmation_image = image_to_cnn_input(recreated_image)

			# Forward pass
			confirmation_out = self.model(prep_confirmation_image)

			# Predict class of adversarial image
			_, confirmation_prediction = confirmation_out.data.max(1)

			# Calculate probability of correctness
			confirmation_confidence = \
			nn.functional.softmax(confirmation_out,0)[0][confirmation_prediction].data.numpy()[0]

			# Make tensor into an integer value
			confirmation_prediction = confirmation_prediction.numpy()[0]

			



			#now it reamains to confirm that the predictions are indeed different

			#Not targeted
			if not targeted_op:
				if confirmation_prediction != origin_label:
					print('Alex net normally predicts image as:', origin_label," which represents ",label_dict["%d:" %origin_label],
						  'which after untargeted FGSM attack is predicted as:', confirmation_prediction,
						  " which represents ",label_dict["%d:" %confirmation_prediction],
						  'with a confidence value of:', confirmation_confidence)
					# Create the image for noise as: Original image - generated image
					noise_image = original_image - recreated_image

					if is_save:
						cv2.imwrite('../AlexNet_Adversarial_Images/untargeted_adv_noise_from_' + str(label_dict["%d:" %origin_label]) + '_to_' +
									str(label_dict["%d:" %confirmation_prediction]) + '.jpg', noise_image)
						# Write image
						cv2.imwrite('../AlexNet_Adversarial_Images/untargeted_adv_img_from_' + str(label_dict["%d:" %origin_label]) + '_to_' +
									str(label_dict["%d:" %confirmation_prediction]) + '.jpg', recreated_image)
					break
				else:
					if i==epochs-1:
					
						print("adversarial training terminated without breaking prediction class")


			else: #Targeted -

				if confirmation_prediction == target_label:
					print('Alex net normally predicts image as :',  origin_label," which represents ",label_dict["%d:" %origin_label],
						  'which after targeted FGSM attack is predicted as:', confirmation_prediction,
						  " which represents ",label_dict["%d:" %confirmation_prediction],
						  'with a confidence value of:', confirmation_confidence)

					# Noise= Original image - Pertubated image
					noise_image = original_image - recreated_image

					#write pertubated image and noise to file if option selected
					if is_save:

						cv2.imwrite('../AlexNet_Adversarial_Images/targeted_adv_noise_from_' + str(label_dict["%d:" %origin_label]) + '_to_' +
									str(label_dict["%d:" %confirmation_prediction]) + '.jpg', noise_image)
						# Write image
						cv2.imwrite('../AlexNet_Adversarial_Images/targeted_adv_img_from_' + str(label_dict["%d:" %origin_label]) + '_ to_' +
									str(label_dict["%d:" %confirmation_prediction]) + '.jpg', recreated_image)
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

	target_label = 70  # Rock Python


	target_example = 0# Apple
	(original_image, prep_img, origin_label, _, pretrained_model) =\
		fetch_info(target_example)

	FGS = FGSM(pretrained_model, 0.01)
	FGS.create_adversarial(original_image, origin_label, target_label,targeted_op,training_epochs,save_op)

		

	#interestingly tabby cat when attacked naturally gradients into the egyptian cat category
	#however when we set an attack class to end up in we can force; 

