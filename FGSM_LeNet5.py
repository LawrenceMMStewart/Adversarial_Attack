#  ______ _____  _____ __  __           _   _             _    
# |  ____/ ____|/ ____|  \/  |     /\  | | | |           | |   
# | |__ | |  __| (___ | \  / |    /  \ | |_| |_ __ _  ___| | __
# |  __|| | |_ |\___ \| |\/| |   / /\ \| __| __/ _` |/ __| |/ /
# | |   | |__| |____) | |  | |  / ____ \ |_| || (_| | (__|   < 
# |_|    \_____|_____/|_|  |_| /_/    \_\__|\__\__,_|\___|_|\_\
		   


															 
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt







#load pretrained model 
#-- c h a n g e   t h i s   t o    y o u r    s h o r t c u t 

lenet="/Users/lawrence/Desktop/DLProject/Stored_Weights/LeNet_Weights.pt"
use_cuda=False





#  _          _   _      _          _____ 
# | |        | \ | |    | |        | ____|
# | |     ___|  \| | ___| |_ ______| |__  
# | |    / _ \ . ` |/ _ \ __|______|___ \ 
# | |___|  __/ |\  |  __/ |_        ___) |
# |______\___|_| \_|\___|\__|      |____/ 
	  


#Redefine the Lnet to have access to gradients:


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 20, 5, 1)
		self.conv2 = nn.Conv2d(20, 50, 5, 1)
		self.fc1 = nn.Linear(4*4*50, 500)
		self.fc2 = nn.Linear(500, 10)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2, 2)
		x = x.view(-1, 4*4*50)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)






# GPU vs CPU option
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")



# Import the Mnist dataset
test_loader = torch.utils.data.DataLoader(
	datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
			transforms.ToTensor(),
			])), 
		batch_size=1, shuffle=True)



#define model
model = Net().to(device)

# Load the Lnet
model.load_state_dict(torch.load(lenet, map_location='cpu'))

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()





# FGSM function --


"""
recall that the attack takes place using the following update rule:

adversarial image to train  =image + eps * sign(data_grad) =  x+  ϵ*sign(∇xJ(θ,x,y))
"""



#          _   _             _       __                  _   _                 
#     /\  | | | |           | |     / _|                | | (_)                
#    /  \ | |_| |_ __ _  ___| | __ | |_ _   _ _ __   ___| |_ _  ___  _ __  ___ 
#   / /\ \| __| __/ _` |/ __| |/ / |  _| | | | '_ \ / __| __| |/ _ \| '_ \/ __|
#  / ____ \ |_| || (_| | (__|   <  | | | |_| | | | | (__| |_| | (_) | | | \__ \
# /_/    \_\__|\__\__,_|\___|_|\_\ |_|  \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
                                                                             
                                                                             



def fgsm(im, eps, data_grad,):

	sign_data_grad = data_grad.sign()

	# Create the attack image by incrementing each pixel of input by grad
	perturbed_im = im + eps*sign_data_grad

	# Ensure that the image stays within set range i..e 0 to 1 
	perturbed_im = torch.clamp(perturbed_im, 0, 1)

	# Output modified image 
	return perturbed_im



def adv_generator( model, device, test_loader, eps ):

	# Accuracy counter
	correct = 0
	attack_examples = []

	# Loop over all examples in test set
	for data, target in test_loader:

		# Send the data and label to the device
		data, target = data.to(device), target.to(device)

		# Set requires_grad attribute of tensor. Important for Attack
		data.requires_grad = True

		# Forward pass the data through the model
		output = model(data)
		init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

		# If the initial prediction is wrong, dont bother attacking, just move on
		if init_pred.item() != target.item():
			continue

		# Calculate the loss
		loss = F.nll_loss(output, target)

		# Zero all existing gradients
		model.zero_grad()

		# Calculate gradients of model in backward pass
		loss.backward()

		# Collect datagrad
		data_grad = data.grad.data

		# Call FGSM
		attack_data = fgsm(data, eps, data_grad)

		# Re-classify the new adversarial image
		output = model(attack_data)

		# Check for success
		final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
		if final_pred.item() == target.item():
			correct += 1
			# Special case for saving 0 eps examples
			if (eps == 0) and (len(attack_examples) < 5):
				adv_ex = attack_data.squeeze().detach().cpu().numpy()
				attack_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
		else:
			# Save some adv examples for visualization later
			if len(attack_examples) < 5:
				adv_ex = attack_data.squeeze().detach().cpu().numpy()
				attack_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

	# Calculate final accuracy for this eps
	final_acc = correct/float(len(test_loader))
	print("eps: {}\tTest Accuracy = {} / {} = {}".format(eps, correct, len(test_loader), final_acc))

	# Return the accuracy and an adversarial example
	return final_acc, attack_examples






#   _____            _             _   _____                 _ 
#  / ____|          | |           | | |  __ \               | |
# | |     ___  _ __ | |_ _ __ ___ | | | |__) |_ _ _ __   ___| |
# | |    / _ \| '_ \| __| '__/ _ \| | |  ___/ _` | '_ \ / _ \ |
# | |___| (_) | | | | |_| | | (_) | | | |  | (_| | | | |  __/ |
#  \_____\___/|_| |_|\__|_|  \___/|_| |_|   \__,_|_| |_|\___|_|
															 
															 


if __name__ == '__main__':

	#Define options for display and printing all booleans:

	#1) generate attack examples
	attack_op=True

	#2) display graph of epsilons vs accuracy
	epsilons_op= True

	#3) Display generated adverarial images and their misclassified labels for various epsilons
	displ_op=True

	#4) save results:
	save_im_op=True

	#5) show plots:
	show_plt_op=False





	#epsilon values that we wish to use (where e=0 is no attack):
	epsilon_vals=np.linspace(0,0.5,5)



	#directory to save images to:
	im_direc="LeNetImages/"
	





	if attack_op:

		print(" Beginning adversarial FGSM attack on Lenet-5 ")

		accuracies = []
		examples = []

		# Run test for each epsilon
		for eps in epsilon_vals:
			acc, ex = adv_generator(model, device, test_loader, eps)
			accuracies.append(acc)
			examples.append(ex)




		if epsilons_op:
			plt.figure(figsize=(5,5))
			plt.plot(epsilon_vals, accuracies, "*-",alpha=0.7)
			plt.yticks(np.arange(0, 1.1, step=0.1))
			plt.xticks(np.arange(0, .35, step=0.05))
			plt.title("Accuracy vs Epsilon")
			plt.xlabel("Epsilon")
			plt.ylabel("Accuracy")

			if show_plt_op:
				plt.show()
			if save_im_op:
				plt.savefig(im_direc + 'epsilons_vs_acc.png', dpi = 300)


			
		if displ_op:
			cnt = 0
			plt.figure(figsize=(12,19))
			plt.suptitle("Generated Adversarial Images", fontsize=10)
			for i in range(len(epsilon_vals)):
				for j in range(len(examples[i])):
					cnt += 1
					plt.subplot(len(epsilon_vals),len(examples[0]),cnt)
					plt.xticks([], [])
					plt.yticks([], [])
					if j == 0:
						plt.ylabel("Eps,Acc: {}".format((epsilon_vals[i],accuracies[i])), fontsize=10)
					orig,adv,ex = examples[i][j]
					plt.title("{} -> {}".format(orig, adv))
					plt.imshow(ex, cmap="gray")

			plt.tight_layout()
			if show_plt_op:
				plt.show()

			if save_im_op:
				plt.savefig(im_direc + 'Generated_adversarials.png', dpi = 300)















#  _   _       _            
# | \ | |     | |           
# |  \| | ___ | |_ ___  ___ 
# | . ` |/ _ \| __/ _ \/ __|
# | |\  | (_) | ||  __/\__ \
# |_| \_|\___/ \__\___||___/
                          
                          

# the structure of examples is a list of eps elemenets

#take the any element (is actually a tuple where), the first element is the number that it is
#(i.e its mnist number, then the second number is the misidentification.) the 3rd element is
#the image a 28x28 array.