Python functions allowing for faster building and
training simple feed forward neural networks implemented
eith Tensorflow. 
Built with Python 3.6.3, Tensorflow 1.6.0.

Feel free to use the code however you want if you find it helpful.

Code written in response to the YT video:
https://www.youtube.com/watch?v=BhpvH5DuVu8&list=PLSPWNkAMSvv5DKeSVDbEbUKSsK4Z-GgiP&index=3

F.A.Szombara


neural-net-func.py
FUNCTIONS:

construct_neural_net(layers_nodes, input_size, output_size):
	
	Creates layers of a neural network and returns them in a form 
	of dictionary

	Parameters:
	layers_nodes: a list of numbers of neurons for each layer of the net
	eg. [500,500,500]- 3 layers, 500 neuons each

	input_size: length of the input vector

	output_size: size of the output vector

	Returns:
	dictionary where keys are the names of each layer in the format:
	hidden_lX, where X is the numer of a layer
	eg. "hidden_l1"- key of the first hidden layer
		"hidden_l0"- key of the input layer
	each layer is represented as a dictionary where "weights" is the key of
	weights and "biases" is the key of biases
	

neural_network_model(x, neural_net):
	
	Defines the operations in the neural network and returns the output layer

	Parameters:

	x: tf placeholder for the data

	neural_net: neural network weights and biases in a form od a dictionary
	(function construct_neural_net is design fot creating such a dictionary)

train_neural_network(x, y, neural_net, hm_epochs=10, save_path=""):
	
	Trains a neural network, prints the final accuracy score. Saves
	the network if save_path parameter is provided

	Parameters:

	x: tf placeholder for the data

	y: tf placeholder for the output vector

	neural_net: neural network weights and biases in a form od a dictionary
	(function construct_neural_net is design fot creating such a dictionary)

	hm_epochs: how many epochs to train the net (default hm_epochs=10)

	save_path: if provided saves the model to a file outside the program,
	tf documentation: https://www.tensorflow.org/programmers_guide/saved_model#save_and_restore_models
	(default: save_path=""- network will not be saved)

