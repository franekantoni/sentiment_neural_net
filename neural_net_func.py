import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def construct_neural_net(layers_nodes, input_size, output_size):
	"""
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
	"""

	layers_nodes.append(output_size)

	Layers = {}
	num_of_layers = len(layers_nodes)
	for i, layer in enumerate(layers_nodes):
		if i == 0:
			Layers["hidden_l{}".format(i)] = {"weights": tf.Variable(tf.random_normal([input_size, layers_nodes[i]])),
			"biases": tf.Variable(tf.random_normal([layers_nodes[i]]))}
		else:
			Layers["hidden_l{}".format(i)] = {"weights": tf.Variable(tf.random_normal([layers_nodes[i-1], layers_nodes[i]])),
			"biases": tf.Variable(tf.random_normal([layers_nodes[i]]))}
	return Layers


def neural_network_model(x, neural_net):
	"""
	Defines the operations in the neural network and returns the output layer

	Parameters:

	x: tf placeholder for the data

	neural_net: neural network weights and biases in a form od a dictionary
	(function construct_neural_net is design fot creating such a dictionary)

	Reurns:
	output layer of neural net

	"""

	Layers ={}
	num_of_layers = len(neural_net)

	for i, layer in enumerate(neural_net):
		if i == 0:

			Layers["l{}".format(i)] = tf.add(tf.matmul(x, neural_net[layer]["weights"]), neural_net[layer]["biases"])

			Layers["l{}".format(i)] = tf.nn.relu(Layers["l{}".format(i)])
		elif i == num_of_layers -1: #if you hit the last layer
			return tf.add(tf.matmul(Layers["l{}".format(i-1)], neural_net[layer]["weights"]), neural_net[layer]["biases"]) #return the output of the last layer
		else:
			Layers["l{}".format(i)] = tf.add(tf.matmul(Layers["l{}".format(i-1)], neural_net[layer]["weights"]), neural_net[layer]["biases"])
			Layers["l{}".format(i)] = tf.nn.relu(Layers["l{}".format(i)])



def train_neural_network(train_x, train_y, test_x, test_y, neural_net, hm_epochs=10, save_path="",batch_size=100):
	"""
	Trains a neural network, prints the final accuracy score. Saves
	the network if save_path parameter is provided

	* Parameters:

	train_x: feature vectors for training

	train_y: lable vectors for training 

	test_x: feature vectors for testing

	test_y: lable vectors for testing 

	neural_net: neural network weights and biases in a form od a dictionary
	(function construct_neural_net is design fot creating such a dictionary)

	hm_epochs: how many epochs to train the net (default hm_epochs=10)

	save_path: if provided saves the model to a file outside the program,
	tf documentation: https://www.tensorflow.org/programmers_guide/saved_model#save_and_restore_models
	(default: save_path=""- network will not be saved)

	batch_size: size of training batch (default: batch_size=100)

	"""
	placeholder_x = tf.placeholder('float',[None, len(train_x[0])], name="input")
	placeholder_y = tf.placeholder('float')

	prediction = neural_network_model(placeholder_x, neural_net)
	prediction_op = neural_network_model(placeholder_x, neural_net)

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=placeholder_y,
    logits=prediction))
	#default learning rate = 0.001
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(1,hm_epochs+1):
			epoch_loss = 0
			
			i = 0
			while i < len(train_x):
				start = i
				end = start+batch_size

				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])

				_, c = sess.run([optimizer, cost], feed_dict = {placeholder_x: batch_x, placeholder_y: batch_y})
				epoch_loss += c

				i += batch_size

			print("Epoch", epoch, "completed; out of:", hm_epochs, "epoch_loss=", epoch_loss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(placeholder_y,1))

		#export prediction operation to be reused
		hack_pred= tf.equal(tf.argmax(prediction, 1), [0,1], name="pred_op")

		accuracy = tf.reduce_mean(tf.cast(correct, "float"))

		print("Accuracy:", accuracy.eval({placeholder_x: test_x, placeholder_y: test_y}))
		if save_path:
			saver = tf.train.Saver()
			saver.save(sess=sess, save_path=save_path)








