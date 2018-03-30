import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#EXAMPLE PARAMETERS:

#Number of Nodes of a Layer
nodes_num_l1 = 500
nodes_num_l2 = 500
nodes_num_l3 = 500

layers_nodes = [nodes_num_l1, nodes_num_l2, nodes_num_l3]

num_of_classes = 10 #10 digits
batch_size = 100

squashed_size = 784 #image of size 28*28 squashed into a flat vector

x = tf.placeholder('float', [None, squashed_size])
y = tf.placeholder('float')

input_size = squashed_size

output_size = num_of_classes


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



def train_neural_network(x, y, neural_net, hm_epochs=10, save_path=""):
	"""
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

	"""

	prediction = neural_network_model(x, neural_net)

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,
    logits=prediction))
	#default learning rate = 0.001
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(1,hm_epochs+1):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
				epoch_loss += c
			print("Epoch:", epoch, "completed out of:", hm_epochs, "loss:", epoch_loss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))

		accuracy = tf.reduce_mean(tf.cast(correct, "float"))

		print("Accuracy:", accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
		if save_path:
			saver = tf.train.Saver()
			saver.save(sess=sess, save_path=save_path)

#try:
#neural_net = construct_neural_net(layers_nodes, input_size, output_size)
#train_neural_network(x, y, neural_net, save_path="models/model1")







