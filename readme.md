Python functions allowing for faster building and
training simple feed forward neural networks implemented
eith Tensorflow. 
Built with Python 3.6.3, Tensorflow 1.6.0.

Feel free to use the code however you want if you find it helpful.

Code written in response to the YT video:
https://www.youtube.com/watch?v=BhpvH5DuVu8&list=PLSPWNkAMSvv5DKeSVDbEbUKSsK4Z-GgiP&index=3

F.A.Szombara

DATA: 
pos.txt- file containing positive sentences
neg.txt- file containing negative sentences

1. neural-net-func.py

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

	* Parameters:

	features: feature vectors for training

	labels: lable vectors for training 

	neural_net: neural network weights and biases in a form od a dictionary
	(function construct_neural_net is design fot creating such a dictionary)

	hm_epochs: how many epochs to train the net (default hm_epochs=10)

	save_path: if provided saves the model to a file outside the program,
	tf documentation: https://www.tensorflow.org/programmers_guide/saved_model#save_and_restore_models
	(default: save_path=""- network will not be saved)

	batch_size: size of training batch (default: batch_size=100)

2. file: data_processing.py

create_lexticon(*files,num_of_lines=0,max_freq=0.005,min_freq=0.00001):

	Creates lexicon word stems found in given files
	
	* Inputs:
	files*: names/paths of files cotaining sentences for later sentiment analisis
	
	* Parameters:
	num_of_lines: number of lines to be read from each file,
	default set to 0 makes function read all the lines of each file

	max_freq:how frequentlya word can be used be added to the lexicon- upper bound

	min_freq:how frequently a word must be used be added to the lexicon- lower bound

	* Returns:
	a list of stems of words found in files


def create_lexticon_of_given_length(*files,num_of_lines=0,lex_len=550, max_freq=0.004):

	Creates lexicon of a given length containing word stems found in given files
	
	* Inputs:
	files*: names/paths of files cotaining sentences for later sentiment analisis
	
	* Parameters:
	num_of_lines: number of lines to be read from each file,
	default set to 0 makes function read all the lines of each file

	max_freq:how frequentlya word can be used be added to the lexicon- upper bound

	lex_len: desired length of lexicon. Important: length of lexicon will become
	the length of word vector in fuction sentences_vectorisation

	* Returns:
	a list of length lex_len containing stems of words found in files 


def sentences_vectorisation(file, lexicon, sentiment, num_of_lines=0):

	For each sentece in a file creates a tuple with sentence vector (multiple hot array created
	according to a given lexicon) as its zero element and sentiment lable vector
	([1,0]-positive, [0,1]-negative) as the second element

	* Inputs:
	file: name/path of  a file cotaining sentences for later sentiment analisis

	lexicon: lexicon conatining stemmed words occuring in file (go to :create_lexticon())

	sentiment: label vector of sentiment: [1,0]-positive, [0,1]-negative

	* Parameters:
	num_of_lines: number of lines to be read from each file,
	default set to 0 makes function read all the lines of the file

	* Returns:
	list o tuples where every tuple represents a sentence. Each tuple has a sentence vector 
	(multiple hot array created according to a given lexicon; vector is of length len(lexicon)) 
	as its zero element and sentiment lable vector([1,0]-positive, [0,1]-negative)


def string_vectorisation(string, lexicon):

	creates a string vector (multiple hot array created
	according to a given lexicon) 

	*Inputs:
	string: sentence to predict

	* Returns:
	a vector (in for of a list) representation of a string made
	according to given lexicon


def create_train_test_featuresets(pos, neg, test_size=0.1):

	Creates test and train feature, lables sets from files containing positive and negative sentences

	*Inputs
	pos: file name containing positive sentences

	neg: file name containing negative sentences

	*Parameters:
	test_size: size of the test set (default set to 0.1)

	Returns: train_features, train_labels, test_features, test_labels
	[1,0]- positive label
	[0,1]- negative label

3. pred.py

def get_pred_on_string(string, lexicon, model='models.meta'):

	predicts if a sentence is positive or negative

	*Inputs:
	string: sentence to predict

	*Parameters:
	lexicon: lexicon of words to create a sentence vector

	model: file containing tf model meta

	Returns:
	True if the sentence is positive,
	False if the sentence is positive

4. sentiment_neural_net.py- file executing sentiment analisis using
provided functions

