import neural_net_func as nnf
import sentiment
import pickle

with open("train_and_test.pickle", 'rb') as file:
	train_features, train_labels, test_features, test_labels = pickle.load(file)

layers_nodes, input_size, output_size = ([550,550,550,550,500,300,300], len(train_features[0]), 2)
neural_net = nnf.construct_neural_net(layers_nodes, input_size, output_size)

nnf.train_neural_network(train_features, train_labels, neural_net, save_path="./models")