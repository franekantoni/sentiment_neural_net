import neural_net_func as nnf
import data_prosessing as pd
import tensorflow as tf
import numpy as np 

def get_pred_on_string(string, lexicon, model='models.meta'):
	"""
	predicts if a sentence is positive or negative

	*Inputs:
	string: sentence to predict

	*Parameters:
	lexicon: lexicon of words to create a sentence vector

	model: file containing tf model meta

	Returns:
	True if the sentence is positive,
	False if the sentence is positive
	"""
	with tf.Session() as sess:
		vector = dp.string_vectorisation(string, lexicon)
		new_saver = tf.train.import_meta_graph(model)
		new_saver.restore(sess, tf.train.latest_checkpoint('./'))
		graph = tf.get_default_graph()

		placeholder_x = graph.get_tensor_by_name("input:0")
		pred_op = graph.get_tensor_by_name("pred_op:0")

		feed_dict ={placeholder_x:np.array(vector).reshape(1,550)}

		output = sess.run(pred_op,feed_dict)
		if output[0] == True:
			return True
		else:
			return False

if __name__ == "__main__":
	lexicon = dp.create_lexticon_of_given_length("pos.txt", "neg.txt")
	print(get_pred_on_string(input("Give me a sentence to analize:\n"), lexicon))