import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np 
import random
import pickle
from collections import Counter
from nltk import PorterStemmer


def create_lexticon(*files,num_of_lines=0,max_freq=0.005,min_freq=0.00001):
	"""
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
	"""
	lexicon = []
	for file in (files):
		with open(file, "r") as file:
			if not num_of_lines:
				to_read = file
			else:
				to_read = file.readlines()[:num_of_lines]
			for line in to_read:
				words = word_tokenize(line.lower())
				lexicon += words

	#lemmatize (stem) each word in the lexicon list:
	lexicon = [PorterStemmer().stem(word) for word in lexicon]
	
	#create dictionary with words as keys and number of occurrences as values:
	word_count = Counter(lexicon)


	#eliminate the most frequently and occasionally used words:
	total_words_num = len(word_count)
	final_lexicon = []
	for word in word_count:
		#if word use frequency falles into given range, add it to the final lexicon
		if word_count[word]/total_words_num <= max_freq and word_count[word]/total_words_num >= min_freq:
			final_lexicon.append(word)

	return final_lexicon



def create_lexticon_of_given_length(*files,num_of_lines=0,lex_len=550, max_freq=0.004):
	"""
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
	"""
	lexicon = []
	for file in (files):
		with open(file, "r") as file:
			if not num_of_lines:
				to_read = file
			else:
				to_read = file.readlines()[:num_of_lines]
			for line in to_read:
				words = word_tokenize(line.lower())
				lexicon += words

	#lemmatize (stem) each word in the lexicon list:
	lexicon = [WordNetLemmatizer().lemmatize(word) for word in lexicon]
	
	#create dictionary with words as keys and number of occurrences as values:
	word_count = Counter(lexicon)


	#eliminate the most frequently and occasionally used words:
	total_words_num = len(word_count)
	word_score = []
	for word in word_count:
		if word_count[word]/total_words_num <= max_freq:
			word_score.append((word_count[word], word))

	word_score = sorted(word_score, reverse=True)

	final_lexicon = []
	for word in word_score[:lex_len]:
		final_lexicon.append(word[1])
		

	return final_lexicon




def sentences_vectorisation(file, lexicon, sentiment, num_of_lines=0):
	"""
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

	"""

	list_of_vectors = []
	with open(file, "r") as file:

		if not num_of_lines:
			to_read = file
		else:
			to_read = file.readlines()[:num_of_lines]

		for line in to_read:
			words = word_tokenize(line.lower())
			words = [WordNetLemmatizer().lemmatize(word) for word in words]
			vector = np.zeros(len(lexicon))

			for word in words:
				if word.lower() in lexicon:
					word_index = lexicon.index(word.lower())
					vector[word_index] += 1

			list_of_vectors.append((vector,sentiment))

		return list_of_vectors


def string_vectorisation(string, lexicon):
	"""
	creates a string vector (multiple hot array created
	according to a given lexicon) 

	*Inputs:
	string: sentence to predict

	* Returns:
	a vector (in for of a list) representation of a string made
	according to given lexicon

	"""
	words = word_tokenize(string.lower())
	words = [WordNetLemmatizer().lemmatize(word) for word in words]
	vector = np.zeros(len(lexicon))

	for word in words:
		if word.lower() in lexicon:
			word_index = lexicon.index(word.lower())
			vector[word_index] += 1
	return vector


def create_train_test_featuresets(pos, neg, test_size=0.1):
	"""
	Creates test and train feature, lables sets from files containing positive and negative sentences

	*Inputs
	pos: file name containing positive sentences

	neg: file name containing negative sentences

	*Parameters:
	test_size: size of the test set (default set to 0.1)

	Returns: train_features, train_labels, test_features, test_labels
	[1,0]- positive label
	[0,1]- negative label

	"""
	lexicon = create_lexticon_of_given_length(pos, neg)
	features_labels = []
	features_labels += sentences_vectorisation(pos, lexicon, [1,0])
	features_labels += sentences_vectorisation(neg, lexicon, [0,1])

	random.shuffle(features_labels)

	features_labels = np.array(features_labels)

	test_size = int(len(features_labels)*test_size)

	train_features = list(features_labels[:,0][:-test_size])
	train_labels = list(features_labels[:,1][:-test_size])

	test_features = list(features_labels[:,0][-test_size:])
	test_labels = list(features_labels[:,1][-test_size:])

	return train_features, train_labels, test_features, test_labels





if __name__ == '__main__':
	train_features,train_labels,test_features,test_labels = create_train_test_featuresets("pos.txt","neg.txt", test_size=0.1)
	with open("train_and_test.pickle", "wb") as file:
		pickle.dump((train_features,train_labels,test_features,test_labels), file)


