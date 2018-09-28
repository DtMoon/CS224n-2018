#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q2: Recurrent neural nets for SA
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import os
import sys
import time
from datetime import datetime

import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import itertools

from sa_model import SAModel

from util import word2index
from utils.treebank import StanfordSentiment
import utils.glove as glove

logger = logging.getLogger("hw1.q5")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    n_word_features = 2 # Number of features for every word in the input.
    window_size = 1
    n_features = (2 * window_size + 1) * n_word_features # Number of features for every word in the input.
    max_length = 120 # longest sequence to parse
    n_classes = 5
    dropout = 0.5
    embed_size = 50
    hidden_size = 300
    batch_size = 32
    n_epochs = 10
    max_grad_norm = 10.
    lr = 0.001

    def __init__(self, args):
        self.cell = args.cell

        if "model_path" in args:
            # Where to save things.
            self.output_path = args.model_path
        else:
            self.output_path = "results/{}/{:%Y%m%d_%H%M%S}/".format(self.cell, datetime.now())
            os.makedirs(self.output_path, exist_ok=True)
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.conll_output = self.output_path + "{}_predictions.conll".format(self.cell)
        self.log_output = self.output_path + "log"
        
def pad_sequences(data, max_length):
    """Ensures each input-output seqeunce pair in @data is of length
    @max_length by padding it with zeros and truncating the rest of the
    sequence.

    TODO: In the code below, for every sentence, labels pair in @data,
    (a) create a new sentence which appends zero feature vectors until
    the sentence is of length @max_length. If the sentence is longer
    than @max_length, simply truncate the sentence to be @max_length
    long.
    (b) create a new label sequence similarly.
    (c) create a _masking_ sequence that has a True wherever there was a
    token in the original sequence, and a False for every padded input.

    Example: for the (sentence, labels) pair: [[4,1], [6,0], [7,0]], [1,
    0, 0], and max_length = 5, we would construct
        - a new sentence: [[4,1], [6,0], [7,0], [0,0], [0,0]]
        - a new label seqeunce: [1, 0, 0, 4, 4], and
        - a masking seqeunce: [True, True, True, False, False].

    Args:
        data: is a list of (sentence, labels) tuples. @sentence is a list
            containing the words in the sentence and @label is a list of
            output labels. Each word is itself a list of
            @n_features features. For example, the sentence "Chris
            Manning is amazing" and labels "PER PER O O" would become
            ([[1,9], [2,9], [3,8], [4,8]], [1, 1, 4, 4]). Here "Chris"
            the word has been featurized as "[1, 9]", and "[1, 1, 4, 4]"
            is the list of labels. 
        max_length: the desired length for all input/output sequences.
    Returns:
        a new list of data points of the structure (sentence', labels', mask).
        Each of sentence', labels' and mask are of length @max_length.
        See the example above for more details.
    """
    ret = []

    # Use this zero vector when padding sequences.
    zero_vector = 0

    for sentence, label in data:
        ### YOUR CODE HERE (~4-6 lines)
        if len(sentence) < max_length:
            new_sentence = sentence.copy() + [zero_vector] * (max_length - len(sentence))
            mask = [False] * (len(sentence) - 1) + [True] + [False] * (max_length - len(sentence))
            ret.append((new_sentence, label, mask))
        else:
            ret.append((sentence[:max_length].copy(), label, [False] * (max_length - 1) + [True]))
        ### END YOUR CODE ###
    return ret

class RNNModel(SAModel):
    """
    Implements a recursive neural network with an embedding layer and
    single hidden layer.
    This network will predict a sequence of labels (e.g. PER) for a
    given token (e.g. Henry) using a featurized window around the token.
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of  shape (None, self.max_length, n_features), type tf.int32
        labels_placeholder: Labels placeholder tensor of shape (None, self.max_length), type tf.int32
        mask_placeholder:  Mask placeholder tensor of shape (None, self.max_length), type tf.bool
        dropout_placeholder: Dropout value placeholder (scalar), type tf.float32

        TODO: Add these placeholders to self as the instance variables
            self.input_placeholder
            self.labels_placeholder
            self.mask_placeholder
            self.dropout_placeholder

        HINTS:
            - Remember to use self.max_length NOT Config.max_length

        (Don't change the variable names)
        """
        ### YOUR CODE HERE (~4-6 lines)
        self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_length))
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None,))
        self.mask_placeholder = tf.placeholder(tf.bool, shape=(None, self.config.max_length))
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=())
        ### END YOUR CODE

    def create_feed_dict(self, inputs_batch, mask_batch, labels_batch=None, dropout=1):
        """Creates the feed_dict for the dependency parser.

        A feed_dict takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.
        Hint: When an argument is None, don't add it to the feed_dict.

        Args:
            inputs_batch: A batch of input data.
            mask_batch:   A batch of mask data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        ### YOUR CODE (~6-10 lines)
        feed_dict = {}
        feed_dict[self.input_placeholder] = inputs_batch
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        feed_dict[self.mask_placeholder] = mask_batch
        feed_dict[self.dropout_placeholder] = dropout
        ### END YOUR CODE
        return feed_dict

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:

        TODO:
            - Create an embedding tensor and initialize it with self.pretrained_embeddings.
            - Use the input_placeholder to index into the embeddings tensor, resulting in a
              tensor of shape (None, max_length, n_features, embed_size).
            - Concatenates the embeddings by reshaping the embeddings tensor to shape
              (None, max_length, n_features * embed_size).

        HINTS:
            - You might find tf.nn.embedding_lookup useful.
            - You can use tf.reshape to concatenate the vectors. See
              following link to understand what -1 in a shape means.
              https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#reshape.

        Returns:
            embeddings: tf.Tensor of shape (None, max_length, embed_size)
        """
        ### YOUR CODE HERE (~4-6 lines)
        embed_matrix = tf.get_variable('embed_matrix', shape=self.pretrained_embeddings.shape,
                                    initializer=tf.constant_initializer(self.pretrained_embeddings))
        embed = tf.nn.embedding_lookup(embed_matrix, self.input_placeholder)
        embeddings = tf.reshape(embed, (-1, self.config.max_length, self.config.embed_size))
        ### END YOUR CODE
        return embeddings

    def add_prediction_op(self):
        """Adds the unrolled RNN:
            h_0 = 0
            for t in 1 to T:
                o_t, h_t = cell(x_t, h_{t-1})
                o_drop_t = Dropout(o_t, dropout_rate)
                y_t = o_drop_t U + b_2

        TODO: There a quite a few things you'll need to do in this function:
            - Define the variables U, b_2.
            - Define the vector h as a constant and inititalize it with
              zeros. See tf.zeros and tf.shape for information on how
              to initialize this variable to be of the right shape.
              https://www.tensorflow.org/api_docs/python/tf/zeros
              https://www.tensorflow.org/api_docs/python/tf/shape
            - In a for loop, begin to unroll the RNN sequence. Collect
              the predictions in a list.
            - When unrolling the loop, from the second iteration
              onwards, you will HAVE to call
              tf.get_variable_scope().reuse_variables() so that you do
              not create new variables in the RNN cell.
              See https://www.tensorflow.org/api_guides/python/state_ops#Sharing_Variables
            - Concatenate and reshape the predictions into a predictions
              tensor.
        Hint: You will find the function tf.stack (similar to np.asarray)
              useful to assemble a list of tensors into a larger tensor.
              https://www.tensorflow.org/api_docs/python/tf/stack
        Hint: You will find the function tf.transpose and the perms
              argument useful to shuffle the indices of the tensor.
              https://www.tensorflow.org/api_docs/python/tf/transpose

        Remember:
            * Use the xavier initilization for matrices.
            * Note that tf.nn.dropout takes the keep probability (1 - p_drop) as an argument.
            The keep probability should be set to the value of self.dropout_placeholder

        Returns:
            pred: tf.Tensor of shape (batch_size, max_length, n_classes)
        """

        x = self.add_embedding()
        dropout_rate = self.dropout_placeholder

        preds = [] # Predicted output at each timestep should go here!

        # Use the cell defined below. For Q2, we will just be using the
        # RNNCell you defined, but for Q3, we will run this code again
        # with a GRU cell!
        if self.config.cell == "rnn":
            cell = RNNCell(Config.n_features * Config.embed_size, Config.hidden_size)
        elif self.config.cell == "gru":
            cell = GRUCell(Config.n_features * Config.embed_size, Config.hidden_size)
        elif self.config.cell == "lstm":
            cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.config.hidden_size)
        else:
            raise ValueError("Unsuppported cell type: " + self.config.cell)

        # Define U and b2 as variables.
        # Initialize state as vector of zeros.
        ### YOUR CODE HERE (~4-6 lines)
        U = tf.get_variable('U', shape=(self.config.hidden_size, self.config.n_classes), initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable('b2', shape=(self.config.n_classes,), initializer=tf.constant_initializer(0))
#         h = tf.zeros((tf.shape(x)[0], self.config.hidden_size)) # lstm cell will raise an error if use this initialization
        h = cell.zero_state(tf.shape(x)[0], tf.float32)
        ### END YOUR CODE

        with tf.variable_scope("RNN"):
            for time_step in range(self.max_length):
                ### YOUR CODE HERE (~6-10 lines)
                o_t, h = cell(x[:, time_step, :], h)
                o_drop_t = tf.nn.dropout(o_t, dropout_rate)
                y_t = tf.matmul(o_drop_t, U) + b2
                preds.append(y_t)
                if time_step == 0:
                    tf.get_variable_scope().reuse_variables()
                ### END YOUR CODE

        # Make sure to reshape @preds here.
        ### YOUR CODE HERE (~2-4 lines)
        preds = tf.stack(preds)
        preds = tf.transpose(preds, [1, 0, 2])
        ### END YOUR CODE

        assert preds.get_shape().as_list() == [None, self.max_length, self.config.n_classes], "predictions are not of the right shape. Expected {}, got {}".format([None, self.max_length, self.config.n_classes], preds.get_shape().as_list())
        preds = tf.boolean_mask(preds, self.mask_placeholder)
        return preds

    def add_loss_op(self, preds):
        """Adds Ops for the loss function to the computational graph.

        TODO: Compute averaged cross entropy loss for the predictions.
        Importantly, you must ignore the loss for any masked tokens.

        Hint: You might find tf.boolean_mask useful to mask the losses on masked tokens.
        Hint: You can use tf.nn.sparse_softmax_cross_entropy_with_logits to simplify your
                    implementation. You might find tf.reduce_mean useful.
        Args:
            pred: A tensor of shape (batch_size, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        ### YOUR CODE HERE (~2-4 lines)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_placeholder, 
                                                                             logits=preds))
        ### END YOUR CODE
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Use tf.train.AdamOptimizer for this model.
        Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        ### YOUR CODE HERE (~1-2 lines)
        train_op = tf.train.AdamOptimizer().minimize(loss)
        ### END YOUR CODE
        return train_op

    def preprocess_sequence_data(self, examples):
        return pad_sequences(examples, self.max_length)

    def predict_on_batch(self, sess, inputs_batch, mask_batch):
        feed = self.create_feed_dict(inputs_batch=inputs_batch, mask_batch=mask_batch)
        predictions = sess.run(tf.argmax(self.pred, axis=1), feed_dict=feed)
        return predictions

    def train_on_batch(self, sess, inputs_batch, labels_batch, mask_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch, mask_batch=mask_batch,
                                     dropout=Config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def __init__(self, config, pretrained_embeddings, tokens, report=None):
        super(RNNModel, self).__init__(config, report)
        self.max_length = self.config.max_length
        self.pretrained_embeddings = pretrained_embeddings
        self.tokens = tokens

        # Defining placeholders.
        self.input_placeholder = None
        self.labels_placeholder = None
        self.mask_placeholder = None
        self.dropout_placeholder = None

        self.build()

def outputConfusionMatrix(pred, labels, filename):
    """ Generate a confusion matrix """
    cm = confusion_matrix(labels, pred, labels=range(5))
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.colorbar()
    classes = ["- -", "-", "neut", "+", "+ +"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename)
    
def do_train(args):
    # Set up some parameters.
    config = Config(args)
    
    # Load the dataset
    dataset = StanfordSentiment()
    tokens = dataset.tokens()
    nWords = len(tokens)

    if args.vector == "yourvectors":
        _, wordVectors, _ = load_saved_params()
        wordVectors = np.concatenate(
            (wordVectors[:nWords,:], wordVectors[nWords:,:]),
            axis=1)
    elif args.vector == "pretrained":
        wordVectors = glove.loadWordVectors(tokens)

    # Load the train set
    trainset = dataset.getTrainSentences()
    train_max_length, train, train_raw = word2index(tokens, trainset)
    print(train_raw[0])
    print(train[0])

    # Prepare dev set features
    devset = dataset.getDevSentences()
    _, dev, dev_raw = word2index(tokens, devset)

    # Prepare test set features
    testset = dataset.getTestSentences()
    _, test, test_raw = word2index(tokens, testset)
        
    config.max_length = train_max_length
    config.embed_size = wordVectors.shape[1]

    handler = logging.FileHandler(config.log_output)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    report = None #Report(Config.eval_output)

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = RNNModel(config, wordVectors, tokens)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            model.fit(session, saver, train, dev)

            # do some error analysis
            if args.vector == "pretrained":
                y_true, preds = model.output(session, dev_raw)
                outputConfusionMatrix(preds, y_true, "q5_dev_conf.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains and tests an NER model')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('train', help='')
    command_parser.add_argument('-v', '--vector', choices=["yourvectors", "pretrained"], default="pretrained", help="word vector to use")
    command_parser.add_argument('-c', '--cell', choices=["rnn", "gru", "lstm"], default="lstm", help="Type of RNN cell to use.")
    command_parser.set_defaults(func=do_train)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
