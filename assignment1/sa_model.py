#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A model for sentiment analysis.
"""
import pdb
import logging

import math
import tensorflow as tf
from util import Progbar, minibatches, word2index
from model import Model
from sklearn.metrics import confusion_matrix, f1_score

logger = logging.getLogger("hw1.q5")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class SAModel(Model):
    """
    Implements special functionality for SA models.
    """

    def __init__(self, config, report=None):
        self.config = config
        self.report = report

    def evaluate(self, sess, examples, examples_raw):
        """Evaluates model performance on @examples.

        This function uses the model to predict labels for @examples and constructs a confusion matrix.

        Args:
            sess: the current TensorFlow session.
            examples: A list of vectorized input/output pairs.
            examples: A list of the original input/output sequence pairs.
        Returns:
            The F1 score for predicting tokens as named entities (Token-level and Entity level).
        """
        y_true, preds = self.output(sess, examples_raw, examples)
        cm = confusion_matrix(y_true, preds)
        f1 = f1_score(y_true, preds, average='micro')
        return cm, f1

    def output(self, sess, inputs_raw, inputs=None):
        """
        Reports the output of the model on examples (uses helper to featurize each example).
        """
        if inputs is None:
            inputs = self.preprocess_sequence_data(word2index(self.tokens, inputs_raw)[1])

#         inputs = inputs[:self.config.batch_size] # just for debug
        preds = []
        prog = Progbar(target=math.ceil(len(inputs) / self.config.batch_size))
        y_true = []
        for i, batch in enumerate(minibatches(inputs, self.config.batch_size, shuffle=False)):
            # Ignore predict
            y_true.extend(batch[1])
            batch = batch[:1] + batch[2:]
            preds_ = self.predict_on_batch(sess, *batch)
            preds += list(preds_)
            prog.update(i + 1, [])
        return y_true, preds

    def fit(self, sess, saver, train_examples_raw, dev_set_raw):
        best_score = 0.

        train_examples = self.preprocess_sequence_data(train_examples_raw)
        print(train_examples[0])
        dev_set = self.preprocess_sequence_data(dev_set_raw)

        for epoch in range(self.config.n_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            # You may use the progress bar to monitor the training progress
            # Addition of progress bar will not be graded, but may help when debugging
            prog = Progbar(target=math.ceil(len(train_examples) / self.config.batch_size))
			
			# The general idea is to loop over minibatches from train_examples, and run train_on_batch inside the loop
			# Hint: train_examples could be a list containing the feature data and label data
			# Read the doc for utils.get_minibatches to find out how to use it.
                        # Note that get_minibatches could either return a list, or a list of list
                        # [features, labels]. This makes expanding tuples into arguments (* operator) handy

            ### YOUR CODE HERE (2-3 lines)
            for i, batch in enumerate(minibatches(train_examples, self.config.batch_size, shuffle=True)):
                # batch[0] is 
                loss = self.train_on_batch(sess, *batch)
                prog.update(i + 1, [("train loss", loss)])
                if self.report:
                    self.report.log_train_loss(loss)
            ## END YOUR CODE

            logger.info("Evaluating on development data")
            cm, score = self.evaluate(sess, dev_set, dev_set_raw)
            logger.debug("confusion matrix:\n" + str(cm))
            logger.info("f1 score: %.2f", score)
            
            if score > best_score:
                best_score = score
                if saver:
                    logger.info("New best score! Saving model in %s", self.config.model_output)
                    saver.save(sess, self.config.model_output)
            print("")
            if self.report:
                self.report.log_epoch()
                self.report.save()
        return best_score
