#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 20:35:51 2023

@author: francisco
"""
import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input,concatenate,GRU


def embeddingModel():
        inputEmbedding=Input(name="sequenceEmbedding", shape=(512))
        x=Dense(512, activation="relu")(inputEmbedding)
        x=Dropout(0.2)(x)
        x=Dense(256,activation="relu")(x)
        x=BatchNormalization()(x)
        #model.add(Dropout(0.2))
        outputEmbedding=Dense(128,activation="relu")(x)
        embedding = tf.keras.Model(inputEmbedding, outputEmbedding, name="OutputEmbedding")
    
        return embedding
    
class DistanceLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        
        def call(self, anchor, positive, negative):
            ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
            an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
            return (ap_distance, an_distance)
    
class PredictDistanceLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        
        def call(self, anchor, positive):
            ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
            
            return ap_distance
    
class SiameseModel(tf.keras.Model):
        """The Siamese Network model with a custom training and testing loops.
    
        Computes the triplet loss using the three embeddings produced by the
        Siamese Network.
    
        The triplet loss is defined as:
           L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
        """
    
        def __init__(self, siamese_network, margin=0.5):
            super().__init__()
            self.siamese_network = siamese_network
            self.margin = margin
            self.loss_tracker = tf.keras.metrics.Mean(name="loss")
            
    
        def call(self, inputs):
            return self.siamese_network(inputs)
    
        def train_step(self, data):
            # GradientTape is a context manager that records every operation that
            # you do inside. We are using it here to compute the loss so we can get
            # the gradients and apply them using the optimizer specified in
            # `compile()`.
            with tf.GradientTape() as tape:
                loss = self._compute_loss(data)
    
            # Storing the gradients of the loss function with respect to the
            # weights/parameters.
            gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
    
            # Applying the gradients on the model using the specified optimizer
            self.optimizer.apply_gradients(
                zip(gradients, self.siamese_network.trainable_weights)
            )
    
            # Let's update and return the training loss metric.
            self.loss_tracker.update_state(loss)
            
            
            return {"loss": self.loss_tracker.result()}
    
        def test_step(self, data):
            loss = self._compute_loss(data)
    
            # Let's update and return the loss metric.
            self.loss_tracker.update_state(loss)
            
            return {"loss": self.loss_tracker.result()}
    
        def _compute_loss(self, data):
            # The output of the network is a tuple containing the distances
            # between the anchor and the positive example, and the anchor and
            # the negative example.
            ap_distance, an_distance = self.siamese_network(data)
    
            # Computing the Triplet Loss by subtracting both distances and
            # making sure we don't get a negative value.
            loss = ap_distance - an_distance
            loss = tf.maximum(loss + self.margin, 0.0)
            return loss
    
        @property
        def metrics(self):
            # We need to list our metrics here so the `reset_states()` can be
            # called automatically.
            return [self.loss_tracker]



def siameseNN(embedding):#embedding model
    

    #embedding=embeddingModel()
    shape=embedding.input_shape[1]
    anchor_input = Input(name="anchor", shape=(shape))
    positive_input = Input(name="positive", shape=(shape))
    negative_input = Input(name="negative", shape=(shape))

    distances = DistanceLayer()(
    embedding(anchor_input),
    embedding(positive_input),
    embedding(negative_input),
    )
    siamese_network = tf.keras.Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=distances
    
    )
    
    return SiameseModel(siamese_network)

def siameseNNInference(embedding):#embedding model
    

    #embedding=embeddingModel()
    shape=embedding.input_shape[1]
    anchor_input = Input(name="anchor", shape=(shape))
    positive_input = Input(name="positive", shape=(shape))
    

    distances = PredictDistanceLayer()(
    embedding(anchor_input),
    embedding(positive_input)
    )
    siamese_network = tf.keras.Model(
    inputs=[anchor_input, positive_input], outputs=distances
    
    )
    
    return SiameseModel(siamese_network)
    


def classifier():
    claim_input = tf.keras.Input(
        shape=(512), name="claim_embedding"
    ) 
    
    hipothesis_input = tf.keras.Input(
        shape=(5,512), name="hipothesis"
    ) 

    hipothesis_features =Dropout(0.2)( GRU(128)(hipothesis_input))
    

    x=concatenate([claim_input,hipothesis_features])
    
    x=Dense(128,activation='relu')(x)
    pred = Dense(4,activation='softmax',name="output")(x)
    
    model = tf.keras.Model(
        inputs=[claim_input, hipothesis_input],
        outputs=pred,
    )
    
    model.compile(loss=['categorical_crossentropy'], optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')])
  

    return model

    
    
