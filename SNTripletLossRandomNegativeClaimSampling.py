#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 00:41:52 2023

@author: francisco
"""

import tensorflow as tf
import json
from gensim.models import Word2Vec
import os
import numpy as np
from tqdm import tqdm
import random
from operator import itemgetter
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from inferences import evaluateEvidenceRetrieval,EvaluateCallback
import tensorflow_hub as hub
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from Models import siameseNN



def readJson(path):
    f=open(path)
    data=json.load(f)
    f.close()
    return data

datasetPaths="/home/francisco/Documents/NLPproject/project-data/"

train_claims=readJson(datasetPaths+'train-claims.json')
dev_claims=readJson(datasetPaths+'dev-claims.json')

evidence=readJson(datasetPaths+'evidence.json')


print("--------------Loading Google Universal Sentence Encoder")
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
print("--------------Done")

#---------------triplets creation (A,P,N)
print("-------------Generating train and dev triplets")
train_triplets=[]
dev_triplets=[]
evidenceIds = list(evidence.keys())
for claimId, claim in tqdm(train_claims.items()):
    for evidenceId in claim["evidences"]:
        
        #train_triplets.append((claim["claim_text"],embeddings_evidence[evidenceId], embeddings_evidence[np.random.choice(evidenceIds)]))
        anchor=embed([claim["claim_text"]])
        positive=embed([evidence[evidenceId]])
        
        #negative sample is a claim and not an evidence
        negative=embed([train_claims[  np.random.choice([idC for idC in train_claims.keys() if idC != claimId])]['claim_text']]) 
        
        train_triplets.append((anchor,positive,negative  ))

for claimId, claim in tqdm(dev_claims.items()):
    for evidenceId in claim["evidences"]:
        anchor=embed([claim["claim_text"]])
        positive=embed([evidence[evidenceId]])
        negative=embed([dev_claims[np.random.choice([idC for idC in dev_claims.keys() if idC != claimId])]['claim_text'] ]) 
        
        dev_triplets.append((anchor,positive,negative  ))
        
np.random.shuffle(train_triplets)
np.random.shuffle(dev_triplets)

print("-------------Done")



checkpoint=ModelCheckpoint('/home/francisco/Documents/NLPproject/checkpoint/model.{epoch:d}.h5',save_best_only=False, save_freq='epoch')
tensorboard_callback=TensorBoard('/home/francisco/Documents/NLPproject/logs',histogram_freq=1)



# def embeddingModel():
#         inputEmbedding=Input(name="sequenceEmbedding", shape=(512))
#         x=Dense(512, activation="relu")(inputEmbedding)
#         x=Dropout(0.2)(x)
#         x=Dense(256,activation="relu")(x)
#         x=Dropout(0.2)(x)
#         x=BatchNormalization()(x)
#         #model.add(Dropout(0.2))
#         outputEmbedding=Dense(128,activation="relu")(x)
#         embedding = tf.keras.Model(inputEmbedding, outputEmbedding, name="OutputEmbedding")
    
#         return embedding
def embeddingModel():
        inputEmbedding=Input(name="sequenceEmbedding", shape=(512))

        x=Dense(128,activation="relu")(inputEmbedding)
        x=Dropout(0.2)(x)
        x=BatchNormalization()(x)
        #model.add(Dropout(0.2))
        outputEmbedding=x
        embedding = tf.keras.Model(inputEmbedding, outputEmbedding, name="OutputEmbedding")
    
        return embedding
embeddingShape=128
embedding=embeddingModel()
siamese_model=siameseNN(embedding)


#siamese_model=SiameseModel(siamese_network)
siamese_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))



evaluate_callback = EvaluateCallback(embed,dev_claims,evidence,evidenceIds, siamese_model) 
epochs=30
siamese_model.built = True
#siamese_model.fit(train_triplets,epochs=epochs,batch_size=100,validation_data=dev_triplets,callbacks=[tensorboard_callback,checkpoint,evaluate_callback])
#tf.keras.utils.plot_model(siamese_model, to_file='/home/francisco/Documents/NLPprojvect/project-data/i.png', show_shapes=True)
#siamese_model.fit(train_triplets,epochs=epochs,batch_size=100,validation_data=dev_triplets)

#access embedding layer
#siamese_model.get_layer("functional_1").get_layer("OutputEmbedding")(dev_triplets[0])




siamese_model.load_weights('/home/francisco/Documents/NLPproject/checkpoint/model.7.h5')
embedding=siamese_model.get_layer("functional_1").get_layer("OutputEmbedding")

trainingClaimsTransformed=[]
for i in tqdm(train_claims.values()):
    trainingClaimsTransformed.append( embedding(embed([ i["claim_text"] ])).numpy() )
trainingClaimsTransformed=np.array(trainingClaimsTransformed).reshape((-1,embeddingShape))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

# Assuming you have a numpy array called 'claimsTransformed'
# with shape (samples, features)

# Creating an empty list to store the inertia values
inertia = []

# Specifying the range of k values to try for K-means clustering
k_values = range(1, 11)

# Performing K-means clustering for each value of k
#for k in k_values:
k=4
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(trainingClaimsTransformed)
#kmeans = GaussianMixture(n_components=4, random_state=0).fit(trainingClaimsTransformed)


y_supervised=[]
x_supervised=[]
evidenceOrClaim=[]
for claimId, claimInfo in train_claims.items():
    x=embedding(embed([ claimInfo["claim_text"] ])).numpy()
    y=kmeans.predict(x)[0]
    y_supervised.append(y)
    x_supervised.append(x.reshape(-1))
    evidenceOrClaim.append(1)
    for evidenceId in claimInfo["evidences"]:
        x=embedding(embed([evidence[evidenceId]])).numpy()
        y_supervised.append(y)
        x_supervised.append(x.reshape(-1))
        evidenceOrClaim.append(0)
        
y_supervised=np.array(y_supervised)
x_supervised=np.array(x_supervised)
evidenceOrClaim=np.array(evidenceOrClaim)
idx = np.random.permutation(len(y_supervised))
x_supervised,y_supervised,evidenceOrClaim = x_supervised[idx], y_supervised[idx],evidenceOrClaim[idx]

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as knn

#clf = SVC(kernel="poly",degree=3)#polinomial grade 3
clf = SVC(kernel="rbf")
clf.fit(x_supervised, y_supervised)



def evaluateTopicClf(dev_claims):
    
    totEvidences=0
    acumMatch=0
    y_true=[]
    y_pred=[]
    for claimId, claimInfo in dev_claims.items():
        x=embedding(embed([ claimInfo["claim_text"] ])).numpy()
        y_claim=clf.predict(x)[0]
        
        for evidenceId in claimInfo["evidences"]:
            x=embedding(embed([evidence[evidenceId]])).numpy()
            
            y_evidence=clf.predict(x)[0]
            y_true.append(y_claim)
            y_pred.append(y_evidence)
            if y_evidence==y_claim:
                acumMatch+=1
            totEvidences+=1
    ratio=acumMatch/totEvidences
    print("ratio of evidences in the same topic as the claim: ",ratio)
    return y_true,y_pred,ratio
evaluateTopicClf(dev_claims)



# claimInfo=dev_claims["claim-104"]
# x=embedding(embed([ claimInfo["claim_text"] ])).numpy()
# clf.predict(x)[0]

# x=embedding(embed([evidence[claimInfo["evidences"][0]]])).numpy() 
# clf.predict(x)[0]



# x=embedding(embed([txt])).numpy() 
# clf.predict(x)[0]




#inertia.append(kmeans.inertia_)

# Plotting the elbow curve
# plt.plot(k_values, inertia, 'bx-')
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Inertia')
# plt.title('The Elbow Method')
# plt.show()




# from tensorflow.keras.layers import LSTM,concatenate
# maxLen=5
# embeddingSize=512

# def classifier():
#     claim_input = tf.keras.Input(
#         shape=(embeddingSize), name="claimEmbedding"
#     ) 
    
#     SeqEvidences_input = tf.keras.Input(
#         shape=(maxLen,embeddingSize), name="SequenceEvidences"
#     ) 
    

#     evidences_features =Dropout(0.2)( LSTM(128)(SeqEvidences_input))
    
    
    
#     x=concatenate([evidences_features, claim_input])#128+512
#     x=Dense(256,activation='relu')(x)
#     x=Dropout(0.2)(Dense(128,activation='relu')(x))
#     x=Dense(32,activation='relu')(x)
#     pred = Dense(4,activation='softmax',name="output")(x)
    
#     model = tf.keras.Model(
#         inputs=[claim_input, SeqEvidences_input ],
#         outputs=pred,
#     )

#     model.compile(loss=['categorical_crossentropy'], optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),tf.keras.metrics.AUC(name='auc'),tf.keras.metrics.AUC(name='prc', curve='PR')])
  

#     return model

# def generateDatasetClf(embeddings_train_claims,EvidencesIds):
#     labels=[]
#     claim_input=[]
    
#     for claimId, claimInfo in embeddings_train_claims.items():
#         labels.append(claimInfo["claim_label"])
#         claim_input.append(claimInfo["claim_text"].numpy())
#         top100=retrieveClosestEvidenceCosine(claimInfo,EvidencesIds,100)
        
#         evidenceList=retrieveClosestEvidencesSiamese(claimInfo,top100)
    
#         sequenceEmbedingsEvidence=[ embeddings_evidence[IdE] for IdE in evidenceList]
    