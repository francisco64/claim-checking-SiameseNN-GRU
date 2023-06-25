#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 21:16:22 2023

@author: francisco
"""


from scipy.spatial import distance
from operator import itemgetter
import numpy as np
from tqdm import tqdm
import random
import tensorflow as tf
def retrieveClosestEvidenceCosine(embed,evidence,claimInfo,EvidencesIds,k):#claim info: dict(claim text:txt, evidences:list of Ids),k: how many to retrieve

    claimEmbedding=embed(claimInfo["claim_text"])

    distanceClaim={}

    embeddings_evidence_test={idE : embed([evidence[idE]]) for idE in claimInfo["evidences"]}
    
    embeddings_evidence_test.update({idE : embed([evidence[idE]]) for idE in EvidencesIds})

    
    for evidenceId,embeddingEvidence in embeddings_evidence_test.items():
        #print("---------------distance: ",siamese_model((claimEmbedding,embeddingEvidence,embeddingEvidence))[0])
        distanceClaim[evidenceId]=distance.cosine(claimEmbedding.numpy().reshape(-1),embeddingEvidence.numpy().reshape(-1))
        
        
   
        
    filtered=list(dict(sorted(distanceClaim.items(), key = itemgetter(1))[:k]).keys())
        
    return filtered

def retrieveRelevantEvidences(embed,evidence,embeddingVector,embeddings_evidence_test,k=100000,measure="cosine"):#claim info: dict(claim text:txt, evidences:list of Ids),k: how many to retrieve

    
    distanceClaim={}

    

    
    for evidenceId,embeddingEvidence in embeddings_evidence_test.items():
        #print("---------------distance: ",siamese_model((claimEmbedding,embeddingEvidence,embeddingEvidence))[0])
        
        if measure=="cosine":
            distanceClaim[evidenceId]=distance.cosine(embeddingVector.numpy().reshape(-1),embeddingEvidence.numpy().reshape(-1))
        else:
            distanceClaim[evidenceId]=distance.euclidean(embeddingVector.numpy().reshape(-1),embeddingEvidence.numpy().reshape(-1))
        
    filtered=list(dict(sorted(distanceClaim.items(), key = itemgetter(1))[:k]).keys())
    return filtered


def retrieveClosestEvidencesSiamese(embed,evidence,claimInfo,embeddings_evidence_test,siamese_model,k=None,thres=0.12):#claim info: dict(claim text:embedding, evidences:list of Ids)

    claimEmbedding=embed([claimInfo["claim_text"]])

    distanceClaim={}


    for evidenceId,embeddingEvidence in embeddings_evidence_test.items():
        #print("---------------distance: ",siamese_model((claimEmbedding,embeddingEvidence,embeddingEvidence))[0])
        distanceClaim[evidenceId]=siamese_model((claimEmbedding,embeddingEvidence,embeddingEvidence))[0].numpy()[0]
    if k is None:   
        #gaussian=lambda x: np.exp(-np.power(x - np.mean(list(distanceClaim.values())) , 2.) / (2 * np.power(np.std(list(distanceClaim.values())), 2.)))
            
        #filtered=[k for k,v in distanceClaim.items()  if gaussian(v)<thresProb and v<np.mean(list(distanceClaim.values()))]
        filtered=[k for k,v in distanceClaim.items()  if v<thres]
        
        if len(filtered)==0:
            filtered=list(dict(sorted(distanceClaim.items(), key = itemgetter(1))[:1]).keys())
    else:
        filtered=list(dict(sorted(distanceClaim.items(), key = itemgetter(1))[:k]).keys())

        
    return filtered
    



def evaluateEvidenceRetrieval(embed,claims,evidence,siamese_model,possibleEvidenceIds,k=100):#k: evidence ids selected randomly to test retrieval, if None selects all possibleEvidenceIds
    AR=0
    AP=0
    avgK=0
    if k is None:
        embeddings_evidence_test={idE : embed([evidence[idE]]) for idE in possibleEvidenceIds}
    
    for claimId, claimInfo in tqdm(claims.items()):
        if k is not None:
            testEvidencesIds=random.sample(possibleEvidenceIds,k)
            embeddings_evidence_test={idE : embed([evidence[idE]]) for idE in testEvidencesIds}  
        else:
            testEvidencesIds=possibleEvidenceIds
            
          
        embeddings_evidence_test.update({idE : embed([evidence[idE]]) for idE in claimInfo["evidences"]})
    
        
        filtered = retrieveClosestEvidencesSiamese(embed,evidence,claimInfo,embeddings_evidence_test,siamese_model,k=None)#(claimInfo,testEvidencesIds,siamese_model)
        #filtered = retrieveClosestEvidencesSiamese(claimInfo,testEvidencesIds,siamese_model,100)
        #filtered =retrieveClosestEvidenceCosine(claimInfo,testEvidencesIds,100)
        avgK+=len(filtered)
        
        recallN=lambda filtered : np.sum([ev in filtered for ev in claimInfo["evidences"] ]) / (len(claimInfo["evidences"]) + 1e-15 )
        
        PrecisionN=lambda filtered : np.sum([ev in filtered for ev in claimInfo["evidences"] ]) / (len(filtered) + 1e-15)
        
        #ATh+=np.mean(list(distanceClaim.values())) - np.std(list(distanceClaim.values()))*np.sqrt(2*np.log(1/thresProb))
        
        
        AR+=recallN(filtered)
        
        AP+=PrecisionN(filtered)
    avgK=avgK/(len(claims)  + 1e-15)
    AR=AR/(len(claims)  + 1e-15)
    AP=AP/(len(claims)  + 1e-15)
    #ATh=ATh/len(embeddings_claims)
    f1=1/((1/AR)+(1/AP))
    print("AR: ",AR,"AP: ",AP,"F1-score: ",f1,"len(retrieval): ",avgK)
    return AR,AP,f1
    #print("top5AR: ",ARecall5,"top10AR: ",ARecall10,"topNRecall: ",ARecallN)
    
def getOptimalThProb(embed,claims,evidence,siamese_model,possibleEvidenceIds,k=100):#k: evidence ids selected randomly to test retrieval, if None selects all possibleEvidenceIds
    listAR=[]
    listAP=[]
    listf1=[]    
    
    for thProb in tqdm(np.arange(0.05,0.2,2/100)):

        AR=0
        AP=0
        avgK=0
        if k is None:
            embeddings_evidence_test={idE : embed([evidence[idE]]) for idE in possibleEvidenceIds}
        
        for claimId, claimInfo in tqdm(claims.items()):
            if k is not None:
                testEvidencesIds=random.sample(possibleEvidenceIds,k)
                embeddings_evidence_test={idE : embed([evidence[idE]]) for idE in testEvidencesIds}  
            else:
                testEvidencesIds=possibleEvidenceIds
                
              
            embeddings_evidence_test.update({idE : embed([evidence[idE]]) for idE in claimInfo["evidences"]})
        
            
            filtered = retrieveClosestEvidencesSiamese(embed,evidence,claimInfo,embeddings_evidence_test,siamese_model,k=None,thres=thProb)#(claimInfo,testEvidencesIds,siamese_model)
            #filtered = retrieveClosestEvidencesSiamese(claimInfo,testEvidencesIds,siamese_model,100)
            #filtered =retrieveClosestEvidenceCosine(claimInfo,testEvidencesIds,100)
    
            
            recallN=lambda filtered : np.sum([ev in filtered for ev in claimInfo["evidences"] ]) / (len(claimInfo["evidences"]) + 1e-15 )
            
            PrecisionN=lambda filtered : np.sum([ev in filtered for ev in claimInfo["evidences"] ]) / (len(filtered) + 1e-15)
            
            #ATh+=np.mean(list(distanceClaim.values())) - np.std(list(distanceClaim.values()))*np.sqrt(2*np.log(1/thresProb))
            
            avgK+=len(filtered)
            AR+=recallN(filtered)
            
            AP+=PrecisionN(filtered)
    
        AR=AR/(len(claims)  + 1e-15)
        AP=AP/(len(claims)  + 1e-15)
        avgK=avgK/(len(claims)  + 1e-15)
        #ATh=ATh/len(embeddings_claims)
        f1=1/((1/AR)+(1/AP))
        print("AR: ",AR,"AP: ",AP,"F1-score: ",f1,"th: ",thProb,"len(retrieval): ",avgK)
        listAR.append(AR)
        listAP.append(AP)
        listf1.append(f1)
    return listAR,listAP,listf1

class EvaluateCallback(tf.keras.callbacks.Callback):
    def __init__(self, embed,claims,evidence,possibleEvidenceIds, siamese_model,log_dir='/home/francisco/Documents/NLPproject/logs/metrics'):
        super(EvaluateCallback, self).__init__()
        self.claims = claims
        self.siamese_model = siamese_model
        self.possibleEvidenceIds=possibleEvidenceIds
        self.log_dir=log_dir
        self.embed=embed
        self.evidence=evidence
    def on_epoch_end(self, epoch, logs=None):
        # Evaluate the model
        AR, AP,f1= evaluateEvidenceRetrieval(self.embed,self.claims,self.evidence,self.siamese_model,self.possibleEvidenceIds)
        # Create a summary writer for TensorBoard
        file_writer = tf.summary.create_file_writer(self.log_dir)

        # Save the scalar summary
        with file_writer.as_default():
            tf.summary.scalar('AR', AR, step=epoch)
            tf.summary.scalar('AP', AP, step=epoch)
            tf.summary.scalar('f1score', f1, step=epoch)
            #tf.summary.scalar('Average distance threshold', ATh, step=epoch)
