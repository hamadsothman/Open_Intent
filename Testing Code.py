import nltk
import string
import spacy
import pandas as pd
import re
import numpy as np
from sklearn.manifold import TSNE
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)
nlp = spacy.load("en_core_web_md")
from spacy.symbols import dobj
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage


#Reading in Council Dataset
CouncilData = pd.read_csv("/Users/hh/Google Drive (hamadsothman@gmail.com)/DATA SCIENCE/DISO/Training Data/Training_Data_April.csv", nrows = 50)

pattern = re.compile(r"\d*([^\d\W]+)\d*")

#Making sure each word is lower case and striping white space
CouncilData["Description"] = CouncilData["Description"].str.lower().str.strip()
for word in CouncilData["Description"]:
    pattern.sub(r"\1", word)


#Creating empty list for training pairs
TrainingPairs = []

def Pair_Extraction(Text, List):
 for doc in Text.apply(nlp):
         for possible_subject in doc:
             #Removing stop words, digits and punctuations within the pipeline
            if possible_subject.is_stop == False and possible_subject.is_digit == False and possible_subject.is_punct == False:
                #Finding pairs of words with head verb and its child object
                if possible_subject.dep == dobj and possible_subject.head.pos_ == "VERB":
                    List.append([doc.text,possible_subject.text, possible_subject.head.text + " " + possible_subject.text,
                                doc.vector, possible_subject.head.vector, possible_subject.head.vector * possible_subject.vector])

#Running the Pair extraction method
Pair_Extraction(CouncilData["Description"], TrainingPairs)


# # Creating Dataframe for three outputs of sentence pairs and vectors
CouncilDF = pd.DataFrame(TrainingPairs, columns=("Sentence", "Subject", "Pair", "Sentence_Vector",
                                                 "Subject_Vector", "Pair_Vector"))
                         # dtype={"Sentence": str, "Subject": str,
                         #                     "Pair": str, "Vector_Pair": float,
                         #                     "Subject_Vector": float})

#Making sure the types are str
CouncilDF.Sentence.astype(str)
CouncilDF.Subject.astype(str)
CouncilDF.Pair.astype(str)


##Create model framework

#TSNE model with dimentionality reduction to two components
Tsne_model = TSNE(n_components = 2)


# #Creating Subject Vector Array for input to TSNE
Subject_Vector = np.array([vector for vector in CouncilDF["Subject_Vector"]])
#Doing same process for labels of subjects
Subject_Label = np.array([vector for vector in CouncilDF["Subject"]])


# Creating Pair Vector and Pair NP arrays
Pair_Vector = np.array([vector for vector in CouncilDF["Pair_Vector"]])

Pair_label = np.array([vector for vector in CouncilDF["Pair"]])

#Using TSNE and fitting and transforming the subject vectors
new_values = Tsne_model.fit_transform(Subject_Vector)

#Hierachical Clustering of subject
l = linkage(Subject_Vector, method = "complete", metric = "seuclidean")
plt.figure(figsize=(10,10))
plt.title("Clustering")
plt.ylabel("word")
plt.xlabel("distance")

dendrogram(
    l,
    leaf_rotation=0.,
    leaf_font_size=7.,
    orientation="left",
    leaf_label_func=lambda v: str(Subject_Label[v])
)
plt.show()


#Hierachical Clustering of pairs
l = linkage(Pair_Vector, method = "complete", metric = "seuclidean")
plt.figure(figsize=(10,10))
plt.title("Clustering")
plt.ylabel("word")
plt.xlabel("distance")

dendrogram(
    l,
    leaf_rotation=0.,
    leaf_font_size=7.,
    orientation="left",
    leaf_label_func=lambda v: str(Pair_label[v])
)
plt.show()

#Saving the Council DF
CouncilDF.to_csv(r'/Users/hh/Google Drive (hamadsothman@gmail.com)/DATA SCIENCE/DISO/Council_Test_DF.csv', header = True)

#Required if above method doesn't work
# #Running TSNE model on the Subject_Vector array
# new_values = Tsne_model.fit_transform(Subject_Vector)
# Subject_VecX = []
# Subject_VecY = []
#
# for value in new_values:
#     Subject_VecX.append(value[0])
#     Subject_VecY.append(value[1])
#
# plt.figure(figsize = (141,141))
#



