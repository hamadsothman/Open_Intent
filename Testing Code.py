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


# lemmatizer = nlp.get_pipe("lemmatizer")


# Reading in Twitter Dataset
Twitter = pd.read_csv("/Users/hh/Google Drive (hamadsothman@gmail.com)/DATA SCIENCE/DISO/sample.csv", nrows=50)
#Reading in Airline Dataset
AirlineData = pd.read_csv("/Users/hh/Google Drive (hamadsothman@gmail.com)/DATA SCIENCE/DISO/AirlineData.csv",nrows =50)
#Reading in Council Dataset
CouncilData = pd.read_csv("/Users/hh/Google Drive (hamadsothman@gmail.com)/DATA SCIENCE/DISO/Training Data/Training_Data_April.csv")

pattern = re.compile(r"\d*([^\d\W]+)\d*")


CouncilData["Description"] = CouncilData["Description"].str.lower().str.strip()
for word in CouncilData["Description"]:
    pattern.sub(r"\1", word)



# #Tokenizing each row in Twitter file
# # TwitterText = Twitter["text"].apply(nlp)
# # AirlineText = AirlineData["Text"].apply(nlp)
# #
#
# # Finding a verb with a subject from below — good
# #Dependency Parsing and finding vector
# TwitterPairs = []
# AirlinePairs = []

#Creating empty list for training pairs
TrainingPairs = []

def Pair_Extraction(Text, List):
 for doc in Text.apply(nlp):
         for possible_subject in doc:
             #Removing stop words, digits and punctuations within the pipeline
            if possible_subject.is_stop == False and possible_subject.is_digit == False and possible_subject.is_punct == False:
                #Finding pairs of words with head verb and its child
                if possible_subject.dep == dobj and possible_subject.head.pos_ == "VERB":
                    List.append([doc.text,possible_subject.text, possible_subject.head.text + " " + possible_subject.text,
                                doc.vector, possible_subject.head.vector, possible_subject.head.vector * possible_subject.vector])
Pair_Extraction(CouncilData["Description"], TrainingPairs)

# # Pair_Extraction(Twitter["text"], TwitterPairs)
# # Pair_Extraction(AirlineData["Text"], AirlinePairs)

# # Creating Dataframe for three outputs of sentence pairs and vectors
# # TwitterDF = pd.DataFrame(TwitterPairs, columns=("Sentence", "Pair", "Vector"))
# # AirlineDF = pd.DataFrame(AirlinePairs, columns=("Sentence", "Pair", "Vector"))
CouncilDF = pd.DataFrame(TrainingPairs, columns=("Sentence", "Subject", "Pair", "Sentence_Vector",
                                                 "Subject_Vector", "Pair_Vector"))
                         # dtype={"Sentence": str, "Subject": str,
                         #                     "Pair": str, "Vector_Pair": float,
                         #                     "Subject_Vector": float})
CouncilDF.Sentence.astype(str)
CouncilDF.Subject.astype(str)
CouncilDF.Pair.astype(str)
#
#
#Create model framework
Tsne_model = TSNE(n_components = 2)

# # Subject_Vector = []
# #Creating Subject Vector Array for input to TSNE
Subject_Vector = np.array([vector for vector in CouncilDF["Subject_Vector"]])
#
Subject_Label = np.array([vector for vector in CouncilDF["Subject"]])


# Creating Pair Vector and Pair NP arrays

Pair_Vector = np.array([vector for vector in CouncilDF["Pair_Vector"]])

Pair_label = np.array([vector for vector in CouncilDF["Pair"]])

new_values = Tsne_model.fit_transform(Subject_Vector)

#Hierachical Clustering
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
# #
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
#
# print(CouncilDF.dtypes)

print(CouncilData)



# def cleaner(df):
#     "Extract relevant text from DataFrame using a regex"
#     # Regex pattern for only alphanumeric, hyphenated text with 3 or more chars
#     pattern = re.compile(r"[A-Za-z0-9\-]{3,50}")
#     df['clean'] = df['Text'].str.findall(pattern).str.join(' ')
#     if limit > 0:
#         return df.iloc[:limit, :].copy()
#     else:
#         return df
# cleaner(AirlineData["Text"])
# print(AirlineData.head())
#
# def lemmatize_pipe(doc):
#     lemma_list = [str(tok.lemma_).lower() for tok in doc
#                   if tok.is_alpha and tok.text.lower() not in stopwords]
#     return lemma_list
#
# def preprocess_pipe(texts):
#     preproc_pipe = []
#     for doc in nlp.pipe(texts, batch_size=20):
#         preproc_pipe.append(lemmatize_pipe(doc))
#     return preproc_pipe
# # for token in AirlineTokenize:
# #     print(token, token.pos_)

# You want list of Verb tokens
# print("Verbs:", [token.text for token in AirlineTokenize if token.pos_ == "VERB"])

# NLPAirline = nlp(AirlineTokenize["Text"])
# print(NLPAirline.head())
#
# # AirlinePos = AirlineTokenize.apply(lambda x: print(x.pos_))

# for token in AirlineTokenize:
#     # Print the text and the predicted part-of-speech tag
#     print(token.text, token.pos_)
