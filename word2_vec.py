import  pandas as pd
import gensim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


path_="../spam.csv"

data_=pd.read_csv(path_,encoding="latin-1")
data_=data_.drop(labels=["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)
data_.columns= ["labels","text"]


# Let's clean the data!
print(data_)

data_["text_clean"]= data_["text"].apply(lambda x: gensim.utils.simple_preprocess(x))

X_train, X_test, Y_train, Y_test= train_test_split(data_["text_clean"], data_["labels"],
                                                   test_size=0.2)
# Let me write the parameters
w2v_model= gensim.models.Word2Vec(
    X_train,
    size=100,
    window=5,
    min_count=2
)

# Let me get the word vector of the "king"

vector__ = w2v_model.wv["king"]
print(vector__)

# Let me find the most similar words  to "king" based on word vectors from our trained model

similarities= w2v_model.wv.most_similar("king")
print(similarities)

# Let me show all of the words in the corpus! These words are all the words that word2vec model learned for in the training data at least twice
words_ =w2v_model.wv.index2word

print(words_)


# Let me generate aggregated sentence vectors based on the word vectors for each word in the sentence


print(X_test)

w2v_vector= np.array(np.array([w2v_model.wv[i] for ls in X_test for i in ls if i in w2v_model.wv.index2word]))

#
# w2v_vector_avg=[]
#
# for vect in w2v_vector:
#     if len(vect)!=0:
#         w2v_vector_avg.append(vect.mean(axis=0))
#     else:
#         w2v_vector_avg.append(np.zeros(100))
#
# for i,v in enumerate(w2v_vector_avg):
#     print(len(X_test.iloc[i]),len(v))
#
