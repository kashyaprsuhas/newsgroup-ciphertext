import pandas as pd
from matplotlib import pyplot as plot
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from pickle import dump

data = pd.read_csv("dataset/train.csv")
unique_labels = data['target'].drop_duplicates()
sorted_lables = sorted(unique_labels)
print("sorted_lables:: "+str(sorted_lables))
#%%
train_limit = 5000
test_limit = train_limit + 1
train_data = data[:train_limit]
test_data = data[test_limit:]
print("train_data.shape:: "+str(train_data.shape))
print("test_data.shape:: "+str(test_data.shape))

target_labels = train_data['target']
plot.hist(target_labels)
plot.xticks(sorted_lables, sorted_lables);
plot.show()

target_labels = test_data['target']
plot.hist(target_labels)
plot.xticks(sorted_lables, sorted_lables);
plot.show()
#%%
tfidf = TfidfVectorizer(lowercase=False, analyzer='char', ngram_range=(1,5), max_features=20000)
train_data_features = tfidf.fit_transform(train_data['ciphertext'])

train_data_x = train_data_features.tocsr()

test_data_features = tfidf.transform(test_data['ciphertext'])

test_data_x = test_data_features.tocsr()

#%%
model = Pipeline(memory=None, steps=[
        ('scaler', MaxAbsScaler(copy=False)),
        ('clf', LogisticRegression(multi_class='multinomial', verbose=2, n_jobs=-1))
    ])

model.fit(train_data_x, train_data['target'])
#%%

# model.save_model("ciphertext_model.json")
dump(tfidf, open('tfidf.pkl', 'wb'))
dump(model, open('ciphertext_model.pkl', 'wb'))
#%%
