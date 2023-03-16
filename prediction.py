import pandas as pd
from matplotlib import pyplot as plot
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("dataset/test.csv")
unique_labels = data['target'].drop_duplicates()
sorted_lables = sorted(unique_labels)
print("sorted_lables:: "+str(sorted_lables))
#%%
test_data = data
print("test_data.shape:: "+str(test_data.shape))

target_labels = test_data['target']
plot.hist(target_labels)
plot.xticks(sorted_lables, sorted_lables);
plot.show()
#%%
tfidf = TfidfVectorizer(lowercase=False, analyzer='char', ngram_range=(1,5), max_features=20000)

test_data_features = tfidf.transform(test_data['ciphertext'])

test_data_x = test_data_features.tocsr()

del(tfidf)
#%%
model = Pipeline(memory=None, steps=[
        ('scaler', MaxAbsScaler(copy=False)),
        ('clf', LogisticRegression(multi_class='multinomial', verbose=2, n_jobs=-1))
    ])

# model.load_model("ciphertext_model.json")
model = load(open('ciphertext_model.pkl', 'rb'))
predictions = model.predict(test_data_x)
#%%
cm = confusion_matrix(test_data['target'], predictions)
plot.imshow(cm)
plot.ylabel('Actual target label')
plot.xlabel('Predicted label')
print(classification_report(test_data['target'], predictions, digits=3))

print("final predictions", predictions)
#%%
