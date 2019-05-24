import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

#change XLSX_FILE and write the text on which ther will be classification

text=['']
lista=[]


def Classif(tek):
    text=[]

    for i in range(0, len(tek)):
        df = pd.read_excel(XLSX_FILE)
        df = df.fillna('')
        training_labels=tuple(df['Description'].values.tolist())
        training_texts=tuple(df['Sentence'].values.tolist())
        test_label=tuple(df['Description_test'].values.tolist())
        test_text=tuple(df['Sentence_test'].values.tolist())
        
        test_labels = [x for x in test_label if x !='']
        test_texts = [x for x in test_text if x !='']
        
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(training_texts)
        y = training_labels
        
        clf = SGDClassifier()
        clf.fit(X, y)
        X_test = vectorizer.transform(test_texts)
        y_test = test_labels
        
        test_predictions = clf.predict(X_test)
        annotated_test_data = list(zip(test_predictions, test_texts))
        #for i in range(0,len(annotated_test_data)):
            #print(annotated_test_data[i])
        
        y_test = np.array(test_labels)
        #print(metrics.classification_report(y_test, test_predictions))
        #print("Trafnosc: %0.4f" % metrics.accuracy_score(y_test, test_predictions))
        
        check = vectorizer.transform([tek[i]])
        pred=clf.predict(check)
        print('\nResult:')
        print(pred)
        text.append(str(pred).replace("['","").replace("']",""))
        
    return text


lista=Classif(text)