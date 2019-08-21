import os
import time
import numpy as np
import re 

def cleanData(code_snippet):
    code_snippet = code_snippet.replace('&lt;',' ')
    code_snippet = code_snippet.replace('&gt;',' ')
    code_snippet = code_snippet.replace('\n',' ')
    code_snippet = code_snippet.replace('"',' ')
    code_snippet = code_snippet.replace(';','')
    code_snippet = code_snippet.replace('$','')
    code_snippet = code_snippet.replace('=',' ')
    code_snippet = code_snippet.replace(',','')
    code_snippet = code_snippet.replace('.',' ')
    code_snippet = code_snippet.replace(':',' ')
    code_snippet = code_snippet.replace('(',' ')
    code_snippet = code_snippet.replace(')',' ')
    code_snippet = code_snippet.replace('}',' ')
    code_snippet = code_snippet.replace('{',' ')  
    code_snippet = code_snippet.replace('\d+', ' ') # remove digits
    final_body = code_snippet.strip()
    final_body = re.sub(' +', ' ',final_body)
	
    return final_body

def predict(code_snippet,classifier,tfidfVectorizer):
    code_snippet = cleanData(code_snippet)
    test_tfidf = tfidfVectorizer.transform([code_snippet])
    test_pred = classifier.predict(test_tfidf)
    predictions_dict = {}
    predictions_dict['result'] = test_pred
    return predictions_dict

if __name__ == "__main__":
    predict("tr")
