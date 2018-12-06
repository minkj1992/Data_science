from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import json
import numpy as np
import random

category_dict = {"안전/환경": 1,"미래": 2,"일자리": 3,"보건복지": 4,"정치개혁": 5,"경제민주화": 6,"인권/성평등": 7,"외교/통일/국방": 8,"육아/교육": 9,"문화/예술/체육/언론": 10,"반려동물": 11,"교통/건축/국토": 12,"행정": 13,"농산어촌": 14,"저출산/고령화대책": 15,"성장동력": 16,"기타": 17}
# tf_idf load
#vectorizing 한 pickle 파일을 불러온다.
# 육아/교육": 9,"문화/예술/체육/언론": 10,"반려동물": 11,"교통/건축/국토": 12,"행정": 13,"농산어촌": 14,"저출산/고령화대책": 15,"성장동력": 16,"기타": 17}
##here
category = "성장동력"
data_name = category_dict[category]
path  = './'+str(data_name)+'.pkl'



# 토큰화 된 corpus 데이터를 불러온다.
with open('corpus2.json','r') as f:
    json_corpus = json.load(f)
    

for i in category_dict.keys():
    if (i == "기타"):
        break
        
    print(str(i)+" is in train case")
    vectorizer = TfidfVectorizer()
    
    data = json_corpus[i]
    vectorizer.fit(data)
    y_train = [1]*len(data)
    
    data2 = json_corpus[i]
    y_test = [1]*len(data)
    
    data.extend(json_corpus['기타'])
#     x_train = vectorizer.fit_transform(data)
    x_train = vectorizer.transform(data)
    y_train.extend([0]*len(json_corpus['기타']))
    
    
    print(x_train.shape)
    print(len(y_train))
    
    
    
    #랜덤 카테고리 뽑아내기
    category_list = list(category_dict.keys())
    category_list.remove("기타")
    category_list.remove(i)
    test_cate = random.choice(category_list)
    print(str(test_cate)+" is in test case")

    data2.extend(json_corpus[test_cate])
#     빡친다.. 이거 변수 공유 자동으로 되서 padding일어남.

    
    x_test = vectorizer.transform(data2)
    y_test.extend([0]*len(json_corpus[test_cate]))

    
    
    
    print(x_test.shape)
    print(len(y_test))
# #     print("y_test len:")
# #     print(len(tmpy_test))
    
# #     print("x_test.shape")
# #     print(x_test.shape)

    
#     predict_unit = [json_corpus[test_cate][1000]]
#     test_unit = vectorizer.transform(predict_unit)
    print(x_train.shape)
    print(len(y_train))

    X,Y = shuffle(x_train,y_train)
    x_test,y_test = shuffle(x_test,y_test)
    break