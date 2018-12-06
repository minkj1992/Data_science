from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import json
import numpy as np

category_dict = {"안전/환경": 1,"미래": 2,"일자리": 3,"보건복지": 4,"정치개혁": 5,"경제민주화": 6,"인권/성평등": 7,"외교/통일/국방": 8,"육아/교육": 9,"문화/예술/체육/언론": 10,"반려동물": 11,"교통/건축/국토": 12,"행정": 13,"농산어촌": 14,"저출산/고령화대책": 15,"성장동력": 16,"기타": 17}
# tf_idf load
#vectorizing 한 pickle 파일을 불러온다.
# 육아/교육": 9,"문화/예술/체육/언론": 10,"반려동물": 11,"교통/건축/국토": 12,"행정": 13,"농산어촌": 14,"저출산/고령화대책": 15,"성장동력": 16,"기타": 17}
##here
category = "성장동력"
data_name = category_dict[category]
path  = './'+str(data_name)+'.pkl'
with open(path, 'rb') as f:
    vectorizer = pickle.load(f)


# 토큰화 된 corpus 데이터를 불러온다.
with open('corpus2.json','r') as f:
    json_corpus = json.load(f)
    

add_list = list()
for categ,value in json_corpus.items():
    add_list.append(value)



#json 변환(단어: tfidf )
def display_scores(vectorizer, tfidf_result):
    # http://stackoverflow.com/questions/16078015/
    scores = zip(vectorizer.get_feature_names(),
                 np.asarray(tfidf_result.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return sorted_scores


# data = json_corpus[category]

vectorizer = TfidfVectorizer()
# x_train = vectorizer.fit_transform(data)
x_train = vectorizer.fit_transform(add_list)
print(len(vectorizer.vocabulary_))

tmp = display_scores(vectorizer, x_train)
# name = './'+str(data_name)+'_voca.json'
name = './total_voca.json'
with open(name,'w') as f:
    json.dump(dict(tmp),f)


# 혹시 모르니 tfidfvectorizer 다시 생성
vectorizer = TfidfVectorizer()
data = json_corpus[category]
#y_data 값 할당
y_train = [1]*len(data)
print(len(y_train))


# x_train = vectorizer.fit_transform(data)
vectorizer.fit(data)

# padding 시켜주는 data set(기타는 10000개 존재한다. 10000개 파일 padding)
data.extend(json_corpus['기타'])
# y_train = len(json_corpus['안전/환경'])까지는 1 + 10000(기타 데이터) =0
y_train.extend([0]*len(json_corpus['기타']))


# transform
x_train = vectorizer.transform(data)
print(len(data))
print(len(y_train))


# test_data (기타 이외의 다른 값을 test해준다.)
# 잘라내기
x_test = data[:len(y_train)]
y_test = y_train[:len(data)]

# 합치기
# x_test.extend(json_corpus['기타'])
x_test.extend(json_corpus['정치개혁'])
x_test = vectorizer.transform(x_test)
# y_test.extend([0]*(len(json_corpus['일자리'])+len(json_corpus['보건복지'])))
y_test.extend([0]*(len(json_corpus['정치개혁'])))


# predict unit
predict_unit = [json_corpus['정치개혁'][1000]]
test_unit = vectorizer.transform(predict_unit)

from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import RFE
# from sklearn.neural_neural_network import MLP
from sklearn.utils import shuffle, gen_batches


X,Y = shuffle(x_train,y_train)
x_test,y_test = shuffle(x_test,y_test)

batch_size = 500
n_samples= X.shape[0]
# logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
# c = 규제의 강도를 결정하는 매개변수, 높아지면 규제가 감소, 즉 c가 커지면 train에 overfit, c를 낮추면 


# classifier LogisticRegression
clf = LogisticRegression().fit(X, Y)
print("훈련 세트 점수: {:.3f}".format(clf.score(X,Y)))
print("테스트 세트 점수: {:.3f}".format(clf.score(x_test, y_test)))
predictions = clf.predict(test_unit)
print(predictions)

# classifier Perceptron
clf = Perceptron(random_state=0, tol=1e-5)
clf.fit(X,Y)
print("훈련 세트 점수: {:.3f}".format(clf.score(X,Y)))
print("테스트 세트 점수: {:.3f}".format(clf.score(x_test, y_test)))
predictions = clf.predict(test_unit)
print(predictions)


# classifier LinearSVC
clf = LinearSVC(random_state=0, tol=1e-5)
clf.fit(X,Y)
print("훈련 세트 점수: {:.3f}".format(clf.score(X,Y)))
print("테스트 세트 점수: {:.3f}".format(clf.score(x_test, y_test)))
predictions = clf.predict(test_unit)
print(predictions)


# classifier SGDClassifier
# tol == early stopping
clf = SGDClassifier(max_iter=1000, tol=1e-5)
clf.fit(X,Y)
print("훈련 세트 점수: {:.3f}".format(clf.score(X,Y)))
print("테스트 세트 점수: {:.3f}".format(clf.score(x_test, y_test)))
predictions = clf.predict(test_unit)
print(predictions)



# for batch in gen_batches(n_samples, batch_size):
    
#     classifier = LogisticRegression().fit(X, Y)
#     these_sample_indices = get_sub_slice(Y, batch)
#     classifier.fit(X[batch],Y[batch])
#     print("훈련 세트 점수: {:.3f}".format(classifier.score(X, Y)))
#     print("훈련 세트 점수: {:.3f}".format(classifier.score(X[batch],Y[batch])))

# print("테스트 세트 점수: {:.3f}".format(classifier.score(x_test, y_test)))


# 1. 각 데이터 별 4가지 기법 도입
# 2. keras linear 모델 실시




# x_train

# classifier SGDClassifier
# tol == early stopping
clf = SGDClassifier(max_iter=1000, tol=1e-5)
clf.fit(X,Y)
print("훈련 세트 점수: {:.3f}".format(clf.score(X,Y)))
print("테스트 세트 점수: {:.3f}".format(clf.score(x_test, y_test)))
predictions = clf.predict(test_unit)
print(predictions)


# 3. word2vec -> classification with my project (attention mec) 



