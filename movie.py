# %% [markdown]
# 네이버 영화 리뷰 감성 분류하기(Naver Movie Review Sentiment Analysis)
# 
# 1. 네이버 영화 리뷰 데이터에 대한 이해와 전처리

# %%
# pip install konlpy

# %%
# 패키지 준비

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# %%
# 데이터 로드하기

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

# %%
# Pandas를 이용하여 훈련 데이터는 train_data에 테스트 데이터는 test_data에 저장

train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')

# %%
# train_data에 존재하는 영화 리뷰의 개수를 확인

print('훈령용 리뷰 개수 :', len(train_data))

# %%
# 상위 5개 출력하여 데이터 확인

train_data[:5]

# %%
#  test_data의 개수와 상위 5개의 샘플을 확인

print('테스트용 리뷰 개수 :', len(test_data))

# %%
# 상위 5개 출력하여 데이터 확인

test_data[:5]

# %%
# 데이터 정제하기

# %%
# 데이터 중복 확인

train_data['document'].nunique(), train_data['label'].nunique()

# %%
# 중복 샘플 제거

train_data.drop_duplicates(subset=['document'], inplace=True)

# %%
# 전체 샘플 수 확인

print('총 샘플의 수 :', len(train_data))

# %%
# 레이블(label) 값의 분포 확인

train_data['label'].value_counts().plot(kind='bar')

# %%
# 각 레이블의 개수 확인

print(train_data.groupby('label').size().reset_index(name = 'count'))

# %%
# 리뷰 중의 null 값 확인

print(train_data.isnull().values.any())

# %%
# 어떤 열에 존재하는지 확인

print(train_data.isnull().sum())

# %%
# Null 값을 가진 샘플이 어느 인덱스의 위치에 존재하는지 출력

train_data.loc[train_data.document.isnull()]

# %%
# Null 값을 가진 샘플을 제거

train_data = train_data.dropna(how = 'any') # Null 값이 존재하는 행 제거

print(train_data.isnull().values.any()) # Null 값이 존재하는지 확인

# %%
# 확인

print(len(train_data))

# %%
# 데이터 전처리

# train_data에 한글과 공백을 제외하고 모두 제거하는 정규 표현식을 수행

train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

train_data[:5]

# %% [markdown]
# 그런데 사실 네이버 영화 리뷰는 굳이 한글이 아니라 영어, 숫자, 특수문자로도 리뷰를 업로드할 수 있습니다. 다시 말해 기존에 한글이 없는 리뷰였다면 이제 더 이상 아무런 값도 없는 빈(empty) 값이 되었을 것임

# %%
# train_data에 공백만 있거나 빈 값을 가진 행을 Null 값으로 변경
# Null 값 확인

train_data['document'] = train_data['document'].str.replace('^ +', "") # white space 데이터를 empty value로 변경
train_data['document'].replace('', np.nan, inplace=True)
print(train_data.isnull().sum())

# %%
# Null 값이 있는 행을 5개 출력

train_data.loc[train_data.document.isnull()][:5]

# %%
# 의미 없으므로 제거

train_data = train_data.dropna(how = 'any')

print(len(train_data))

# %%
test_data.drop_duplicates(subset = ['document'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") # 정규 표현식 수행
test_data['document'] = test_data['document'].str.replace('^ +', "") # 공백은 empty 값으로 변경
test_data['document'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경
test_data = test_data.dropna(how='any') # Null 값 제거
print('전처리 후 테스트용 샘플의 개수 :',len(test_data))

# %% [markdown]
# 3) 토큰화

# %%
# 불용어 정의

stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

# %%
# 형태소 분석기를 사용하여 토큰화를 하면서 불용어를 제거하여 X_train에 저장

okt = Okt()

X_train = []
for sentence in train_data['document']:
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
    X_train.append(temp_X)

# %%
# 상위 3개 출력하여 확인

print(X_train[:3])

# %%
# 테스트 데이터 토큰화

X_test = []
for sentence in test_data['document']:
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
    X_test.append(temp_X)

# %%
print(X_test[:3])

# %% [markdown]
# 4) 정수 인코딩 : 기계가 텍스트를 숫자로 처리할 수 있도록 훈련 데이터와 테스트 데이터에 정수 인코딩을 수행

# %%
# 단어 집합 생성

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

print(tokenizer.word_index)

# %%
# 등장 빈도수가 3회 미만인 단어들의 분포 확인

threshold = 3
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

# %%
# 등장 빈도수가 2이하인 단어들의 수를 제외한 단어의 개수를 단어 집합의 최대 크기로 제한

# 전체 단어 개수 중 빈도수 2이하인 단어는 제거.
# 0번 패딩 토큰을 고려하여 + 1

vocab_size = total_cnt - rare_cnt + 1
print('단어 집합의 크기 :',vocab_size)

# %%
# 이를 케라스 토크나이저의 인자로 넘겨 텍스트 시퀀스를 숫자 시퀀스로 변환

tokenizer = Tokenizer(vocab_size) 
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# %%
# 정수 인코딩 결과 확인

print(X_train[:3])

# %%
# train_data에서 y_train과 y_test를 별도로 저장

y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])

# %% [markdown]
# 6) 패딩

# %%
print('리뷰의 최대 길이 :',max(len(l) for l in X_train))
print('리뷰의 평균 길이 :',sum(map(len, X_train))/len(X_train))
plt.hist([len(s) for s in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show();

# %%
# 전체 샘플 중 길이가 max_len 이하인 샘플의 비율이 몇 %인지 확인

def below_threshold_len(max_len, nested_list):
  cnt = 0
  for s in nested_list:
    if(len(s) <= max_len):
        cnt = cnt + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))

# %%
# max_len = 30이 적당할 것 같습니다. 이 값이 얼마나 많은 리뷰 길이를 커버하는지 확인

max_len = 30
below_threshold_len(max_len, X_train)

# %%
X_train = pad_sequences(X_train, maxlen = max_len)
X_test = pad_sequences(X_test, maxlen = max_len)

# %% [markdown]
# 2. LSTM으로 네이버 영화 리뷰 감성 분류하기

# %%
# 패키지 준비

from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# %%
# 모델 설계

model = Sequential()
model.add(Embedding(vocab_size, 100))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

# %%
# 모델 검증
from keras.callbacks import EarlyStopping, ModelCheckpoint


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.keras', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

# %%
# 모델 훈련

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=60, validation_split=0.2)

# %%
# 테스트 정확도 측정

loaded_model = load_model('best_model.keras')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

# %% [markdown]
# 3. 리뷰 예측

# %%
def sentiment_predict(new_sentence):
  new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
  new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
  encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
  pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
  score = float(loaded_model.predict(pad_new)) # 예측
  if(score > 0.5):
    print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100))
  else:
    print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100))

# %%
sentiment_predict('이 영화 개꿀잼 ㅋㅋㅋ')

# %%
sentiment_predict('오마쥬야 카피야!!')

# %%
sentiment_predict('이게 맞냐..?')


