
# coding: utf-8

# In[7]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from sklearn import preprocessing
import urllib
import os.path

def make_train_test ( tag ) :
    csv_file =  tag # + '.csv'
    print (csv_file)

    df = pd.read_csv(csv_file)
    label = df['Label']
    le = preprocessing.LabelEncoder()
    le.fit(list(label.value_counts().index))
    num_classes = label.value_counts().shape[0]  # 210
    le_label =  keras.utils.to_categorical(le.transform(list(label)), num_classes)
    df = df.drop(columns = ['Label'], axis = 1)
    x_train, x_test, y_train, y_test = train_test_split(df, le_label, test_size=0.3)
    print ( 'x_train:%s, x_test:%s, y_train:%s, y_test:%s' % (x_train.shape, x_test.shape, y_train.shape, y_test.shape))
    return x_train, x_test, y_train, y_test, num_classes, le, le_label , csv_file

def make_model (x_train, x_test, y_train, y_test, num_classes) :
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(3,)))
    model.add(Dropout(0.25))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.20))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    #model.compile(loss='categorical_crossentropy', optimizer=RMSprop(),  metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),  metrics=['accuracy'])
    batch_size = 2048; epochs = 5
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return model

def save_model ( model, tag) :
    json_file = tag + '.json'
    h5_file = tag + '.h5'

    model_json = model.to_json()
    with open(json_file , "w") as _json_file:
        _json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights( h5_file)
    print("Saved %s and %s to disk" % (json_file, h5_file) )
    return model, json_file , h5_file

def do_predict_slice (model, slice = None) :
    if not slice:
      slice = x_test.iloc[0:1,:]

    sample_df = pd.DataFrame( slice)
    ans = model.predict( sample_df)
    y_pred_idx = [pd.Series(elem).idxmax() for elem in ans]
    y_pred = [ le.classes_[n] for n in y_pred_idx]

    if False:
      print("=  SLICE ==============")
      print(sample_df)
      print(ans)
      print(y_pred_idx)
      print(y_pred)
      print("==================")
    return y_pred, sample_df

def do_predict ( model, le, r, g, b):
    sample_df = pd.DataFrame( {'R':[r], 'G':[g] , 'B':[b] }, columns=['R', 'G', 'B'])
    ans = model.predict( sample_df)
    y_pred_idx = [pd.Series(elem).idxmax() for elem in ans]
    y_pred = [ le.classes_[n] for n in y_pred_idx]

    return y_pred, sample_df

def salila_ml (tag, r, g, b) :
    x_train, x_test, y_train, y_test, num_classes, le, le_label , csv_file = make_train_test (tag )
    model = make_model ( x_train, x_test, y_train, y_test, num_classes )
    model, json_file, h5_file = save_model ( model, tag )

    y_pred , x_instance = do_predict (model, le, r, g, b)

    return y_pred, x_instance

# if __name__ == '__main__' :
#     y_pred, x_instance = salila_ml('../Fluoride_colors.csv' , 45, 117, 103)
#     print (x_instance, y_pred ) ; print("=======")
#     y_pred, x_instance = salila_ml('../Fluoride_colors.csv' , 131, 149, 156)
#     print (x_instance, y_pred ) ; print("=======")