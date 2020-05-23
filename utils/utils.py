from __future__ import print_function
import numpy as np
import keras
from keras.models import Sequential, model_from_json, Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Lambda, ActivityRegularization
from keras import backend as K
from keras import regularizers
import numpy as np
import tensorflow as tf
import scipy.io
# import matplotlib.pyplot as plt
import random
import copy

    
'''
  This function is to ignore bad data such as -1s or NaNs, and pick the consecutive ones.
  Last Major Updated 08/14/2018:
    - Change the way of judging whether data is consecutive.
'''
def to_consecutive_list(x_and_y, t_x, hour, day, label=0):
    consecutive_list = []
    count = 0
    temp_start = 0
    for i in range(x_and_y.shape[0]):
        if False:
            temp_start = -1
        else:
            if temp_start == -1:
                temp_start = i
        if (i-temp_start) >= t_x and temp_start != -1:
            consecutive_list.append(x_and_y[i-t_x + 1: i + 1, :])

    return consecutive_list


'''
  This is to build a model. Now there is a LSTM layer and a dense layer following it.
  Updated at 08/15/2018:
    - Add a layer to deal with zeros.
    - Change the implementation of the model. 
  
  Updated at 08/16/2018:
    - Now able to predict N-hour ahead solar irradiance by changing the "num_to_pred" config.
    
  Updated at 08/27/2018:
    - Insert a batch norm layer between the LSTM and dense layers.
'''
def weather_forecast_model(config):
    t_x = config["t_x"]
    num_to_pred = config["num_to_pred"]
    inp = Input(shape=(t_x-num_to_pred, config['paras']))
    y = keras.layers.normalization.BatchNormalization(input_shape=(t_x-num_to_pred,config['paras']), epsilon=1e-6)(inp)
    y = LSTM(config["out_d"], kernel_regularizer=regularizers.l2(10), recurrent_regularizer=regularizers.l2(10), return_sequences=False, name="lstm_1", dropout=0)(y)
    y = Dropout(config["dropout"])(y)
    y = keras.layers.normalization.BatchNormalization(epsilon=1e-6)(y)
    y = Dense(config["num_to_pred"], kernel_regularizer=regularizers.l2(100), bias_regularizer=regularizers.l2(100))(y)
    y = Activation('relu')(y)
    model = Model(inputs=inp, outputs=y)
    return model
    
'''
    To plot predictions and save to files
    
    Updated at 08/21/2018:
      - Save corresponding time with labels.
'''
def prediction(model, test_data, config):
    indices = {"value":0, "hour":1, "day":-2, "month":3, "year":-1}
    t_x = config["t_x"]
    num_to_pred = config["num_to_pred"]
    rd = random.randrange(0, int(0.8*test_data.shape[0]))
    to_plot_real = [] 
    to_plot_pred = []
    years, months, days, hours = [], [], [], []
    to_plot_real = (np.reshape(test_data[rd, -num_to_pred:, 0], [num_to_pred]))
    feed_x = np.reshape(test_data[rd, :-num_to_pred, :-2], [1,t_x-num_to_pred,-1])
    preds = model.predict(x=feed_x[:, :, 0:config['paras']])
    to_plot_pred = (np.reshape(preds, [num_to_pred]))
    years = test_data[rd, -num_to_pred:, indices["year"]]
    months = test_data[rd, -num_to_pred:, indices["month"]]
    days = test_data[rd, -num_to_pred:, indices["day"]]
    hours = test_data[rd, -num_to_pred:, indices["hour"]]
    scipy.io.savemat('output/short-term-predictions.mat', {'pred': to_plot_pred, 'label':to_plot_real, 'year': years, 'month': months, 'day':days, 'hour': hours})

'''
    To plot scatters and save to files
'''
def scatter(model, test_data, config):
    indices = {"value":0, "hour":1, "day":-2, "month":2, "year":-1}
    t_x = config["t_x"]
    num_to_pred = config["num_to_pred"]
    print("Generating scatters... It may take more than a minute..")
    to_plot_pred = []
    to_plot_real = []
    years, months, days, hours, hours_ahead = [], [], [], [], []
    for i in range(test_data.shape[0]):
        feed_x = np.reshape(test_data[i, :-num_to_pred, :-2], [1, t_x-num_to_pred, -1])
        preds = np.reshape(model.predict(x=feed_x[0:config['paras']]), [-1])
        real = test_data[i, -num_to_pred:, indices["value"]]
        years += (test_data[i, -num_to_pred:, indices["year"]].tolist())
        months.extend(test_data[i, -num_to_pred:, indices["month"]].tolist())
        days.extend(test_data[i, -num_to_pred:, indices["day"]].tolist())
        hours.extend(test_data[i, -num_to_pred:, indices["hour"]].tolist())
        hours_ahead.extend(range(1, num_to_pred+1))
        to_plot_pred.extend(preds.tolist())
        to_plot_real.extend(real.tolist())
    # plt.scatter(to_plot_real, to_plot_pred, s=1)
    # plt.plot([0,1200], [0,1200], color="red")
    # plt.savefig('images/scatter.png', format='png')
    scipy.io.savemat('output/all-predictions.mat', {'pred': to_plot_pred, 'label':to_plot_real, 'year': years, 'month': months, 'day':days, 'hour': hours, 'hours_ahead': hours_ahead})
    print("Graph has been generated")

def read_configs(filename):
    try:
        with open(filename, 'r') as f:
            return eval(f.read())
    except Exception as e:
        print(e)
        print("Invalid data format or file not found.")
        print("Check whether the data is in json format")
        return -1
    
    
if __name__ == "__main__":
# use for test purpose
    test = np.array([
        [2018, 8, 3, 14, 1, -1],
        [2018, 8, 3, 14, 2, -1],
        [2018, 8, 3, 14, 3, -1],
        [2018, 8, 3, 14, 4, 2],
        [2018, 8, 3, 14, 5, 2],
        [2018, 8, 4, 14, 6, 2],
        [2018, 8, 4, 14, 7, 2],
        [2018, 8, 4, 14, 8, 2],
        [2018, 8, 4, 14, 9, -1],
        [2018, 8, 4, 14, 10, 2],
        [2018, 8, 4, 14, 11, 2],
        [2018, 8, 4, 14, 12, 2],
        [2018, 8, 4, 14, 13, 2],
    ])
    for item in to_consecutive_list(test, 2):
        print(item)
