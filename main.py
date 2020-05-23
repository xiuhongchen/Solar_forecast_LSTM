from __future__ import print_function

import argparse
import keras
import signal
from utils.utils import *

# python 2 and python 3
try:
    input = raw_input
except:
    pass
    
# deal with ctrl+c
class InterruptException(Exception):
    pass
    
def interrupt(signum, frame):
    raise InterruptException('')
    
    
def main(): 
    loss_list = []  
    '''
     Parse arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--new', action='store_true', help='if true, erase the local file and train a new model')
    parser.add_argument('-p', '--predict', action='store_true', help='if true, skip the training step and make predictions only')
    parser.add_argument('-a', '--all', action='store_true', help='if true, feed all related parameters into the model')
    parser.add_argument('-g1', '--graph-prediction', action='store_true', help='if true, generate the prediction graph')
    parser.add_argument('-g2', '--graph-scatter', action='store_true', help='if true, generate the scatter graph and the predictions in the output folder')
    args = parser.parse_args()
    print("New model:", args.new)
    print("Predictions only:", args.predict)
    if (args.graph_prediction and args.graph_scatter):
        print("Cannot generate two graphs at the same time.")
        return
    if (args.new==False):
        print('Warning: If you change the configs of the structure, you may use the "--new" or "-n" option.')
    ''' 
    Configurations:
    - t_x means the number of data for predicting the next one (similar to num_of_units in the previous version)
    - out_d means the output dimension of the rnn units
    - num_of_dense1 and num_of_dense2 are the numbers of cells in two dense layers
    - num_to_pred represents the number of hours ahead to be predicted.
    
    - epoch means the times of training(using all data)
    - batch size means the batch of data to feed during per training time.
    
    Updated at 08/14/2018:
        - Erase the default config, and it will now throw exceptions if there is no valid configurations. 
        - The original version may lead to misuse
    '''
    config = read_configs("config")
    if config == -1:
        return
    
    ''' 
        Load and Preprocess Datas
    
    Updated at 08/14/2018:
        - Change the order of parameters stored in the matrix
        - Change the way to pick consecutive data
    
    '''
    data_1 = scipy.io.loadmat(config["data_file"])
    x_mat = np.vstack((data_1["hour"], data_1["month"], data_1["relative_humidity"], data_1["air_temperature"], data_1["total_cloud_fraction"], data_1["toa_irradiance"], data_1["day"], data_1["year"]))
    y_mat = np.array(data_1["surface_irradiance"])
    x_and_y = np.hstack((y_mat.T, x_mat.T))
    if args.all:
        paras = x_and_y.shape[1]-2   # factors used for prediction
    else:
        paras = 1
    config['paras'] = paras
    t_x = config["t_x"]
    num_to_pred = config["num_to_pred"]
    consecutive_data_list = to_consecutive_list(x_and_y, t_x, day=-2, hour=1, label=0)
    training_data = np.array(consecutive_data_list).astype(int)
    training_data = np.reshape(training_data[:, :, :],[len(consecutive_data_list),t_x, -1])
    raw_data = copy.deepcopy(training_data)
    training_data = raw_data[0:int(0.8*len(consecutive_data_list)),:,:]
    test_data = raw_data[int(0.8*len(consecutive_data_list)):,:,:]
    print(raw_data.shape, training_data.shape, test_data.shape)
    
    '''
        Build the model
    
    Updated at 08/16/2018:
      - Now able to predict N-hour ahead solar irradiance by changing the "num_to_pred" config.
    '''
    if args.new:
        model = weather_forecast_model(config)
    else:
        with open('model-files/'+config["file_model"], 'r') as f:
            model = model_from_json(f.read())
    if not args.new:
        model.load_weights('model-files/' + config['file_weights'])
        print("Using pre-trained model.")
    model.compile(loss='mse',optimizer=keras.optimizers.Adam(lr=config["lr"]))
    print("")
    model.summary()
    
    ''' print loss '''
    if not args.predict:
        loss_train = model.evaluate(x=training_data[:, :-num_to_pred, :paras], y=training_data[:, -num_to_pred:, 0],batch_size=config['batch-size'])
        print('The loss of the training set is', loss_train)
        loss_test = model.evaluate(x=test_data[:, :-num_to_pred, :paras], y=test_data[:, -num_to_pred:, 0],batch_size=config['batch-size'])
        print('The loss of the test set is', loss_test)
        loss_list.append([loss_train, loss_test])
    
    ''' train '''
    if not args.predict:
        print("")
        try:
            signal.signal(signal.SIGINT, interrupt)
    
            for i in range(int(config["epoch"]/5)):
              
                model.fit(training_data[:, :-num_to_pred, :paras], training_data[:, -num_to_pred:, 0], epochs=5, batch_size=config['batch-size'])
                loss_train = model.evaluate(x=training_data[:, :-num_to_pred, :paras], y=training_data[:, -num_to_pred:, 0], batch_size=config['batch-size'])
                loss_test = model.evaluate(x=test_data[:, :-num_to_pred, :paras], y=test_data[:, -num_to_pred:, 0], batch_size=config['batch-size'])
                print('The loss of the training set is', loss_train)
                print('The loss of the test set is', loss_test)
                loss_list.append([loss_train, loss_test])
        except InterruptException:
            if input("\nInput y to save the model, otherwise abandon:  ")=='y':
                pass
            else:
                return
    
    ## print loss ##
    
    loss_train = model.evaluate(x=training_data[:, :-num_to_pred, :paras], y=training_data[:, -num_to_pred:, 0],batch_size=config['batch-size'])
    print('The loss of the training set is', loss_train)
    loss_test = model.evaluate(x=test_data[:, :-num_to_pred, :paras], y=test_data[:, -num_to_pred:, 0],batch_size=config['batch-size'])
    print('The loss of the test set is', loss_test)
    with open('loss-list.txt', 'a') as f:
        for tup in loss_list:
            f.write(str(tup[0]) + ', ' + str(tup[1]) + '\n')
    with open('model-files/'+config["file_model"], 'w') as f:
        f.write(model.to_json())
        print("Successfully save models to local")
    model.save_weights('model-files/' + config['file_weights'])
    print("Successfully save weights to local")
    
    '''
       Generate graphs
    '''
    if args.graph_prediction:
        prediction(model, training_data, config)
    if args.graph_scatter:
        scatter(model, test_data, config)
    

if __name__ == "__main__":
    
    main()
    print("##########################END OF PROGRAM##########################")
    print("")


