# cs6910_assignment3
Assignment 3 of the course CS6910: Fundamentals of Deep Learning offered at IIT Madras by Aastha Tiwari (CS22M005)

## Train.py

* This python file train.py helps in building an RNN based seq2seq model which supports lstm, gru and rnn cells.
* The function final_run() contains a configuration dictionary that takes various parameters as input on command line using argparse to train the model.
* Firstly data pre-processing is done to convert word pairs into tensors.
* The function train_iters() takes various parameters as input from the configuration dictionary and trains the model for 10 epochs by calling the train() function. 
* The train function returns the loss after training the model at each epoch.
* The model is then evaluated on the validation dataset using the evaluate function at each epoch.

### Evaluate the Best Model
* After running the sweep with and without attention as mentioned in the report the best configuration obtained is :\
   configuration = {\
             |   "hidden_size" : 512,\
             |   "input_lang" : 'eng',\
             |   "output_lang" : 'hin',\
             |   "cell_type"   : 'LSTM',\
             |   "num_layers_encoder" : 2,\
             |   "num_layers_decoder" : 2,\
             |   "drop_out"    : 0.2,\ 
             |   "embedding_size" : 128,\
             |   "bi_directional" : True,\
             |   "batch_size" : 32,\
             |   "attention" : False ,\
             |   "learning_rate" : 0.001,\
         }
* The best accuracy obtained on validation set is - 38.94% 
* The best accuracu obtained on test dataset is -36.92%  and loss - 0.3086
* 
