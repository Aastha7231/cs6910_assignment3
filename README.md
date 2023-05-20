# cs6910_assignment3
Assignment 3 of the course CS6910: Fundamentals of Deep Learning offered at IIT Madras by Aastha Tiwari (CS22M005)

## Train.py

* This python file train.py helps in building an RNN based seq2seq model which supports lstm, gru and rnn cells.
* The function final_run() contains a configuration dictionary that takes various parameters as input on command line using argparse to train the model.
* Firstly data pre-processing is done to convert word pairs into tensors.
* The function train_iters() takes various parameters as input from the configuration dictionary and trains the model for 10 epochs by calling the train() function. 
* The train function returns the loss after training the model at each epoch.
* The model is then evaluated on the validation dataset using the evaluate function at each epoch.

## Evaluate the Best Model
* After running the sweep with and without attention as mentioned in the report the best configuration obtained is :\
   configuration = {\
                "hidden_size" : 512,\
                "input_lang" : 'eng',\
                "output_lang" : 'hin',\
                "cell_type"   : 'LSTM',\
                "num_layers_encoder" : 2,\
                "num_layers_decoder" : 2,\
                 "embedding_size" : 128,\
                "bi_directional" : True,\
                "batch_size" : 32,\
                "attention" : False ,\
                "learning_rate" : 0.001,\
                "drop_out" : 0.2,\
         }
* The function evaluate_testset is run to find test accuracy for best model configuration.
* The best accuracy obtained on validation set is - 38.94% 
* The best accuracy obtained on test dataset is -36.92%  and loss - 0.3086

## Predictions_Vanilla
The folder predictions_vanilla contains a file predictions_vanilla.csv that contains the input words, predicted words in hindi and also the actual output in hindi for the english word. This is obtained when the code is executed without attention.

## Predictions_attention
The folder predictions_attention contains a file predictions_attention.csv that contains the input words, predicted words in hindi and also the actual output in hindi for the english word. This is obtained when the code is executed with attention.

## Sweep Configuration Used
The following configuration is used to run the sweep on wandb.\
Inorder to run the sweep on wandb uncomment the sweep configuration in the code and also uncomment the call to sweepfunction() in the code .\
![Screenshot (199)](https://github.com/Aastha7231/cs6910_assignment3/assets/126596782/33e37d6e-f9bf-4b42-a72a-058c4ab92144)


