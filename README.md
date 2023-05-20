# cs6910_assignment3
Assignment 3 of the course CS6910: Fundamentals of Deep Learning offered at IIT Madras by Aastha Tiwari (CS22M005)

##Train.py

*This python file train.py helps in building an RNN based seq2seq model which supports lstm, gru and rnn cells.
*The function final_run() contains a configuration dictionary that takes various parameters as input on command line using argparse to train the model.
*Firstly data pre-processing is done to convert word pairs into tensors.
*The function train_iters() takes various parameters as input from the configuration dictionary and trains the model for 10 epochs by calling the train() function. 
*The train function returns the loss after training the model at each epoch.
*The model is then evaluated on the validation dataset using the evaluate function at each epoch.



