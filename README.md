## Requirements

* numpy
* pandas
* torch
* sklearn
* lightgbm
* matplotlib

## Project Structure

* Data: include primitive dataset and the processed data set
* logs: record the log file produced by tensorboard
* Method_Code(without feature selecting): use all features to train model and you can choose one model in this file
* model: save all the parameters from the trained model
* New_Method_Code: use the processed features to train model
* New_Results: the result from the model of New_Method_Code
* Results: the result from the model of Method_Code
