Out methods are rather straightford due to limited time, by combining data_augmentation, multihead-output models and model averaging.

Two csv files are: attr_query_index.csv indicating the results of test images and Query_result.csv including the result of query.txt file


1. Keras = 2.2.0 tensorflow = 1.6.0 python = 3.5
2. Please note this project share the same parent directory with 'training/test data'.
3. Weights are saved in three models respectively in attr_model.h5, attr_modelv2.h5,attr_model3.h5.
4. Other tools for IO/pre-process/post-process are in prepare_data.py
5. Please run predict.py directly. It will directly output two independent .csv files using test images 
