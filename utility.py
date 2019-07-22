import os

def get_data_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

def  get_train_data():
    return os.path.join(get_data_dir(), "train.p")

def get_test_data():
    return os.path.join(get_data_dir(), "test.p")

def get_val_data():
    return os.path.join(get_data_dir(), "valid.p")
