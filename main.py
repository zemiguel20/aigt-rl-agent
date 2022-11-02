#! /usr/bin/env -S python -u
from genericpath import exists
from data_save_load import load_weights_from_file, save_weights_to_file
from generate_data import generate_data
from nn import create_nn, train_nn_on_datafile


if __name__ == '__main__':

    nn = create_nn()

    if not exists("weightsfile"):
        if not exists('data.txt'):
            generate_data()
        train_nn_on_datafile(nn)
        save_weights_to_file(nn)
    
    load_weights_from_file(nn)
