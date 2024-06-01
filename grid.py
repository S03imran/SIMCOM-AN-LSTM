import pandas as pd
import numpy as np
import networkx as nx
import random
from comm_dyn import features_comm_opti
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from elp_all_link_pred_algo import normalize

if __name__ == '__main__':
    def grid_search_lstm(param_grid, X_train, y_train, X_test, y_test):
        best_score = -np.inf
        best_params = None

        # Iterate over all combinations of hyperparameters
        for params in ParameterGrid(param_grid):
            print("Trying parameters:", params)

            # Define LSTM model
            model = Sequential()
            model.add(LSTM(params['lstm_units'], input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(Dense(params['dense_units'], activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            # Train LSTM model
            model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)

            # Evaluate LSTM model
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

            # Update best parameters if the current combination performs better
            if accuracy > best_score:
                best_score = accuracy
                best_params = params

        print("Best parameters found:", best_params)
        print("Best accuracy found:", best_score)

    # Assuming you have X_train, y_train, X_test, and y_test data prepared

    G = nx.Graph()
    with open('fb-forum.txt', 'r') as file:
        for line in file:
            edge = line.strip().split()
            source = int(edge[0])
            target = int(edge[1])
            G.add_edge(source, target)

    graph_original = G
    adj_original = nx.adjacency_matrix(graph_original).todense()
    edges = np.array(list(graph_original.edges))
    nodes = list(range(len(adj_original)))
    np.random.shuffle(edges)
    edges_original = edges
    ratio = 0.8
    edges_train = np.array(edges_original, copy=True)
    np.random.shuffle(edges_train)
    edges_train = random.sample(list(edges_train), int(ratio * (len(edges_train))))
    graph_train = nx.Graph()
    graph_train.add_nodes_from(nodes)
    graph_train.add_edges_from(edges_train)
    adj_train = nx.adjacency_matrix(graph_train).todense()
    graph_test = nx.Graph()
    graph_test.add_nodes_from(nodes)
    graph_test.add_edges_from(edges_original)
    graph_test.remove_edges_from(edges_train)
    t = 'fb-forum'
    m = 5
    prob_mat = features_comm_opti(m, t)

    prob_mat = normalize(prob_mat)

    # Define sequence length for LSTM
    seq_length = 10  # You may adjust this as needed

    # Reshape prob_mat into sequences for LSTM
    sequences = []
    labels = []
    for i in range(0, prob_mat.shape[0] - seq_length + 1):
        sequences.append(prob_mat[i:i + seq_length, :])
        # Check if edge exists between the nodes represented by the sequence
        if graph_train.has_edge(i, i + seq_length - 1):
            labels.append(1)
        else:
            labels.append(0)
    sequences = np.array(sequences)
    labels = np.array(labels)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

    param_grid = {
        'seq_length': [5, 10],  # sequence length for LSTM
        'lstm_units': [32, 64],  # number of units in LSTM layer
        'dense_units': [32, 64],  # number of units in Dense layer
        'batch_size': [32, 64],  # batch size for training
        'epochs': [5, 10, 15]  # number of training epochs
    }

    grid_search_lstm(param_grid, X_train, y_train, X_test, y_test)
