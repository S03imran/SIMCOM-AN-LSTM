import optuna
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import random

from comm_dyn import features_comm_opti  # Your community feature extraction function
from elp_all_link_pred_algo import normalize  # Your normalizer

def generate_sequences(feature_matrix, graph_train, seq_length):
    sequences = []
    labels = []
    n_samples = feature_matrix.shape[0]
    for i in range(n_samples - seq_length + 1):
        sequences.append(feature_matrix[i:i + seq_length])  # shape: (seq_length, feature_dim)
        # Labeling logic: For example, consider edge existence between first and last indices as label
        x, y = i, i + seq_length - 1
        labels.append(1 if graph_train.has_edge(x, y) else 0)
    return np.array(sequences), np.array(labels)

def objective(trial):
    lstm_units = trial.suggest_categorical('lstm_units', [32, 64])
    dense_units = trial.suggest_categorical('dense_units', [32, 64])
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    epochs = trial.suggest_categorical('epochs', [5, 10, 15])
    seq_length = trial.suggest_categorical('seq_length', [5, 10])

    sequences, labels = generate_sequences(prob_mat, graph_train, seq_length)

    if len(sequences) == 0:
        return 1.0  # Penalize trials with insufficient sequences

    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=0.2, random_state=42
    )

    model = Sequential()
    model.add(LSTM(lstm_units, input_shape=(X_train.shape[1], X_train.shape[1])))
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    return 1.0 - accuracy  # Optuna minimizes the objective

if __name__ == "__main__":
    # Load your graph
    G = nx.Graph()
    with open('fb-forum.txt', 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            source, target = map(int, parts[:2])
            G.add_edge(source, target)

    # Remove isolated nodes to avoid computation errors in community detection
    G.remove_nodes_from(list(nx.isolates(G)))

    # Use largest connected component for stability
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()

    graph_original = G
    nodes = list(graph_original.nodes)
    edges_original = np.array(list(graph_original.edges))

    # Prepare training edge sample (80%)
    np.random.shuffle(edges_original)
    train_ratio = 0.8
    num_train = int(len(edges_original) * train_ratio)
    edges_train = edges_original[:num_train]

    graph_train = nx.Graph()
    graph_train.add_nodes_from(nodes)
    graph_train.add_edges_from(edges_train)

    # Compute community feature matrix and normalize
    m = 5  # or your chosen parameter for temporal slicing
    t = 'fb-forum'  # your dataset identifier
    prob_mat = features_comm_opti(m, t)
    prob_mat = normalize(prob_mat)

    # Run Bayesian optimization to tune LSTM hyperparameters
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)  # Increase trials for better results

    print("Optimal hyperparameters:", study.best_trial.params)
    print("Best accuracy:", 1.0 - study.best_trial.value)
