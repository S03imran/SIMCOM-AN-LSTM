To run SIMCOM-AN-LSTM use
  python first.py

first.py calls elp_all_pred_algo.py which calculates the local, global and quasi-local features of the dynamic graph data

comm_dyn caluculates the community feature and returns the feature set

Feature set returned by comm_dyn is used to generate the LTM model
