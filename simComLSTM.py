import networkx as nx
import numpy as np
import pandas as pd
from keras_self_attention import SeqSelfAttention

from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, \
    precision_score, f1_score, precision_recall_curve, accuracy_score, balanced_accuracy_score

# -------------------------------------methods called-----------------------------------------------

from elp_all_link_pred_algo import aa, ra, cclp, cclp2, cn, pa, jc, car, \
    rooted_pagerank_linkpred, normalize, clp_id_main, elp, nlc

from comm_dyn import features_comm_opti,local_path
from sklearn.model_selection import train_test_split

import time
import random
from xlwt import Workbook
import xlrd
import datetime
import os
import shutil
import subprocess
import sys
from psutil import virtual_memory
from scipy.stats import friedmanchisquare
import scikit_posthocs as sci_post


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


from networkx import read_gml

if __name__ == '__main__':
    starttime_full = time.time()
    var_dict_main = {}

    def auprgraph_all (adj,file_name,algo):
        #print("for algo - "+str(algo))
        file_write_name = './result_all_elp/result_'+algo+'/' + file_name + ".txt"
        os.makedirs(os.path.dirname(file_write_name), exist_ok=True)
        starttime_aup = time.time()
        ratio = []
        aupr = []
        recall = []
        auc = []
        avg_prec = []
        acc_score = []
        bal_acc_score = []
        f1 = []
        prec = []
        G = nx.Graph(adj)
        G.remove_edges_from(nx.selfloop_edges(G))
        print("nodes - " + str(len(adj)) + " edges - " + str(G.number_of_edges()) + " name - " + str(file_name))
        for i in [0.7,0.8,0.9]: # range is the fraction of edge values included in the graph
            print("nodes - " + str(len(adj)) + " edges - " + str(G.number_of_edges()) + " name - " + str(file_name))
            print ("For ratio : " , i-1)
            if algo in ["cn","aa","jc","pa","cclp","clp_id_main","elp","nlc","cosp","mfi","l3","act","cclp2","car","lp"]:
                print("algo - "+algo)
                avg_array = avg_seq_all(G, file_name, i, algo)
            elif algo in ["feature_comm_opti"]:
                avg_array = avg_seq(G, file_name, i, algo)
            aupr.append(avg_array[0])
            #recall.append(avg_array[1])
            auc.append(avg_array[1])
            avg_prec.append(avg_array[2])
            #acc_score.append(avg_array[4])
            #bal_acc_score.append(avg_array[5])
            #f1.append(avg_array[6])
            #prec.append(avg_array[7])
            ratio.append(i-1)
        print("Ratio:-", ratio)
        print("AUPR:-",aupr)
        #print("Recall:-",recall)
        print("AUC:-",auc)
        print("Avg Precision:-",avg_prec)
        #print("Accuracy Score:-",acc_score)
        #print("Balanced Accuracy Score:-", bal_acc_score)
        #print("F1 Score:-", f1)
        #print("Precision Score:-", prec)
        endtime_aup = time.time()
        print('That aup took {} seconds'.format(endtime_aup - starttime_aup))

        # Workbook is created
        wb = Workbook()
        # add_sheet is used to create sheet.
        sheet1 = wb.add_sheet('Sheet 1', cell_overwrite_ok=True)
        sheet1.write(0, 0, 'Ratio')
        sheet1.write(0, 1, 'AUPR')
        #sheet1.write(0, 2, 'RECALL')
        sheet1.write(0, 2, 'AUC')
        sheet1.write(0, 3, 'AVG PRECISION')
        #sheet1.write(0, 5, 'ACCURACY SCORE')
        #sheet1.write(0, 6, 'BAL ACCURACY SCORE')
        #sheet1.write(0, 7, 'F1 MEASURE')
        #sheet1.write(0, 8, 'PRECISION')
        for i in range(3):
            sheet1.write(3 - i, 0, ratio[i]*-1)
            sheet1.write(3 - i, 1, aupr[i])
            #sheet1.write(5 - i, 2, recall[i])
            sheet1.write(3 - i, 2, auc[i])
            sheet1.write(3 - i, 3, avg_prec[i])
            #sheet1.write(5 - i, 5, acc_score[i])
            #sheet1.write(5 - i, 6, bal_acc_score[i])
            #sheet1.write(5 - i, 7, f1[i])
            #sheet1.write(5 - i, 8, prec[i])

        wb.save('./result_all_elp/result_'+algo+'/' + file_name + ".xls")

        currentDT = datetime.datetime.now()
        print(str(currentDT))

        file_all = open('./result_all_elp/current_all.txt','a')
        text_final = "full algo = "+algo+" file name = "+file_name+" time = "+\
                     str((endtime_aup - starttime_aup))+" date_time = "+str(currentDT)+"\n"
        file_all.write(text_final)
        print(text_final)
        file_all.close()

        return aupr,ratio,recall,auc,avg_prec,acc_score,bal_acc_score,f1,prec
    
    

    '''def avg_seq(g, file_name, ratio, algo):
        start_time_ratio = time.time()
        auc = 0
        avg_prec = 0
        avg_acc = 0
        loop = 50
        ratio = round(ratio, 1)
        graph_original = g
        print("avg sequential called for algo - " + str(algo) + " ratio - " + str(ratio))

        for single_iter in range(loop):
            print("old number of edges - " + str(len(graph_original.edges)) + " for ratio - " + str(ratio))
            adj_original = nx.adjacency_matrix(graph_original).todense()
            starttime = time.time()
            edges = np.array(list(graph_original.edges))
            nodes = list(range(len(adj_original)))
            np.random.shuffle(edges)
            edges_original = edges
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
            print("new number of edges - " + str(len(graph_train.edges)) + " for ratio - " + str(ratio))

            if algo == 'feature_comm_opti':
                m = 5
                t = "mit"
                prob_mat = features_comm_opti(m, t)

            prob_mat = normalize(prob_mat)

            # Define sequence length for LSTM
            seq_length = 5  # You may adjust this as needed

            # Reshape prob_mat into sequences for LSTM
            sequences = []
            labels = []
            for i in range(0, prob_mat.shape[0] - seq_length + 1):
                sequences.append(prob_mat[i:i+seq_length, :])
                # Check if edge exists between the nodes represented by the sequence
                if graph_train.has_edge(i, i+seq_length-1):
                    labels.append(1)
                else:
                    labels.append(0)

            sequences = np.array(sequences)
            labels = np.array(labels)

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

            # Reshape y_train to match the shape of y_pred
            y_train = np.expand_dims(y_train, axis=-1)

            # Define LSTM model with attention layer
            model = Sequential()
            model.add(LSTM(32, input_shape=(seq_length, X_train.shape[2]), return_sequences=True))
            model.add(SeqSelfAttention(attention_activation='sigmoid'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            # Train LSTM model
            model.fit(X_train, y_train, epochs=5, batch_size=32)

            # Evaluate LSTM model
            y_pred = model.predict(X_test)

            # Calculate AUC using the last timestep's prediction
            y_pred_last = y_pred[:, -1, :]  # Extract predictions from the last timestep
            auc += roc_auc_score(y_test, y_pred_last)

            # Calculate average precision
            avg_prec += average_precision_score(y_test, y_pred_last)

            # Calculate accuracy
            y_pred_binary = (y_pred_last > 0.5).astype(int)
            avg_acc += accuracy_score(y_test, y_pred_binary)

            endtime = time.time()
            currentDT = datetime.datetime.now()

            file_all = open('./result_all_elp/current.txt', 'a')
            text_inside_single = "single algo = " + algo + " file name = " + file_name + \
                                 " ratio = " + str(ratio) + " time = " + \
                                 str(endtime - starttime) + " sec date_time = " + str(currentDT) + "\n"
            file_all.write(text_inside_single)
            print(text_inside_single)
            file_all.close()
            print("Shapes:")
            print("X_train shape:", X_train.shape)
            print("y_train shape:", y_train.shape)
            print("X_test shape:", X_test.shape)
            print("y_test shape:", y_test.shape)

            print("Unique values in y_train:", np.unique(y_train))
            print("Unique values in y_test:", np.unique(y_test))

        # Calculate average AUC, average precision, and average accuracy
        avg_auc = auc / loop
        avg_avg_prec = avg_prec / loop
        avg_avg_acc = avg_acc / loop

        return avg_auc, avg_avg_prec, avg_avg_acc'''


    def avg_seq(g, file_name, ratio, algo):
        start_time_ratio = time.time()
        auc = 0
        avg_prec = 0
        avg_acc = 0
        loop = 50
        ratio = round(ratio, 1)
        graph_original = g
        print("avg sequential called for algo - " + str(algo) + " ratio - " + str(ratio))

        for single_iter in range(loop):
            print("old number of edges - " + str(len(graph_original.edges)) + " for ratio - " + str(ratio))
            adj_original = nx.adjacency_matrix(graph_original).todense()
            starttime = time.time()
            edges = np.array(list(graph_original.edges))
            nodes = list(range(len(adj_original)))
            np.random.shuffle(edges)
            edges_original = edges
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
            print("new number of edges - " + str(len(graph_train.edges)) + " for ratio - " + str(ratio))

            if algo == 'feature_comm_opti':
                m = 2
                t = "Eu-core"
                prob_mat = features_comm_opti(m, t)

            prob_mat = normalize(prob_mat)

            # Define sequence length for LSTM
            seq_length = 5  # You may adjust this as needed

            # Reshape prob_mat into sequences for LSTM
            sequences = []
            labels = []
            for i in range(0, prob_mat.shape[0] - seq_length + 1):
                sequences.append(prob_mat[i:i+seq_length, :])
                # Check if edge exists between the nodes represented by the sequence
                if graph_train.has_edge(i, i+seq_length-1):
                    labels.append(1)
                else:
                    labels.append(0)

            sequences = np.array(sequences)
            labels = np.array(labels)

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

            # Reshape y_train to match the shape of y_pred
            y_train = np.expand_dims(y_train, axis=-1)

            # Define LSTM model with attention layer
            model = Sequential()
            model.add(LSTM(32, input_shape=(seq_length, X_train.shape[2]), return_sequences=True))
            model.add(SeqSelfAttention(attention_activation='sigmoid'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            # Train LSTM model
            model.fit(X_train, y_train, epochs=5, batch_size=32)

            # Evaluate LSTM model
            y_pred = model.predict(X_test)

            # Calculate AUC using the last timestep's prediction
            y_pred_last = y_pred[:, -1, :]  # Extract predictions from the last timestep
            auc += roc_auc_score(y_test, y_pred_last)

            # Calculate average precision
            avg_prec += average_precision_score(y_test, y_pred_last)

            # Calculate accuracy
            y_pred_binary = (y_pred_last > 0.5).astype(int)
            avg_acc += accuracy_score(y_test, y_pred_binary)

            endtime = time.time()
            currentDT = datetime.datetime.now()

            file_all = open('./result_all_elp/current.txt', 'a')
            text_inside_single = "single algo = " + algo + " file name = " + file_name + \
                                 " ratio = " + str(ratio) + " time = " + \
                                 str(endtime - starttime) + " sec date_time = " + str(currentDT) + "\n"
            file_all.write(text_inside_single)
            print(text_inside_single)
            file_all.close()
        # Calculate average AUC, average precision, and average accuracy
        avg_auc = auc / loop
        avg_avg_prec = avg_prec / loop
        avg_avg_acc = avg_acc / loop

        return avg_auc, avg_avg_prec, avg_avg_acc

    def avg_seq_all(g, file_name, ratio, algo) :

        start_time_ratio = time.time()
        aupr = 0
        recall = 0
        auc = 0
        avg_prec = 0
        acc_score = 0
        bal_acc_score = 0
        f1 = 0
        prec = 0
        loop = 50
        ratio = round(ratio, 1)
        graph_original = g
        print("avg sequential called for algo - " + str(algo) + " ratio - " + str(ratio))

        for single_iter in range(loop):

            print("old number of edges - " + str(len(graph_original.edges)) + " for ratio - " + str(ratio))
            # making original graph adjacency matrix
            adj_original = nx.adjacency_matrix(graph_original).todense()
            starttime = time.time()
            # finding edges and nodes of original graph
            edges = np.array(list(graph_original.edges))
            nodes = list(range(len(adj_original)))
            np.random.shuffle(edges)
            edges_original = edges
            edges_train = np.array(edges_original, copy=True)
            np.random.shuffle(edges_train)
            edges_train = random.sample(list(edges_train), int(ratio * (len(edges_train))))
            # finding training set of edges according to ratio
            graph_train = nx.Graph()
            # making graph based on the training edges
            graph_train.add_nodes_from(nodes)
            graph_train.add_edges_from(edges_train)
            adj_train = nx.adjacency_matrix(graph_train).todense()
            # making test graph by removing train edges from original
            graph_test = nx.Graph()
            graph_test.add_nodes_from(nodes)
            graph_test.add_edges_from(edges_original)
            graph_test.remove_edges_from(edges_train)
            print("new number of edges - " + str(len(graph_train.edges)) + " for ratio - " + str(ratio))

            # sending training graph for probability matrix prediction
            if algo == 'cn': prob_mat = cn(adj_train)
            if algo == 'ra': prob_mat = ra(adj_train)
            if algo == 'car': prob_mat = car(adj_train)
            if algo == 'cclp': prob_mat = cclp(adj_train)
            if algo == 'jc': prob_mat = jc(adj_train)
            if algo == 'pa': prob_mat = pa(adj_train)
            if algo == 'clp_id_main': prob_mat = clp_id_main(adj_train)
            if algo == 'aa': prob_mat = aa(adj_train)
            if algo == 'elp': prob_mat = elp(adj_train)
            if algo == 'clp_id': prob_mat = clp_id_main(adj_train, 25, 1.0)
            if algo == 'nlc': prob_mat = nlc(adj_train)
            if algo == 'cosp': prob_mat = cosp(adj_train)
            if algo == 'mfi' : prob_mat = mfi(adj_train) 
            if algo == 'cclp2' : prob_mat = cclp2(adj_train)  
            if algo == 'lp' : prob_mat = local_path(adj_train) 


            prob_mat = normalize(prob_mat)
            endtime = time.time()
            print('{} for probability matrix prediction'.format(endtime - starttime))

            # making adcancecy test from testing graph
            adj_test = nx.adjacency_matrix(graph_test).todense()
            # making new arrays to pass to function
            array_true = []
            array_pred = []
            for i in range(len(adj_original)):
                for j in range(len(adj_original)):
                    if not graph_original.has_edge(i, j):
                        array_true.append(0)
                        array_pred.append(prob_mat[i][j])
                    if graph_test.has_edge(i, j):
                        array_true.append(1)
                        array_pred.append(prob_mat[i][j])
            # flattening adjacency matrices
            '''pred = pred.flatten()
            adj_original = np.array(adj_original).flatten()
            adj_test = np.array(adj_test).flatten()'''
            pred = array_pred
            adj_test = array_true
            #adj_test = np.array(adj_test)
            #adj_train = np.array(adj_train)
            #X_train, X_test, y_train, y_test = train_test_split(adj_train, adj_train, test_size=0.2, random_state=42)
            #model = Sequential([
            #LSTM(50, input_shape=(adj_train.shape[1], adj_test.shape[2])),
            #Dense(adj_train.shape[2], activation='sigmoid')
            #])

            # return precision recall pairs for particular thresholds
            prec_per, recall_per, threshold_per = precision_recall_curve(adj_test, pred)
            prec_per = prec_per[::-1]
            recall_per = recall_per[::-1]
            aupr_value = np.trapz(prec_per, x=recall_per)
            auc_value = roc_auc_score(adj_test, pred)
            avg_prec_value = average_precision_score(adj_test, pred)

            test_pred_label = np.copy(pred)
            a = np.mean(test_pred_label)

            for i in range(len(pred)):
                if pred[i] < a:
                    test_pred_label[i] = 0
                else:
                    test_pred_label[i] = 1
            recall_value = recall_score(adj_test, test_pred_label)
            acc_score_value = accuracy_score(adj_test, test_pred_label)
            bal_acc_score_value = balanced_accuracy_score(adj_test, test_pred_label)
            precision_value = precision_score(adj_test, test_pred_label)
            f1_value = f1_score(adj_test, test_pred_label)

            endtime = time.time()
            print('{} for metric calculation'.format(endtime - starttime))

            currentDT = datetime.datetime.now()
            print(str(currentDT))

            file_all = open('./result_all_elp/current.txt', 'a')
            text_inside_single = "single algo = " + algo + " file name = " + file_name + \
                                 " ratio = " + str(ratio) + " time = " + \
                                 str(endtime - starttime) + " sec date_time = " + str(currentDT) + "\n"
            file_all.write(text_inside_single)
            print(text_inside_single)
            file_all.close()

            aupr += aupr_value
            recall += recall_value
            auc += auc_value
            avg_prec += avg_prec_value
            acc_score += acc_score_value
            bal_acc_score += bal_acc_score_value
            f1 += f1_value
            prec += precision_value

        currentDT = datetime.datetime.now()
        print(str(currentDT))
        end_time_ratio = time.time()
        file_all = open('./result_all_elp/current.txt', 'a')
        text_inside = "full algo = " + algo + " file name = " + file_name + \
                           " ratio = " + str(ratio) + " time = " + \
                           str(end_time_ratio - start_time_ratio) + " date_time = " \
                           + str(currentDT) + "\n"
        file_all.write(text_inside)
        file_all.close()

        return aupr / loop, auc / loop, avg_prec / loop


    def result_parser_combine(file_name_array,algo_result_all_metric):

        algo_all = algo_result_all_metric

        for file_name in file_name_array:
            file_write_name = './result_all_elp/' + file_name + "_combine.xls"
            os.makedirs(os.path.dirname(file_write_name), exist_ok=True)
            # Workbook is created
            wb_write = Workbook()
            # add_sheet is used to create sheet.
            AUPR = wb_write.add_sheet('AUPR', cell_overwrite_ok=True)
            #RECALL = wb_write.add_sheet('RECALL', cell_overwrite_ok=True)
            AUC = wb_write.add_sheet('AUC', cell_overwrite_ok=True)
            AVG_PREC = wb_write.add_sheet('AVG PREC', cell_overwrite_ok=True)
            #ACC_SCORE = wb_write.add_sheet('ACC SCORE', cell_overwrite_ok=True)
            #BAL_ACC_SCORE = wb_write.add_sheet('BAL ACC SCORE', cell_overwrite_ok=True)
            #F1_SCORE = wb_write.add_sheet('F1 SCORE', cell_overwrite_ok=True)
            #PRECISION = wb_write.add_sheet('PRECISION', cell_overwrite_ok=True)
            sheet_array = [AUPR, AUC, AVG_PREC]
            for sheet_single in sheet_array:
                sheet_single.write(0, 0, 'Ratio')
                sheet_single.write(1, 0, '0.1')
                sheet_single.write(2, 0, '0.2')
                sheet_single.write(3, 0, '0.3')
            current_algo = 1
            for algo in algo_all:
                single_algo_file = "./result_all_elp/result_" + str(algo) + '/' + file_name + ".xls"
                wb_read = xlrd.open_workbook(single_algo_file)
                main_sheet = wb_read.sheet_by_name('Sheet 1')
                for sheet_single in sheet_array:
                    sheet_single.write(0, current_algo, str(algo).upper())
                for row_read in range(3):
                    row_read += 1
                    row_write = row_read
                    for col_read in range(3):
                        col_read += 1
                        print("reading--" + file_name + " --of algo--" + algo)
                        value = float(main_sheet.cell(row_read, col_read).value)
                        value = round(value, 3)
                        sheet_no = col_read - 1
                        sheet_array[sheet_no].write(row_write, current_algo, value)
                current_algo = current_algo + 1
            wb_write.save(file_write_name)

        wb_dataset_write = Workbook()
        file_dataset_write_name = './result_all_elp/all_datasets_combine_elp.xls'
        sheet_name_array = ['AUPR', 'AUC', 'AVG PREC']
        AUPR_write = wb_dataset_write.add_sheet(sheet_name_array[0], cell_overwrite_ok=True)
        #RECALL_write = wb_dataset_write.add_sheet(sheet_name_array[1], cell_overwrite_ok=True)
        AUC_write = wb_dataset_write.add_sheet(sheet_name_array[1], cell_overwrite_ok=True)
        AVG_PREC_write = wb_dataset_write.add_sheet(sheet_name_array[2], cell_overwrite_ok=True)
        #ACC_SCORE_write = wb_dataset_write.add_sheet(sheet_name_array[4], cell_overwrite_ok=True)
        #BAL_ACC_SCORE_write = wb_dataset_write.add_sheet(sheet_name_array[5], cell_overwrite_ok=True)
        #F1_SCORE_write = wb_dataset_write.add_sheet(sheet_name_array[6], cell_overwrite_ok=True)
        #PRECISION_write = wb_dataset_write.add_sheet(sheet_name_array[7], cell_overwrite_ok=True)
        sheet_dataset_write_array = [AUPR_write, AUC_write, AVG_PREC_write]
        count = 0
        for file_name in file_name_array:
            file_read_name = './result_all_elp/' + file_name + "_combine.xls"
            wb_read = xlrd.open_workbook(file_read_name)
            AUPR = wb_read.sheet_by_name(sheet_name_array[0])
            #RECALL = wb_read.sheet_by_name(sheet_name_array[1])
            AUC = wb_read.sheet_by_name(sheet_name_array[1])
            AVG_PREC = wb_read.sheet_by_name(sheet_name_array[2])
            #ACC_SCORE = wb_read.sheet_by_name(sheet_name_array[4])
            #BAL_ACC_SCORE = wb_read.sheet_by_name(sheet_name_array[5])
            #F1_SCORE = wb_read.sheet_by_name(sheet_name_array[6])
            #PRECISION_SCORE = wb_read.sheet_by_name(sheet_name_array[7])
            sheet_read_array = [AUPR, AUC, AVG_PREC]
            write_row = file_name_array.index(file_name) + 1
            for sheet_no in range(len(sheet_read_array)):
                sheet_dataset_write_array[sheet_no].write(0, 1, 'Ratio')
                sheet_dataset_write_array[sheet_no].write(1 + count * 6, 1, '0.1')
                sheet_dataset_write_array[sheet_no].write(2 + count * 6, 1, '0.2')
                sheet_dataset_write_array[sheet_no].write(3 + count * 6, 1, '0.3')
                sheet_dataset_write_array[sheet_no].write(0, 0, 'FILE_NAME')
                sheet_dataset_write_array[sheet_no].write(1 + count * 6, 0, str(file_name))
                for ratio in range(3):
                    read_row = ratio + 1
                    write_row = ratio + 1 + count * 6
                    for algo_no in range(len(algo_all)):
                        read_col = algo_no + 1
                        write_col = algo_no + 2
                        value = sheet_read_array[sheet_no].cell(read_row, read_col).value
                        print("value read = " + str(value))
                        # sheet_dataset_write_array[sheet_no].write(count * 6, write_col, str(algo_all[algo_no]).upper())
                        sheet_dataset_write_array[sheet_no].write(0, write_col, str(algo_all[algo_no]).upper())
                        sheet_dataset_write_array[sheet_no].write(write_row, write_col, value)
            count += 1
        wb_dataset_write.save(file_dataset_write_name)


    def read_txt(filename):
        G = nx.Graph()
        with open(filename, 'r') as file:
            for line in file:
                edge = line.strip().split()
                source = int(edge[0])
                target = int(edge[1])
                G.add_edge(source, target)
        return G

    
    
    def aupgraph_control_multiple_dataset_all(file_name_array):
        file_write_name = "./datasets_dynamic/data_info/current_all.txt"
        #file_write_name = './result_all_elp/current_all.txt'
        os.makedirs(os.path.dirname(file_write_name), exist_ok=True)
        algo_ego = ['feature_comm_opti']
        for algo in algo_ego :
            if algo not in [] :
                for file_name in file_name_array:
                    ds = './datasets_dynamic/' + file_name
                    G = read_txt(ds+'.txt')
                    adj_mat_s = nx.adjacency_matrix(G)
                    n = adj_mat_s.shape[0]
                    print("nodes = "+str(n))
                    adj_mat_d = adj_mat_s.todense()
                    adj = adj_mat_d
                    auprgraph_all(adj, file_name, algo)

    #algo_result_ego = ['cn', 'aa', 'jc', 'pa', 'cclp','clp_id', 'elp','nlc']
    algo_result_ego = ['feature_comm_opti']
    

    #file_name_array = ['CollegeMsg','Eu-core','fb-forum','lkml-reply','mathoverflow','mit','radoslaw-email']
    #file_name_array = ['mit','CollegeMsg','Eu-core','fb-forum','radoslaw-email']
    file_name_array = ['Eu-core']

    aupgraph_control_multiple_dataset_all(file_name_array)
    result_parser_combine(file_name_array,algo_result_ego)

    print('That took {} seconds'.format(time.time() - starttime_full))
