import networkx as nx
import numpy as np
import time
import math
import random
import linkpred
import node2vec
from gensim.models import Word2Vec
import math
from scipy.sparse.csgraph import laplacian
from sklearn.cluster import KMeans
import os

var_dict = {}
TW = {}
nodes_per_label = {}


def normalize (n):
    max = 0
    max_n = 0
    for i in range(len(n)):
        for j in range(len(n[0])) :
            if n[i][j] >= 0 :
                if max < n[i][j] : max = n[i][j]
            else :
                if max_n > n[i][j] : max_n = n[i][j]
    if max < max_n*-1 : max = max_n*-1
    for i in n:
        if max != 0 :
            for j in range(len(i)):
                i[j] = i[j] / max
    return n


def aa (adj) :
    print("running new aa")
    G = nx.Graph(adj)
    common = np.zeros(shape = (len(adj), len(adj)))
    '''result = nx.adamic_adar_index(G)
    for node1, node2, value in result :
        common[node1][node2] = value
        common[node2][node1] = value'''
    for node1 in G :
        for node2 in G :
            common_neighbours_all = nx.common_neighbors(G, node1, node2)
            for common_neighbour in common_neighbours_all:
                if G.degree(common_neighbour) != 0 and G.degree(common_neighbour) != 1:
                    common[node1][node2] = common[node1][node2] + 1 / math.log(G.degree(common_neighbour))
    return common


def car (adj) :
    Graph = nx.Graph(adj)
    triangles = nx.triangles(Graph)
    common = np.zeros(shape = (len(adj), len(adj)))
    for i in range(len(adj)):
        for j in range(len(adj)):
            common_n = nx.common_neighbors(Graph,i,j)
            cn = len(sorted(nx.common_neighbors(Graph,i,j)))
            edges = 0
            if cn > 0 :
                for m in common_n:
                    for n in common_n:
                        edges += adj[m,n]
                        if adj[m,n] !=0 and adj[m,n] != 1 :
                            print (adj[m,n])
            edges = edges /2
            if edges > 0 :
                common[i][j] += cn * (edges - 1)
    return common


def cclp (adj) :
    Graph = nx.Graph(adj)
    triangles = nx.triangles(Graph)
    common = np.zeros(shape = (len(adj), len(adj)))
    for i in range(len(adj)):
        for j in range(len(adj)):
            # common[i][j] = len(sorted(nx.common_neighbors(Graph,i,j)))
            common_n = nx.common_neighbors(Graph,i,j)
            if len(sorted(nx.common_neighbors(Graph,i,j))) > 0 :
                for k in common_n:
                    if Graph.degree(k)>1:
                        common[i][j] += triangles[k]/(Graph.degree(k)*(Graph.degree(k)-1)/2)
    return common


def cclp2 (adj) :
    Graph = nx.Graph(adj)
    triangles = nx.triangles(Graph)
    common = np.zeros(shape = (len(adj), len(adj)))
    for i in range(len(adj)):
        for j in range(len(adj)):
            common_n = nx.common_neighbors(Graph,i,j)
            for m in common_n:
                common_n2 = nx.common_neighbors(Graph,i,m)
                if len(sorted(nx.common_neighbors(Graph,i,m))) > 0 :
                    for k in common_n2:
                        if Graph.degree(k)>1:
                            common[i][j] += triangles[k]/(Graph.degree(k)*(Graph.degree(k)-1)/2)
    return common


def cn (adj) :
    Graph = nx.Graph(adj)
    common = np.zeros(shape = (len(adj), len(adj)))
    for i in range(len(adj)):
        for j in range(len(adj)):
            common[i][j] = len(sorted(nx.common_neighbors(Graph,i,j)))
    # similarity_mat = similarity(common)
    return common


def jc (adj) :
    Graph = nx.Graph(adj)
    common = np.zeros(shape = (len(adj), len(adj)))
    for i in range(len(adj)):
        for j in range(len(adj)):
            n1 = Graph.neighbors(i)
            n2 = Graph.neighbors(j)
            length = len(set().union(n1,n2))
            if length > 0 :
                common[i][j] = len(sorted(nx.common_neighbors(Graph,i,j)))/length
    return common

def shortest_path_lengths(adj):
    Graph = nx.Graph(adj)
    num_nodes = len(adj)
    shortest_paths = np.zeros(shape=(num_nodes, num_nodes), dtype=float)
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # Avoid re-computation and diagonal
            try:
                shortest_path_length = nx.shortest_path_length(Graph, source=i, target=j)
                shortest_paths[i][j] = shortest_paths[j][i] = shortest_path_length
            except nx.NetworkXNoPath:
                # If there's no path between nodes i and j, set the distance to infinity
                shortest_paths[i][j] = shortest_paths[j][i] = float('inf')
    
    return shortest_paths

def cosp(adj):
    Graph = nx.Graph(adj)
    num_nodes = len(adj)
    cos_similarity = np.zeros(shape=(num_nodes, num_nodes), dtype=float)
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # Avoid re-computation and diagonal
            common_neighbors = len(list(nx.common_neighbors(Graph, i, j)))
            degree_i = len(list(Graph.neighbors(i)))
            degree_j = len(list(Graph.neighbors(j)))
            
            if degree_i != 0 and degree_j != 0:
                cosine_sim = common_neighbors / np.sqrt(degree_i * degree_j)
                cos_similarity[i][j] = cos_similarity[j][i] = cosine_sim
    
    return cos_similarity


def nlc (adj) :
    Graph = nx.Graph(adj)
    triangles = nx.triangles(Graph)
    common = np.zeros(shape = (len(adj), len(adj)))
    for i in range(len(Graph)):
        for j in range(len(Graph)):
            if i != j :
                common_n = nx.common_neighbors(Graph,i,j)
                if len(sorted(nx.common_neighbors(Graph,i,j))) > 0 :
                    for k in common_n:
                        cnxz = len(sorted(nx.common_neighbors(Graph,i,k)))
                        cnyz = len(sorted(nx.common_neighbors(Graph,j,k)))
                        kz = Graph.degree(k)
                        cz = triangles[k]/(kz*(kz-1)/2)
                        common[i][j] += (cnxz*cz/(kz-1)) + (cnyz*cz/(kz-1))
    return common


def pa(adj):
    g_train = nx.Graph(adj)
    common = np.zeros(adj.shape)

    '''for u, v, p in nx.preferential_attachment(g_train):  # (u, v) = node indices, p = Jaccard coefficient
        pa_matrix[u][v] = p
        # print (pa_matrix[u][v])
        pa_matrix[v][u] = p  # make sure it's symmetric'''

    for i in range(len(g_train)):
        for j in range(len(g_train)):
            common[i][j] = len(sorted(g_train.neighbors(i)))*len(sorted(g_train.neighbors(j)))

    # Normalize
    # print (pa_matrix)
    #pa_matrix = pa_matrix / pa_matrix.max()
    # print (pa_matrix)
    return common


def ra (adj) :
    G = nx.Graph(adj)
    common = np.zeros(shape = (len(adj), len(adj)))
    '''result = nx.resource_allocation_index(G)
    for node1, node2, value in result:
        common[node1][node2] = value
        common[node2][node1] = value'''
    for node1 in G:
        for node2 in G:
            common_neighbours_all = nx.common_neighbors(G, node1, node2)
            for common_neighbour in common_neighbours_all:
                if G.degree(common_neighbour) != 0 :
                    common[node1][node2] = common[node1][node2] + 1 / G.degree(common_neighbour)
    return common


def rooted_pagerank_linkpred(adj):
    scores = []
    G = nx.Graph(adj)
    common = np.zeros(shape=(len(adj), len(adj)))
    for node1 in G :
        rooted_pagerank = linkpred.network.rooted_pagerank(G, root=node1)
        #print(str(rooted_pagerank))
        for node2 in range(len(adj)) :
            common[node1][node2] = rooted_pagerank[node2]
    '''for node1 in range(len(adj)):
        for node2 in range(len(adj)):
            if common[node1][node2] != common[node2][node1] : print(str(common[node1][node2])+"-"+str(common[node2][node1]))'''
    return common


def max_comm_label (node):
    global var_dict
    G = var_dict['graph']
    all_labels = set()
    #print("initially for node "+str(node)+" label is "+str(var_dict[node]))
    for node_neighbour in G.neighbors(node):
        all_labels.add(var_dict[node_neighbour])
    prob_actual = 1
    label_actual = var_dict[node]
    for label in all_labels:
        #print("for label "+str(label))
        prob_new = 1
        for node_chk in G.neighbors(node):
            #print("u is-"+str(u)+" v is-"+str(v))
            if var_dict[node_chk] == label :
                #print("prob_new = "+str(prob_new)+" edge weight "+str(G[node][node_chk]['weight']))
                chk = 0
                if G.has_edge(node,node_chk) :
                    chk = G[node][node_chk]['weight']
                if var_dict['influence'][node][node_chk] == 1 :
                    #print("influence and edge weight true for "+str(node)+"-"+str(node_chk))
                    prob_new = prob_new * (1 - chk)
        if prob_new < prob_actual :
            prob_actual = prob_new
            label_actual = label
            var_dict[node] = label
    #print("after max_comm_label for node " + str(node) + " label is " + str(var_dict[node]))
    return label_actual


def detachability (label) :
    global var_dict
    G = var_dict['graph']
    internal = 0
    external = 0
    DZ = 0
    # node and node neighbour only taken into account
    for node in G :
        if var_dict[node] == label :
            for node_neighbour in G.neighbors(node) :
                if var_dict[node_neighbour] == label :
                    internal = internal + G[node][node_neighbour]['weight']
                else :
                    external = external + G[node][node_neighbour]['weight']
    if internal + external != 0 :
        DZ = internal / (internal + external)
    return DZ


def clustering_ss(graph):
    print("inside clustering_ss")
    global var_dict
    global TW
    global nodes_per_label
    tao = var_dict['tao']
    theta = var_dict['theta']
    adj = nx.adjacency_matrix(graph).todense()
    G = var_dict['graph'].copy()
    prev = np.zeros((len(adj), len(adj)))
    print("No. of edges - "+str(len(G.edges())))
    i = 1
    A = []
    #giving node labels their number by default
    A = np.zeros((len(adj), len(adj)))
    var_dict['influence'] = A
    # making default value of A as -1, non edge
    for i in range(len(adj)):
        for j in range(len(adj)):
            A[i][j] = -1
    for node in G:
        var_dict[node] = node
        '''CL = node
        z = CL
        CZ = node'''
        for node_neighbour in G.neighbors(node):
            if G.edges[node,node_neighbour]['weight'] > random.uniform(0,1) : A[node][node_neighbour] = 1
            else : A[node][node_neighbour] = 0
        #print("after making Isinfluence matrix")
    #checking for default labels
    #for node in G :
        #print(str(node)+" has default label "+str(var_dict[node]))
    i = 0
    while i <= tao :
        print("for i = "+str(i))
        for node in G:
            #print("for node = "+str(node))
            old_label = var_dict[node]
            new_label = max_comm_label(node)
            var_dict[node] = new_label
            #if old_label != new_label :
                #print("node = "+str(node)+" has new label "+str(var_dict[node])+" old label was "+str(old_label))
        i = i + 1
        total_labels = set()
        for node in G:
            total_labels.add(var_dict[node])
            # print(str(node)+"has final label"+str(var_dict[node]))
        print("number of labels left "+str(len(total_labels)))
        #for label in total_labels:
            #print("final labels " + str(label))
    all_labels = set()
    for node in G:
        all_labels.add(var_dict[node])
    for label in all_labels:
        DZ1 = detachability(label)
        #print("label-"+str(label)+" has detachability "+str(DZ1))
        if DZ1 < theta :
            #print("inside detachability less than threshold")
            just_neighbour = set()
            outer = set()
            TW = np.zeros(len(adj))
            for node in G:
                if var_dict[node] == label :
                    for node_neighbour in G.neighbors(node):
                        just_neighbour.add(node_neighbour)
                        if var_dict[node_neighbour] != label :
                            TW[node_neighbour] += 1
                else :
                    outer.add(node)
                '''for node_neighbour in G.neighbors(node) :
                    if TW[node_neighbour] != 0:
                        print("neighbourhood count of node "+str(node_neighbour)+" is "+str(TW[node_neighbour]))'''
            NE = just_neighbour.intersection(outer)
            c_max = 0
            for node_inter in NE :
                if TW[node_inter] > c_max :
                    c_max = TW[node_inter]
                    #print("for label "+str(label)+" c max is "+str(c_max))
            NS = set()
            for node_inter in NE:
                if TW[node_inter] == c_max :
                    NS.add(node_inter)
            '''print("for label-" + str(label) + " c_max is " + str(c_max))
            print("elements in NE - " + str(len(NE)))
            print("elements in NS - " + str(len(NS)))'''
            CS_label = set()
            for node in NS :
                CS_label.add(var_dict[node])
            MID = -99999
            new_label = label
            for label_other in CS_label :
                factor2 = detachability(label_other)
                #to change label_other to label find all node with label store and change
                to_be_changed = set()
                for node in G :
                    if var_dict[node] == label_other :
                        to_be_changed.add(node)
                        var_dict[node] = label
                factor1 = detachability(label)
                #change back to label_other
                for node in to_be_changed :
                    var_dict[node] = label_other
                TID = factor1 - factor2
                if TID > MID :
                    #print("label change criteria met")
                    MID = TID
                    new_label = label_other
            for node in G :
                if var_dict[node] == label :
                    #if label != new_label :
                        #print("changed label of node "+str(node))
                    var_dict[node] = new_label

    total_labels = set()
    for node in G :
        total_labels.add(var_dict[node])
        #print(str(node)+"has final label"+str(var_dict[node]))
    print("number of labels left finally " + str(len(total_labels)))
    for label in total_labels :
        #print("final labels "+str(label))
        count = 0
        nodes_per_label[label] = count
        for node in G :
            if var_dict[node] == label :
                count += 1
        nodes_per_label[label] = count
        #print("for label "+str(label)+" number of nodes "+str(nodes_per_label[label])+" with count "+str(count))
    cluster_matrix = np.zeros((len(adj), len(adj)))
    no_of_nodes = G.number_of_nodes()
    for i in range(len(adj)):
        for j in range(len(adj)) :
            if var_dict[i] == var_dict[j] :
                cluster_matrix[i][j] = int(nodes_per_label[var_dict[i]]) / no_of_nodes
            else :
                cluster_matrix[i][j] = int(nodes_per_label[var_dict[i]]) / no_of_nodes*-1
            #print("for i = "+str(i)+" for j = "+str(j)+" cluster matrix = "+str(cluster_matrix[i][j]))
    var_dict['cluster_matrix_check'] = normalize(cluster_matrix)
    return cluster_matrix


def clp_id_main (adj,tao,theta) :
    G = nx.Graph(adj)
    var_dict['graph'] = G
    var_dict['tao'] = tao
    var_dict['theta'] = theta
    for (u, v) in G.edges():
        value = random.uniform(0, 1)
        G.edges[u, v]['weight'] = value
        #G.add_edge(v, u)
        #G.edges[v, u]['weight'] = value
        # print("for edge "+str(u)+"-"+str(v)+" assigned weight is "+str(G.edges[u,v]['weight']))
    '''for i in range(len(adj)):
        for j in range(len(adj)):
            if G.has_edge(i,j):
                if G.edges[i, j]['weight'] != G.edges[j, i]['weight'] :
                    print("edge problem")'''
    #check diagonal of adjacency matrix
    cluster_matrix = clustering_ss(G)
    var_dict['cluster_matrix'] = cluster_matrix
    similarity_matrix = np.zeros((len(adj), len(adj)))
    overall_similarity_matrix = np.zeros((len(adj), len(adj)))
    '''for i in range(len(adj)) :
        for j in range(len(adj)) :
            if cluster_matrix[i][j] != var_dict['cluster_matrix_check'][i][j] :
                print("faulty")'''
    print("making similarity matrix")
    for node1 in G :
        for node2 in G :
            similarity_matrix[node1][node2] = 1
            common_neighbour_factor = 1
            for node_neighbour in G.neighbors(node1) :
                if G.has_edge(node2,node_neighbour) :
                    common_neighbour_factor = common_neighbour_factor * (1 - G.edges[node2, node_neighbour]['weight'])
            neighbour_factor = 0
            if G.has_edge(node1, node2): neighbour_factor = G.edges[node1, node2]['weight']
            similarity_matrix[node1][node2] = 1 - common_neighbour_factor + neighbour_factor
    similarity_matrix = normalize(similarity_matrix)
    var_dict['similarity_matrix'] = similarity_matrix
    print("making overall similarity matrix")
    for i in range(len(adj)) :
        for j in range(len(adj)) :
            overall_similarity_matrix[i][j] = similarity_matrix[i][j]*cluster_matrix[i][j]
    var_dict['overall_similarity_matrix'] = overall_similarity_matrix
    link_pred = np.zeros((len(adj), len(adj)))
    print("making link prediction matrix")
    for node1 in G :
        for node2 in G :
            node_neighbour_common = nx.common_neighbors(G,node1,node2)
            for common_node in node_neighbour_common:
                link_pred[node1][node2] += overall_similarity_matrix[node1][common_node] + overall_similarity_matrix[common_node][node2]
    print("returning link prediction matrix")
    return link_pred


def elp (adj) :
    print("running new ego")
    G = nx.Graph(adj)
    common = np.zeros(shape = (len(adj), len(adj)))
    ego_str = np.zeros(shape = (len(adj), len(adj)))
    for node1 in G:
        for node2 in G:
            if G.has_edge(node1,node2) == 0 :
                ego_str[node1][node2] = -1
    triangles = nx.triangles(G)
    for node in G:
        neighbors = G.neighbors(node)
        #first level for direct neighoburs, increase ego strength by 3
        for single in neighbors :
            ego_str[node][single] += 32
        #2nd level where edges between neighbours of node considered
        #print("triangles for node "+str(node)+" is = "+str(triangles[node]))
        neighbors1 = G.neighbors(node)
        for neighbor1 in neighbors1 :
            neighbors2 = G.neighbors(node)
            for neighbor2 in neighbors2 :
                if G.has_edge(neighbor1,neighbor2)  :
                    #print("change 2nd level")
                    ego_str[neighbor1][neighbor2] += 16
        #3rd level where 2 hop edges wrt node are considered
        neighbors = G.neighbors(node)
        set_2hop = set()
        for single in neighbors :
            neighbors_2hop = G.neighbors(single)
            for neighbor_2hop in neighbors_2hop :
                if neighbor_2hop != node and not G.has_edge(neighbor_2hop,node):
                    set_2hop.add(neighbor_2hop)
                    #print("change 3rd level")
                    ego_str[single][neighbor_2hop] += 8
                    neighbors_3hop = G.neighbors(neighbor_2hop)
                    for far_single in neighbors_3hop :
                        if far_single != single and not G.has_edge(far_single,single):
                            ego_str[neighbor_2hop][far_single] += 2
        for node1 in set_2hop :
            for node2 in set_2hop :
                if node1 != node2 and G.has_edge(node1,node2):
                    ego_str[node1][node2] += 4

    #print("ego strength after counting")
    '''for node1 in G:
        for node2 in G:
            if ego_str[node1][node2] != -1 and ego_str[node1][node2] != 0:
                print(ego_str[node1][node2])'''
    no_nodes = len(adj)
    max = 0
    for node1 in G :
        for node2 in G :
            if max < ego_str[node1][node2]: max = ego_str[node1][node2]
            ego_str[node1][node2] = ego_str[node1][node2]/no_nodes
    print("max ego strength is "+str(max))
    '''for node1 in G:
        for node2 in G:
            ego_str[node1][node2] = ego_str[node1][node2] / max'''
    for node1 in G :
        for node2 in G :
            if node1 <= node2 :
                if G.has_edge(node1,node2) :
                    common[node1][node2] = ego_str[node1][node2]
                else :
                    common_neighbors = nx.common_neighbors(G,node1,node2)
                    common[node1][node2] = 0
                    for single in common_neighbors :
                        #numerator = ego_str[node1][single]*ego_str[single][node2]
                        numerator = ego_str[node1][single]+ego_str[single][node2]
                        denominator = 0
                        for neighbor in G.neighbors(single) :
                            denominator += ego_str[single][neighbor]
                        #denominator = denominator**2
                        denominator = denominator
                        common[node1][node2] += numerator/denominator
            else :
                common[node1][node2] = common[node2][node1]
    return common
