import networkx as nx
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
import random

'''
Script to analyse graphs
'''

def count(graph_dir, plot = False):
    """
    Print stats about a dir of graphs
    :param graph_dir: path containing a list of nx graphs
    :param plot: if True, display the distributions
    """
    number_of_nodes = []
    number_of_cc = []
    number_of_node_per_cc = []
    for pdb in os.listdir(graph_dir):
        path = os.path.join(graph_dir, pdb)
        g = nx.read_gpickle(path)
        number_of_cc += [nx.number_connected_components(g)]
        ccs = []
        for cc in nx.connected_components(g):
            ccs.append(len(cc))
        number_of_node_per_cc += [sum(ccs) / len(ccs)]
        number_of_nodes += [nx.number_of_nodes(g)]
    avg_nn = (sum(number_of_nodes) / len(number_of_nodes))
    print('The average number of nodes is {}'.format(avg_nn))

    avg_cc = (sum(number_of_cc) / len(number_of_cc))
    print('The average number of cc is {}'.format(avg_cc))

    avg_ncc = (sum(number_of_node_per_cc) / len(number_of_node_per_cc))
    print('The average number of node per cc is {}'.format(avg_ncc))

    # Plot it if needed
    if plot:
        sns.distplot(number_of_nodes, kde=False, rug=True)
        plt.show()
        sns.distplot(number_of_cc, kde=False, rug=True)
        plt.show()
        sns.distplot(number_of_node_per_cc, kde=False, rug=True)
        plt.show()

    return number_of_nodes, number_of_cc, number_of_node_per_cc


def plot_random(graph_dir, number):
    """
    Plot a random subset
    :param graph_dir: path of the dir
    :param number: number of plots needed
    :return:
    """
    files = os.listdir(graph_dir)
    for i in range(number):
        index = random.randrange(0, len(files))
        path = os.path.join(graph_dir, files[index])
        g = nx.read_gpickle(path)
        printer = dict()
        for edge in g.edges.data():
            printer[(edge[0], edge[1])] = edge[2]['interaction']
        print(printer)
        pos = nx.spring_layout(g)
        nx.draw_networkx_edge_labels(g, pos=pos, edge_labels=printer)
        nx.draw_networkx(g, pos=pos)
        plt.show()


if __name__ == "__main__":
    start_time = time.time()

    # count('../data/output_graph/1radius_12minnodes')
    # count('../data/output_graph/3radius_12minnodes')

    # plot_random('../data/output_graph/v5_hbonds',7)
    # wait = input("press enter to continue")
    # plot_random('../data/output_graph/1radius_12minnodes',7)


    # print('no_hbonds')
    # # count('../data/output_graph/test/1a0i_ATP_0.p')
    # print('hbonds')
    plot_random('../data/output_graph/test',5)

    # print('hbond2')
    # count('../data/output_graph/hbond2')
    #
    # print('3radius_12minnodes')
    # count('../data/output_graph/3radius_12minnodes')
    # # print('1radius_12minnodes')
    # # count('../data/output_graph/1radius_12minnodes')
    # print('border')
    # count('../data/output_graph/border')

    # plot_count('../data/output_graph/border')
    # plot_random('../data/output_graph/border', 10)

    print("--- %s seconds ---" % (time.time() - start_time))
