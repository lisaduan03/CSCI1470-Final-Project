import tensorflow as tf
import os
import sys
from tqdm import tqdm
import networkx as nx
import random

MAP_ENCODE = {'A' : 0, 'C' : 1, 'G' : 2, 'T' : 3}
MAP_DECODE = {'A' : 0, 'C' : 1, 'G' : 2, 'T' : 3}

def map_encode(seq : str):
    split_seq = [*seq]
    return [MAP_ENCODE[x] for x in split_seq]

def map_decode(seq : list[int]):
    split_seq = [MAP_DECODE[x] for x in seq]
    return ''.join(split_seq)

class DataSimulator():

    def __init__(self):
        self.interaction_graph = nx.Graph()

    def add_motif(self, motif : str):
        self.interaction_graph.add_node(motif)
    
    def add_interaction(self, motif1 : str, motif2 : str):
        self.interaction_graph.add_edge(motif1, motif2)
    
    def add_motifs(self, motifs : list[str]):
        self.interaction_graph.add_nodes_from(motifs)
    
    def add_interactions(self, motif_ints : list[tuple[str, str]]):
        self.interaction_graph.add_edges_from(motif_ints)

    def gen_neg(self, len : int):
        seq = ''
        if len == 0:
            return ''
        for _ in range(len):
            seq += random.choice(['A', 'C', 'G', 'T'])
        return seq
    
    def seqs_to_tens(self, seqs : list[str]):
        encoded_seq = [map_encode(x) for x in seqs]
        return tf.convert_to_tensor(encoded_seq)

    def simulate(self, seq_lens : int, num_seqs : int):
        pos_data = []
        num_ints = len(self.interaction_graph.edges())
        for motif1, motif2 in self.interaction_graph.edges():

            for _ in range((num_seqs // 2) // num_ints):
                assert(seq_lens - len(motif1) - len(motif2) > 0)
                start1 = random.randint(0, seq_lens - (len(motif1) + len(motif2)) - 1)
                start2 = random.randint(start1 + len(motif1), seq_lens - len(motif2))
            
                pos_data.append(self.gen_neg(start1) + motif1 + self.gen_neg(start2 - start1 - len(motif1)) + motif2 + self.gen_neg(seq_lens - start2 - len(motif2)))
        
        neg_data = []
        for _ in range(num_seqs - len(pos_data)):
            neg_data.append(self.gen_neg(seq_lens))

        return self.seqs_to_tens(pos_data), self.seqs_to_tens(neg_data)