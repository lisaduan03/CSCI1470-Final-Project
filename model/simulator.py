import tensorflow as tf
import os
import sys
from tqdm import tqdm
import networkx as nx
import random

# for categorical encoding of sequences (prior to OHE)
MAP_ENCODE = {'A' : 0, 'C' : 1, 'G' : 2, 'T' : 3}
MAP_DECODE = {0 : 'A', 1 : 'C', 2 : 'G', 3 : 'T'}


def map_encode(seq : str):
    '''
    Turn sequence in {A, C, G, T}^+ into a list of {0, 1, 2, 3}^+.

    Parameters:
    -----------
    seq - sequence of {A, C, G, T} as a string

    Return:
    -------
    A list of {0, 1, 2, 3}
    '''
    split_seq = [*seq]
    return [MAP_ENCODE[x] for x in split_seq]

def map_decode(seq : list[int]):
    '''
    Turn a list sequence in {0, 1, 2, 3}^+ into a string in {A, C, G, T}^+.

    Parameters:
    -----------
    seq - a list of {0, 1, 2, 3}

    Return:
    -------
    A sequence of {A, C, G, T} as a string
    '''
    split_seq = [MAP_DECODE[x] for x in seq]
    return ''.join(split_seq)

class DataSimulator():
    '''
    A class for simulating motif interaction data.
    '''

    def __init__(self, mode : str):
        '''
        Initialize simulator class with an interaction graph

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        '''
        assert(mode in ['SEQ', 'PWM'])
        self.interaction_graph = nx.Graph()
        self.mode = mode
        self.pwm_dict = {}

    def add_motif(self, motif):
        '''
        Add a motif to the interaction graph.

        Parameters:
        -----------
        motif - motif to add to the interaction graph as a string

        Returns:
        --------
        None
        '''
        if self.mode == 'PWM':
            self.pwm_dict[motif[0]] = motif[1]
            self.interaction_graph.add_node(motif[0])
        else:
            self.interaction_graph.add_node(motif)
    
    def add_interaction(self, motif1, motif2):
        '''
        Add a motif interaction to the interaction graph.

        Parameters:
        -----------
        motif1, motif2 - the two motifs to add to the interaction graph with an 
        undirected edge

        Returns:
        --------
        None
        '''
        if self.mode == 'PWM':
            self.pwm_dict[motif1[0]] = motif1[1]
            self.pwm_dict[motif2[0]] = motif2[1]
            self.interaction_graph.add_node(motif1[0], motif2[0])
        else:
            self.interaction_graph.add_edge(motif1, motif2)
    
    def add_motifs(self, motifs):
        '''
        Add multiple motifs to the interaction graph.

        Parameters:
        -----------
        motifs - list of motifs to add to the graph

        Returns:
        --------
        None
        '''
        if self.mode == 'PWM':
            for motif in motifs:
                self.pwm_dict[motif[0]] = motif[1]
                self.interaction_graph.add_node(motif[0])
        else:
            self.interaction_graph.add_nodes_from(motifs)
    
    def add_interactions(self, motif_ints):
        '''
        Add multiple motif interaction pairs to the interaction graph.

        Parameters:
        -----------
        motif_ints - list of motif interaction pairs to add to the graph

        Returns:
        --------
        None
        '''
        if self.mode == 'PWM':
            for motif1, motif2 in motif_ints:
                self.pwm_dict[motif1[0]] = motif1[1]
                self.pwm_dict[motif2[0]] = motif2[1]
                self.interaction_graph.add_edge(motif1[0], motif2[0])
        else:
            self.interaction_graph.add_edges_from(motif_ints)

    def sample_node_pair(self):
        '''
        Sample a random pair of nodes from the graph.

        Parameters:
        -----------
        None

        Returns:
        --------
        A tuple of motifs from the interaction graph
        '''
        return random.choice([x for x in self.interaction_graph.nodes()]), random.choice([x for x in self.interaction_graph.nodes()])

    def gen_neg(self, len : int):
        '''
        Generate a random sequence in {A, C, G, T}^{len}.

        Parameters:
        -----------
        len - the length of the random sequence to generate

        Returns:
        --------
        The generated random sequence
        '''
        seq = ''
        if len == 0:
            return ''
        for _ in range(len):
            seq += random.choice(['A', 'C', 'G', 'T'])
        return seq
    
    def seqs_to_tens(self, seqs : list[str]):
        '''
        Convert a list of sequences in {A, C, G, T} to a tensor according to the
        categorical maps defined above.

        Parameters:
        -----------
        seqs - list of sequences to convert

        Returns:
        --------
        A tensor of dimension (num_seqs, len_seq) filled with {0, 1, 2, 3}
        '''
        encoded_seq = [map_encode(x) for x in seqs]
        return tf.convert_to_tensor(encoded_seq)
    
    def sample_from_pwm(self, name):
        running_samp = ''
        weights = self.pwm_dict[name]
        for i in range(len(weights[0])):
            running_samp += random.choices(['A', 'C', 'G', 'T'], weights=weights[:, i])[0]
        return running_samp


    def simulate(self, seq_lens : int, num_seqs : int, undir : bool):
        '''
        Simulate 1/2*num_seqs positive sequences and 1/2*num_seqs negative
        sequences, all of length seq_lens. Here, we define positive sequence as 
        one that contains a pair of motifs connected by an edge in the graph, 
        and a negative sequence as either random or containing a pair of 
        unconnected motifs.

        Parameters:
        -----------
        seq_lens - length of sequences to generate
        num_seqs - number of sequences to generate

        Returns:
        --------
        The positive sequences as a tensor, the negative sequences as a tensor, 
        and labels for each tensor (all ones for positive tensor, all zeros for 
        negative tensor)
        '''
        pos_data = []
        num_ints = len(self.interaction_graph.edges())

        # for each pair of interacting motifs
        for motif1, motif2 in self.interaction_graph.edges():
            if self.mode == 'PWM':
                len_motif1, len_motif2 = len(self.pwm_dict[motif1][0]), len(self.pwm_dict[motif2][0])
            else:
                len_motif1, len_motif2 = len(motif1), len(motif2)
            # generate a proportional number of positive sequences
            for _ in range((num_seqs // 2) // num_ints):
                # randomly shuffle motifs (undirected edge model)
                assert(seq_lens - len_motif1 - len_motif2 > 0)
                m1, m2 = motif1, motif2
                lm1, lm2 = len_motif1, len_motif2
                if undir and (random.choice([0, 1]) == 1):
                    m1, m2 = motif2, motif1
                    lm1, lm2 = len_motif2, len_motif1
                
                # choose positions in the new sequence at which to place both 
                # motifs
                start1 = random.randint(0, seq_lens - (lm1 + lm2) - 1)
                start2 = random.randint(start1 + lm1, seq_lens - lm2)
            
                # fill in the gaps with negative sequence data
                if self.mode == "PWM":
                    m1 = self.sample_from_pwm(m1)
                    m2 = self.sample_from_pwm(m2)
                pos_data.append(self.gen_neg(start1) + m1 + self.gen_neg(start2 - start1 - len(m1)) + m2 + self.gen_neg(seq_lens - start2 - len(m2)))
        
        neg_data = []
        rand_neg = (num_seqs - len(pos_data))//2
        sys_neg = (num_seqs - len(pos_data) - rand_neg)
        # uncomment for all random negative set
        # rand_neg = (num_seqs - len(pos_data))
        # sys_neg = 0

        # generate half the negative data as random sequences
        for _ in range(rand_neg):
            neg_data.append(self.gen_neg(seq_lens))

        # generate the other half by the same procedure as positives, but only 
        # for motif pairs NOT in the edge set
        for _ in range(sys_neg):
            # find a negative motif pair
            motif1, motif2 = self.sample_node_pair()
            while (motif1, motif2) in [x for x in self.interaction_graph.edges()]:
                motif1, motif2 = self.sample_node_pair()

            if self.mode == 'PWM':
                len_motif1, len_motif2 = len(self.pwm_dict[motif1][0]), len(self.pwm_dict[motif2][0])
            else:
                len_motif1, len_motif2 = len(motif1), len(motif2)

            # same as for positive motif pair
            m1, m2 = motif1, motif2
            lm1, lm2 = len_motif1, len_motif2
            if undir and (random.choice([0, 1]) == 1):
                m1, m2 = motif2, motif1
                lm1, lm2 = len_motif2, len_motif1 

            start1 = random.randint(0, seq_lens - (lm1 + lm2) - 1)
            start2 = random.randint(start1 + lm1, seq_lens - lm2)
        
            if self.mode == "PWM":
                m1 = self.sample_from_pwm(m1)
                m2 = self.sample_from_pwm(m2)
            neg_data.append(self.gen_neg(start1) + m1 + self.gen_neg(start2 - start1 - len(m1)) + m2 + self.gen_neg(seq_lens - start2 - len(m2)))
            
        return self.seqs_to_tens(pos_data), self.seqs_to_tens(neg_data), tf.ones(len(pos_data), dtype=tf.uint8), tf.zeros(len(neg_data), dtype=tf.uint8)