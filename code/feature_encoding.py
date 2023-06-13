

import numpy as np


# one-hot encoding
def to_one_hot(seqs):
    base_dict = {
        'a': 0, 'c': 1, 'g': 2, 't': 3,
        'A': 0, 'C': 1, 'G': 2, 'T': 3
    }

    one_hot_4_seqs = []
    for seq in seqs:

        one_hot_matrix = np.zeros([4, len(seq)], dtype=float)
        index = 0
        for seq_base in seq:
            one_hot_matrix[base_dict[seq_base], index] = 1
            index = index + 1

        one_hot_4_seqs.append(one_hot_matrix)
    return one_hot_4_seqs

# C2 encoding
def to_C2_code(seqs):
    properties_code_dict = {
        'A': [0, 0], 'C': [1, 1], 'G': [1, 0], 'T': [0, 1],
        'a': [0, 0], 'c': [1, 1], 'g': [1, 0], 't': [0, 1]
    }
    properties_code = []
    for seq in seqs:
        properties_matrix = np.zeros([2, len(seq)], dtype=float)
        m = 0
        for seq_base in seq:
            properties_matrix[:, m] = properties_code_dict[seq_base]
            m = m + 1
        properties_code.append(properties_matrix)
    return properties_code


# NCP encoding
def to_properties_code(seqs):

    properties_code_dict = {
        'A': [1, 1, 1], 'C': [0, 1, 0], 'G': [1, 0, 0], 'T': [0, 0, 1],
        'a': [1, 1, 1], 'c': [0, 1, 0], 'g': [1, 0, 0], 't': [0, 0, 1]
    }
    properties_code = []
    for seq in seqs:
        properties_matrix = np.zeros([3, len(seq)], dtype=float)
        m = 0
        for seq_base in seq:
            properties_matrix[:, m] = properties_code_dict[seq_base]
            m = m + 1
        properties_code.append(properties_matrix)
    return properties_code

# NCP-ND encoding
def to_properties_density_code(seqs):
    properties_code_dict = {
        'A': [1, 1, 1], 'C': [0, 1, 0], 'G': [1, 0, 0], 'T': [0, 0, 1],
        'a': [1, 1, 1], 'c': [0, 1, 0], 'g': [1, 0, 0], 't': [0, 0, 1]
    }
    properties_code = []
    for seq in seqs:
        properties_matrix = np.zeros([4, len(seq)], dtype=float)
        A_num = 0
        C_num = 0
        G_num = 0
        T_num = 0
        All_num = 0
        for seq_base in seq:
            if seq_base == "A":
                All_num += 1
                A_num += 1
                Density = A_num / All_num
                properties_matrix[:, All_num - 1] = properties_code_dict[seq_base] + [Density]
            if seq_base == "C":
                All_num += 1
                C_num += 1
                Density = C_num / All_num
                properties_matrix[:, All_num - 1] = properties_code_dict[seq_base] + [Density]
            if seq_base == "G":
                All_num += 1
                G_num += 1
                Density = G_num / All_num
                properties_matrix[:, All_num - 1] = properties_code_dict[seq_base] + [Density]
            if seq_base == "T":
                All_num += 1
                T_num += 1
                Density = T_num / All_num
                properties_matrix[:, All_num - 1] = properties_code_dict[seq_base] + [Density]
        properties_code.append(properties_matrix)
    return properties_code

