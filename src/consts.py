NUCLEOTIDES = 'ACGT'  # the order is important in some part of the code (promoter strength)
PROMOTER_LENGTH = 36
START_INDEX = -35
RNAp_EDITED_ZONES = [(-33, -30), (-11, -8)]
RNAi_EDITED_ZONES = [(-33, -30), (-10, -7)]
RNAp_SEQ_ORIGINAL = 'TTGAGATCCTTTTTTTCTGCGCGTAATCTGCTGCTT'
RNAi_SEQ_ORIGINAL = 'TTGAAGTGGTGGCCTAACTACGGCTACACTAGAAGA'
RNA_DATA_COLUMNS = ['Promoter Sequence (-35 to +1)', 'Initial Counts', 'Final Counts', 'Growth Rate', 'Copy Number', 'Predicted Promoter Strength (KbT)']
RNAi_PROM_RNAp_COLUMNS = ['Promoter Sequence (-35 to +1)', 'Copy Number', 'Predicted Promoter Strength (KbT)', 'Copy Number', 'RNAp_seq']

USE_SELECTED_FEATURES = {"selective": False}

ALPHA = "CCAUUGACCGAAGUCGUCUCG"   # rna_p[72:72+21]
ALPHA_RANGE = range(72, 72+21)
EXTENDED_ALPHA_RANGE = range(72, 72+59)
BETA = ""
beta = ""
# for i, b in enumerate(rna_p[72:72+21]):
#     match_idx = bpd.get(i+72, -1)
#     beta += "-" if match_idx == -1 else rna_p[match_idx]


CONSENSUS_POSITIONS_3_STEM_LOOPS = dict({(73, 107), (74, 106), (75, 105),
                                        (77, 103), (78, 102), (79, 101),
                                        (80, 100), (81, 99), (83, 97),
                                        (84, 96), (86, 94), (87, 93), (88, 92)})

CONSENSUS_POSITIONS_ALPHA_BETA_FOLD = {
        72: 194,
        73: 193,
        74: 192,
        75: 191,
        76: 190,
        77: 189,
        78: '-',
        79: 187,
        80: 186,
        81: '-',
        82: '-',
        83: 184,
        84: 183,
        85: 182,
        86: 181,
        87: 180,
        88: 179,
        89: 178,
        90: 177,
        91: 176,
        92: 175,
        93: 174}


CONSENSUS_POSITIONS_EXTENDED_ALPHA_BETA_FOLD = dict({(72, 194), (73, 193), (74, 192), (75, 191),
                                                     (76, 190), (77, 189), (79, 187), (80, 186),
                                                     (83, 184), (84, 183), (85, 182), (86, 181),
                                                     (87, 180), (88, 179), (89, 178), (90, 177),
                                                     (91, 176), (92, 175), (93, 174), (95, 173),
                                                     (96, 172), (97, 171), (98, 170), (109, 150),
                                                     (110, 149), (111, 148), (112, 147), (113, 146),
                                                     (114, 145), (117, 142), (118, 141), (119, 140),
                                                     (120, 139), (121, 137), (122, 136), (123, 135),
                                                     (124, 134), (125, 133), (153, 168), (154, 167),
                                                     (155, 166), (156, 165), (157, 164)})