NUCLEOTIDES = 'ACGT'  # the order is important in some part of the code (promoter strength)
PROMOTER_LENGTH = 36
START_INDEX = -35
RNAp_EDITED_ZONES = [(-33, -30), (-11, -8)]
RNAi_EDITED_ZONES = [(-33, -30), (-10, -7)]
RNAp_SEQ_ORIGINAL = 'TTGAGATCCTTTTTTTCTGCGCGTAATCTGCTGCTT'
RNAi_SEQ_ORIGINAL = 'TTGAAGTGGTGGCCTAACTACGGCTACACTAGAAGA'
RNA_DATA_COLUMNS = ['Promoter Sequence (-35 to +1)', 'Initial Counts', 'Final Counts', 'Growth Rate', 'Copy Number', 'Predicted Promoter Strength (KbT)']
RNAi_PROM_RNAp_COLUMNS = ['Promoter Sequence (-35 to +1)', 'Copy Number', 'Predicted Promoter Strength (KbT)', 'Copy Number', 'RNAp_seq']

ALPHA = "CCAUUGACCGAAGUCGUCUCG"   # rna_p[72:72+21]
ALPHA_RANGE = range(72, 72+21)
EXTENDED_ALPHA_RANGE = range(72, 72+59)

#ALPHA ='CCAUUGACCGAAGUCGUCUCG' #72:92
BETA = 'AGCGAGACGAUUAGGACAAUGG' #173:194
BETA_RANGE = range(173, 195)
GAMMA ='UGGCCCAACCUGAGUUCUGCUA' #229:250
GAMMA_RANGE = range(229, 251)

RANGES_DICT = {"alpha_range": ALPHA_RANGE, "beta_range": BETA_RANGE, "gamma_range": GAMMA_RANGE}


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
                                                     (96, 172), (97, 171), (98, 170), (99, 169), (109, 150),
                                                     (110, 149), (111, 148), (112, 147), (113, 146),
                                                     (114, 145), (117, 142), (118, 141), (119, 140),
                                                     (120, 139), (121, 137), (122, 136), (123, 135),
                                                     (124, 134), (125, 133), (153, 168), (154, 167),
                                                     (155, 166), (156, 165), (157, 164)})

CONSENSUS_POSITIONS_C_RICH_AREA = {218: 324,
                                  219: 323,
                                  220: 322,
                                  221: 321,
                                  222: 320,
                                  226: 264,
                                  227: 263,
                                  228: 262,
                                  229: 261,
                                  266: 317,
                                  267: 316,
                                  268: 315,
                                  275: 307,
                                  276: 306,
                                  277: 305,
                                  278: 304,
                                  279: 303,
                                  280: 302,
                                  281: 301,
                                  284: 298,
                                  285: 297}

RNA_TYPE_CONST = {"RNA": 'p'}