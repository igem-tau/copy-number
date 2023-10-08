NUCLEOTIDES = 'ACGT'  # the order is important in some part of the code (promoter strength)
PROMOTER_LENGTH = 36
START_INDEX = -35
RNAp_EDITED_ZONES = [(-33, -30), (-11, -8)]
RNAi_EDITED_ZONES = [(-33, -30), (-10, -7)]
RNAp_SEQ_ORIGINAL = 'TTGAGATCCTTTTTTTCTGCGCGTAATCTGCTGCTT'
RNAi_SEQ_ORIGINAL = 'TTGAAGTGGTGGCCTAACTACGGCTACACTAGAAGA'
RNA_DATA_COLUMNS = ['Promoter Sequence (-35 to +1)', 'Initial Counts', 'Final Counts', 'Growth Rate', 'Copy Number', 'Predicted Promoter Strength (KbT)']
RNAi_PROM_RNAp_COLUMNS = ['Promoter Sequence (-35 to +1)', 'Copy Number', 'Predicted Promoter Strength (KbT)', 'Copy Number', 'RNAp_seq']

ALPHA = 'GGUAACUGGCUUCAGCAGAGC'  # rna_p[3:24]
ALPHA_RANGE = range(3, 24)
EXTENDED_ALPHA_RANGE = range(3, 3+59)

BETA = 'UCGCUCUGCUAAUCCUGUUACC'  # 104:126
BETA_RANGE = range(104, 126)

GAMMA ='UGGCCCAACCUGAGUUCUGCUA'  # 160:181
GAMMA_RANGE = range(160, 181)

RANGES_DICT = {"alpha_range": ALPHA_RANGE, "beta_range": BETA_RANGE, "gamma_range": GAMMA_RANGE}


CONSENSUS_POSITIONS_3_STEM_LOOPS = dict({(4, 33), (5, 32), (6, 31),
                                        (7, 30), (9, 28), (10, 27),
                                        (11, 26), (12, 24), (13, 23),
                                        (14, 22), (15, 21)})

CONSENSUS_POSITIONS_ALPHA_BETA_FOLD = {
        4: 126, 5: 125, 6: 124, 7: 123, 8: 122, 9: 121, 11: 119, 12: 118, 14: 116,
        15: 115, 17: 114, 18: 113, 19: 112, 20: 111, 21: 110, 22: 109, 23: 108, 24: 107,
        25: 106, 27: 105, 28: 104
        }

CONSENSUS_POSITIONS_EXTENDED_ALPHA_BETA_FOLD = {
    4: 126, 5: 125, 6: 124, 7: 123, 8: 122, 9: 121, 11: 119, 12: 118, 14: 116, 15: 115, 17: 114,
    18: 113, 19: 112, 20: 111, 21: 110, 22: 109, 23: 108, 24: 107, 25: 106, 27: 105, 28: 104, 41: 82,
    42: 81, 43: 80, 44: 79, 45: 78, 46: 77, 49: 74, 50: 73, 51: 72, 52: 71, 55: 67, 56: 66, 57: 65,
    85: 100, 86: 99, 87: 98, 88: 97, 89: 96
    }

# CONSENSUS_POSITIONS_C_RICH_AREA = {218: 324,
#                                   219: 323,
#                                   220: 322,
#                                   221: 321,
#                                   222: 320,
#                                   226: 264,
#                                   227: 263,
#                                   228: 262,
#                                   229: 261,
#                                   266: 317,
#                                   267: 316,
#                                   268: 315,
#                                   275: 307,
#                                   276: 306,
#                                   277: 305,
#                                   278: 304,
#                                   279: 303,
#                                   280: 302,
#                                   281: 301,
#                                   284: 298,
#                                   285: 297}

RNA_TYPE_CONST = {"RNA": 'p'}
TARGET_COLUMN = 'Raw Copy Number'