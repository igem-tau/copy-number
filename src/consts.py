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

CONSENSUS_POSITIONS = {
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