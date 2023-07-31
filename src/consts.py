NUCLEOTIDES = 'ACGT' # the order is important in some part of the code (promoter strength)
PROMOTER_LENGTH = 36
START_INDEX = -35
RNAp_EDITED_ZONES = [(-33, -30), (-11, -8)]
RNAi_EDITED_ZONES = [(-33, -30), (-10, -7)]
RNAp_SEQ_ORIGINAL = 'TTGAGATCCTTTTTTTCTGCGCGTAATCTGCTGCTT'
RNAi_SEQ_ORIGINAL = 'TTGAAGTGGTGGCCTAACTACGGCTACACTAGAAGA'
RNA_DATA_COLUMNS = ['Promoter Sequence (-35 to +1)', 'Initial Counts', 'Final Counts', 'Growth Rate', 'Copy Number', 'Predicted Promoter Strength (KbT)']

CONF_MOTIFS_SEC_NAME = "Motifs"
# CONF_NUCLI_SEC_NAME = "NucliFeatures"
CONF_NUCLI_RELATIONS_SEC_NAME = 'NucleotidesRelations'
CONF_ZONES_SEC_NAME = "StrengthZones"
CONF_ONE_HOT_ENC_SEC_NAME = "OneHotEncoding"
CONF_DENOVO_SEC_NAME = "DenovoMotifs"