'''
this code is based on the articles (https://www.nature.com/articles/s41467-022-32829-50) code
'Promoter_Calculator_v1_0' (https://github.com/hsalis/SalisLabCode/tree/master/Promoter_Calculator)

The calculations of this code are based on biophysical properties of the binding of the
interactions between RNA polymerase, sigma factor, and promoter DNA sequences in bacteria.
'''

from src.features.delta_G import util
import numpy as np
import os
import pandas as pd

# Get the directory path of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# constants for the calculation of the Tx_rate in the experiments according to the deltaG model
# if organism == 'in vitro':
#     self.K = 42.00000
#     self.BETA = 0.81632623
# elif organism == 'Escherichia coli str. K-12 substr. MG1655':
#     self.K = 42.00000
#     self.BETA = 1.636217004872062
# else:
#     self.K = 42.00000
#     self.BETA = 1.636217004872062

# the constants we use - could be modified (via machine learning) for the data we are using
LOGK   = -2.80271176
BETA    = 0.81632623
K = 42.00000

def get_matrices(two_mer_encoder, three_mer_encoder, spacer_encoder, coeffs):

    #Extract dG values from model coefficients
    ref10_0 = coeffs.tolist()[0:64]
    ref10_3 = coeffs.tolist()[64:128]
    ref35_0 = coeffs.tolist()[128:192]
    ref35_3 = coeffs.tolist()[192:256]
    discs   = coeffs.tolist()[256:256+64]
    x10     = coeffs.tolist()[256+64:256+64+16]
    spacs   = coeffs.tolist()[256+64+16:256+64+16+3]

    # make dG matrices for each feature
    dg10_0  = util.get_dg_matrices(ref10_0, three_mer_encoder)
    dg10_3  = util.get_dg_matrices(ref10_3, three_mer_encoder)
    dg35_0  = util.get_dg_matrices(ref35_0, three_mer_encoder)
    dg35_3  = util.get_dg_matrices(ref35_3, three_mer_encoder)
    dmers   = util.get_dg_matrices(discs, three_mer_encoder)
    x10mers = util.get_dg_matrices(x10, two_mer_encoder)
    spacers = util.get_dg_matrices(spacs, spacer_encoder)

    return dg10_0, dg10_3, dg35_0, dg35_3, dmers, x10mers, spacers


def linear_free_energy_model(h35, spacer, h10, disc, dg10_0, dg10_3, dg35_0, dg35_3, dmers, x10mers, spacers, coeffs, inters):

    # CATEGORICAL FEATURES
    ext10           = spacer[-3:-1] # TGN motif, contacts sigma
    hex10_0         = h10[0:3]
    hex10_3         = h10[3::]
    hex35_0         = h35[0:3]
    hex35_3         = h35[3::]
    disc_first_3mer = disc[0:3]
    spacer_length   = str(len(spacer))

    # NUMERICAL FEATURES - in our sequences we don't have data for the numerical features (UP and ITR)

    dg10      = dg10_0[hex10_0] + dg10_3[hex10_3]
    dg35      = dg35_0[hex35_0] + dg35_3[hex35_3]
    dg_disc   = dmers[disc_first_3mer]
    dg_ext10  = x10mers[ext10]

    x = float(spacer_length)
    dg_spacer = 0.1463*x**2 - 4.9113*x + 41.119

    dG_apparent  = (dg10 + dg35 + dg_disc + dg_ext10 + dg_spacer + inters[0] - LOGK)/BETA
    dG_total     = dg10 + dg35 + dg_disc + dg_ext10 + dg_spacer + inters[0]

    return dG_total, dG_apparent, dg10, dg35, dg_disc, dg_ext10, dg_spacer  # without UP and ITR regions


def calculate_dG_and_Tx(sequence):
    # Specify fixed promoter regions range:
    TSS = 35
    DISC_length = 6
    HEX10_length = 6
    SPACER_length = 17
    HEX35_length = 6
    def seq_calculate_dG_and_Tx(sequence):
        tempdisc = sequence[TSS - DISC_length:TSS]
        temp10     = sequence[ TSS - DISC_length - HEX10_length : TSS - DISC_length]
        tempspacer = sequence[ TSS - DISC_length - HEX10_length - SPACER_length : TSS - DISC_length - HEX10_length ]
        temp35     = sequence[ TSS - DISC_length - HEX10_length - SPACER_length - HEX35_length : TSS - DISC_length - HEX10_length - SPACER_length]

        # load the constant values of the deltaG model
        model = np.load(os.path.join(current_dir, 'free_energy_coeffs.npy'))
        inters = np.load(os.path.join(current_dir, 'model_intercept.npy'))

        two_mer_encoder = util.kmer_encoders(k = 2)
        three_mer_encoder = util.kmer_encoders(k = 3)
        spacer_encoder = util.length_encoders(16, 18)
        dg10_0, dg10_3, dg35_0, dg35_3, dmers, x10mers, spacers = get_matrices(
            two_mer_encoder=two_mer_encoder, three_mer_encoder=three_mer_encoder,
            spacer_encoder=spacer_encoder, coeffs=model)

        dG_total, dG_apparent, dG_10, dG_35, dG_disc, dG_ext10, dG_spacer = linear_free_energy_model(
            temp35, tempspacer, temp10, tempdisc, dg10_0, dg10_3, dg35_0, dg35_3, dmers, x10mers, spacers, model, inters)

        import math
        results_list = [dG_total, dG_apparent, dG_10, dG_35, dG_disc, dG_ext10, dG_spacer]
        for result in results_list:
            if math.isnan(result) == True:
                print(sequence, dG_total, dG_apparent, dG_10, dG_35, dG_disc, dG_ext10, dG_spacer)

        Tx_rate = K * math.exp(- BETA * dG_total)
        return dG_total, dG_apparent, Tx_rate
    return pd.DataFrame(sequence.apply(seq_calculate_dG_and_Tx).tolist(), columns=['dG_total', 'dG_apparent', 'Tx_rate'])


# check the code
# sequence = 'TTGCAATCCTTTTTTTCTGCGCGTATGCTGCTGCTT'
# dG_total, dG_apparent, Tx_rate = calculate_dG_and_Tx(sequence)
# print(Tx_rate)
# print(dG_total)
# print(dG_apparent)

