from Bio import Entrez
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align.Applications import MuscleCommandline
import os
import pandas as pd


def download_reference_fasta(reference, output_path):
    Entrez.email = 'your_email@example.com'  # Replace with your email address

    try:
        handle = Entrez.efetch(db='nucleotide', id=reference, rettype='fasta', retmode='text')
        with open(output_path, 'w') as f:
            f.write(handle.read())
        print("FASTA file downloaded successfully.")
    except Exception as e:
        print("Failed to download the FASTA file:", str(e))


def get_fastas():
    if not os.path.exists("fastas"):
        os.mkdir("fastas")

    df = pd.read_excel("journal.pgen.1009919.s006.xlsx")
    ecoli_ref_lst = df[df["Genus"] == "Escherichia"]["Reference"]
    for r in ecoli_ref_lst:
        download_reference_fasta(r, os.path.join("fastas", r + ".fasta"))


def find_promoter_alignment(fasta_file, promoter_sequence):
    # Read the FASTA file and create a list of SeqRecord objects
    records = list(SeqIO.parse(fasta_file, "fasta"))

    # Create a temporary FASTA file containing only the promoter sequence
    temp_file = "temp.fasta"
    SeqIO.write(SeqRecord(Seq(promoter_sequence)), temp_file, "fasta")

    # Perform pairwise alignment using Muscle
    muscle_cline = MuscleCommandline(cmd=r"C:\Users\User1\BioInfTools\muscle.exe", input=os.path.abspath(temp_file), fasta=True)
    stdout, stderr = muscle_cline()

    # Print the alignment result
    print(stdout)

    # Clean up the temporary file
    os.remove(temp_file)


if __name__ == '__main__':
    # Usage example:
    fasta_file = r"fastas\AF158026.1.fasta"
    promoter_sequence_p = "TTGAGATCCTTTTTTTCTGCGCGTAATCTGCTGCTT"

    find_promoter_alignment(fasta_file, promoter_sequence_p)
