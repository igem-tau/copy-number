import Bio.Entrez, Bio.Align, Bio.SeqIO, Bio.SeqRecord, Bio.pairwise2, Bio.Seq
import os
import pandas as pd

def download_reference_fasta(reference, output_path):
    Bio.Entrez.email = "your_email@example.com"
    try:
        handle = Bio.Entrez.efetch(
            db="nucleotide", id=reference, rettype="fasta", retmode="text"
        )
        with open(output_path, "w") as f:
            f.write(handle.read())
        print("FASTA file downloaded successfully.")
    except Exception as e:
        print("Failed to download the FASTA file:", str(e))


def download_ecoli_fastas(oris_file, plasmids_fastas_dir):
    if not os.path.exists(plasmids_fastas_dir):
        os.mkdir(plasmids_fastas_dir)

    df = pd.read_excel(oris_file)
    for index,row in df.iterrows(): 
        ref=row["Reference"]
        output=row["Species"]+"_"+row["Genus"]+"_"+row["Family"]+"_"+row["Order"]+"_"+ref
        download_reference_fasta(ref, os.path.join(plasmids_fastas_dir, output + ".fasta"))

if __name__ == "__main__":
    plasmid_families_file = "data/plasmid_families.xlsx"
    plasmids_fastas_dir = "fastas"
    download_ecoli_fastas(plasmid_families_file, plasmids_fastas_dir)