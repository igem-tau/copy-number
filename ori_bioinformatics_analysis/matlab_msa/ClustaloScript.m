%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
COL_NAME = 'RNAP_promoter';
FOLDER_NAME = 'C:\Users\dsmld\Desktop\iGEM';

CLUSTALO_path = "C:\Users\dsmld\Desktop\clustal-omega-1.2.2-win64";
table_file = "C:\Users\dsmld\Desktop\thresh_40bp_rnap_HDB.xlsx";
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sheets = sheetnames(table_file);
sheets = sheets(2: end -1);
for j = 1 : length(sheets)
    T = readtable(table_file,'Sheet', sheets{j}, 'ReadVariableNames', true);
    rowsToRemove = strcmp(T.(COL_NAME), '');
    T(rowsToRemove, :) = [];
    col_data = T.(COL_NAME);
    variables = T.Properties.VariableNames;
    col_ind = find(strcmp(variables, COL_NAME));
    strand_direction = T.(variables{col_ind+1});
    assert(numel(unique(strand_direction)), 'direction column contains incorrect amount of values, should only contain reverse forward and forward')
    data=[]
    for i=1 : length(col_data)
        data(i).Header = string(i);
        if strcmp(strand_direction{i}, 'forward') 
            data(i).Sequence = col_data{i};
        else
            data(i).Sequence = seqrcomplement(col_data{i});
        end
    end
    disp(sheets{j})
    disp(length(data))
    INPUT_FILE_NAME = [FOLDER_NAME,'\',COL_NAME, '_', sheets{j}, '.fasta'];
    OUTPUT_FILE_NAME = [FOLDER_NAME,'\',COL_NAME, '_', sheets{j}, '_msa.fasta'];

    cd(CLUSTALO_path) % path to Clustalo's exe file.
    fastawrite(INPUT_FILE_NAME,data);
    
    % run Clustalo: --full increases accuracy of the alignment 
    [status, cmdout] = system(['clustalo -i ' INPUT_FILE_NAME ' -o ' OUTPUT_FILE_NAME ' --full'],'-echo'); % output file format is fasta.
    if status ~= 0
        error('clustalo returned a non-zero exit status. Exiting script.'); 
    end
    
    f_msa = fastaread(OUTPUT_FILE_NAME);
    seqconsensus_fmsa=seqconsensus(f_msa,'gaps','all')
    [Score,Alignment]=nwalign(seqconsensus_fmsa, "CTTCTTGAGATCCTTTTTTTCTGCGCGTAATCTGCTGCTT")
   
end
% seqalignviewer(f_msa)
% 
% distances = seqpdist(f_msa, 'squareform', true);
% tree = seqlinkage(distances, 'single');



% % RNAI_RNAP_seq_dict={
% %     "RNAP_promoter" : {
% %         "forward": "TTGAGATCCTTTTTTTCTGCGCGTAATCTGCTGCTT",
% %                 CTTCTTGAGATCCTTTTTTTCTGCGCGTAATCTGCTGCTT
% %         "reverse_complement":"AAGCAGCAGATTACGCGCAGAAAAAAAGGATCTCAA"
% %     },
% %     "RNAI_promoter" : {
% %         "forward": "TTGAAGTGGTGGCCTAACTACGGCTACACTAGAAGA",
% %         "reverse_complememt":"TCTTCTAGTGTAGCCGTAGTTAGGCCACCACTTCAA"
% %     },
% %     "RNAI_seq" : {
% %         "forward": "AACAGTATTTGGTATCTGCGCTCTGCTGAAGCCAGTTACCTTCGGAAAAAGAGTTGGTAGCTCTTGATCCGGCAAACAAACCACCGCTGGTAGCGGTGGTTTTTTTGTTTGCAAGCAGCAGATTACGCGCAGAAAAAAAGGATCTCA",
% %         "reverse_complememt":"TGAGATCCTTTTTTTCTGCGCGTAATCTGCTGCTTGCAAACAAAAAAACCACCGCTACCAGCGGTGGTTTGTTTGCCGGATCAAGAGCTACCAACTCTTTTTCCGAAGGTAACTGGCTTCAGCAGAGCGCAGATACCAAATACTGTT"
% %     },
% %     "RNAP_seq" : {
% %         "forward": "GCAAACAAAAAAACCACCGCTACCAGCGGTGGTTTGTTTGCCGGATCAAGAGCTACCAACTCTTTTTCCGAAGGTAACTGGCTTCAGCAGAGCGCAGATACCAAATACTGTTCTTCTAGTGTAGCCGTAGTTAGGCCACCACTTCAAGAACTCTGTAGCACCGCCTACATACCTCGCTCTGCTAATCCTGTTACCAGTGGCTGCTGCCAGTGGCGATAAGTCGTGTCTTACCGGGTTGGACTCAAGACGATAGTTACCGGATAAGGCGCAGCGGTCGGGCTGAACGGGGGGTTCGTGCACACAGCCCAGCTTGGAGCGAACGACCTACACCGAACTGAGATACCTACAGCGTGAGCTATGAGAAAGCGCCACGCTTCCCGAAGGGAGAAAGGCGGACAGGTATCCGGTAAGCGGCAGGGTCGGAACAGGAGAGCGCACGAGGGAGCTTCCAGGGGGAAACGCCTGGTATCTTTATAGTCCTGTCGGGTTTCGCCACCTCTGACTTGAGCGTCGATTTTTGTGATGCTCGTCAGGGGGGCGGAGCCTATGGAAA",
% %         "reverse_complememt":"TTTCCATAGGCTCCGCCCCCCTGACGAGCATCACAAAAATCGACGCTCAAGTCAGAGGTGGCGAAACCCGACAGGACTATAAAGATACCAGGCGTTTCCCCCTGGAAGCTCCCTCGTGCGCTCTCCTGTTCCGACCCTGCCGCTTACCGGATACCTGTCCGCCTTTCTCCCTTCGGGAAGCGTGGCGCTTTCTCATAGCTCACGCTGTAGGTATCTCAGTTCGGTGTAGGTCGTTCGCTCCAAGCTGGGCTGTGTGCACGAACCCCCCGTTCAGCCCGACCGCTGCGCCTTATCCGGTAACTATCGTCTTGAGTCCAACCCGGTAAGACACGACTTATCGCCACTGGCAGCAGCCACTGGTAACAGGATTAGCAGAGCGAGGTATGTAGGCGGTGCTACAGAGTTCTTGAAGTGGTGGCCTAACTACGGCTACACTAGAAGAACAGTATTTGGTATCTGCGCTCTGCTGAAGCCAGTTACCTTCGGAAAAAGAGTTGGTAGCTCTTGATCCGGCAAACAAACCACCGCTGGTAGCGGTGGTTTTTTTGTTTGC"
% %     }
% % }