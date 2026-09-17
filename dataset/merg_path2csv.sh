# bash merge_path2csv.sh
# merge the protein and molecule embedding paths into the target csv

python ./dataset/merge_path2csv.py \
  --target_csv ./dataset/two_splits/seq_idt/kcat/train.csv \
  --output_csv ./dataset/two_splits/seq_idt/kcat/train.csv \
  --prot_map_csvs ./dataset/unique_catapro_all/unique_enzymes_prot_t5_xl_prot_embedding.csv \
  --mol_map_csvs ./dataset/unique_catapro_all/unique_molecules_mol_t5_mol_embedding.csv \
  --overwrite \
  --strict_coverage