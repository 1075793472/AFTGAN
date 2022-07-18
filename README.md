# AFTGAN

## Abstract

Protein-protein interaction (PPI) networks and transcriptional regulatory networks are important for regulating cells and their signaling. A thorough understanding of PPIs can provide insights into cellular physiology in normal and disease states. Although there are many methods to predict PPI, the performance of prediction methods is still challenging for the interaction prediction between unknown proteins. We proposes a novel multi-type PPI prediction method (AFTGAN) by considering the data and interaction feature of unknown proteins. It firstly constructs the protein sequence feature and PPI graph according to the sequence and PPI information of proteins. For the protein sequence feature, in addition to the embedding of amino acid co-occurrence similarity and the one-hot embedding composition of electrostatic and hydrophobic similarity between amino acids, it also adds ESM-1b embedding features as input features. For the PPI graph, it uses proteins as nodes and 7 kinds of protein interactions as edges to construct 7 related PPI graphs as link matrices, and then input them into the graph attention network. For the neural network framework, it applies the Transformer encoder containing the Attention Free Transformer (AFT) module to extract protein sequence features, and then uses the extracted protein sequence features as PPI graph node features, which are input to the graph attention network together with the constructed PPI graph to extract relational features of protein pairs. Each protein node continuously updates its own feature according to the multi-head self-attention mechanism and adjacent node information. Finally, a fully connected layer (FC) is used as a classifier for the multi-label PPI prediction.

## Using AFTGAN

This repository contains:
- Training
- Testing

### Training

Training codes in gnn_train.py, and the run script in run.py.

```   
python gnn_train.py --ppi_path=C:\Users\Administrator\Desktop\AFT\data\9606.protein.actions.all_connected.txt   --pseq_path=C:\Users\Administrator\Desktop\AFT\data\protein.STRING_all_connected.sequences.dictionary.tsv  --vec_path=./data/vec5_CTC.txt  --batch_size=1024   --epochs=2000  --split_new=True --split_mode=bfs --train_valid_index_path=bfs_1 --save_path=./model

```

#### Dataset and Pre-trained model Download:

This repositorie uses the processed dataset download and Pre-trained model path:
- https://pan.baidu.com/s/12FitbTeodAoWlHlR1exqog  (Extraction code: inwb)

### Testing

Testing codes in gnn_test.py , and the run script in run_test.py.

gnn_test.py: It can test the overall performance, and can also make in-depth analysis to test the performance of different test data separately. 
