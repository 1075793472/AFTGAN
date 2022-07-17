import torch
import esm
import joblib

# Load ESM-1b model
model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()

# Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
data = []


def get_protein_aac( pseq_path):
    # aac: amino acid sequences

    pseq_path = pseq_path
    pseq_dict = {}
    protein_len = []

    for line in tqdm(open(pseq_path)):
        line = line.strip().split('\t')
        if line[0] not in self.pseq_dict.keys():
            pseq_dict[line[0]] = line[1]
            protein_len.append(len(line[1]))
    return
    # print("protein num: {}".format(len(self.pseq_dict)))
    # print("protein average length: {}".format(np.average(self.protein_len)))
    # print("protein max & min length: {}, {}".format(np.max(self.protein_len), np.min(self.protein_len)))

    # ("protein2 with mask","KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    # ("protein3",  "K A <mask> I S Q"),

data.append(['protein1',"MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"])
# print(data[0])
batch_labels, batch_strs, batch_tokens = batch_converter(data)
print(batch_labels)
print(batch_strs)
print(batch_tokens)
# Extract per-residue representations (on CPU)
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
token_representations = results["representations"][33]
token_representations= token_representations.reshape((-1,1280))

print(token_representations.shape)
# Generate per-sequence representations via averaging
# NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
sequence_representations = []
for i, (_, seq) in enumerate(data):
    sequence_representations.append(token_representations[i, 1 : len(seq) + 1].mean(0))
print(sequence_representations)
print(sequence_representations[0].shape)
print(results['contacts'].shape)

# Look at the unsupervised self-attention map contact predictions
# import matplotlib.pyplot as plt
# for (_, seq), attention_contacts in zip(data, results["contacts"]):
#     plt.matshow(attention_contacts[: len(seq), : len(seq)])
#     plt.title(seq)
#     plt.show()
