import os


def run_func(description, ppi_path, pseq_path, vec_path,
            index_path, gnn_model, test_all):
    os.system("python gnn_test.py \
            --description={} \
            --ppi_path={} \
            --pseq_path={} \
            --vec_path={} \
            --index_path={} \
            --gnn_model={} \
            --test_all={} \
            ".format(description, ppi_path, pseq_path, vec_path, 
                    index_path, gnn_model, test_all))

if __name__ == "__main__":
    description = "test"

    ppi_path = "./data/protein.actions.SHS27k.STRING.txt"
    # ppi_path = "./data/9606.protein.actions.all_connected.txt"
    pseq_path = "./data/protein.SHS27k.sequences.dictionary.tsv"
    # pseq_path ="./data/protein.STRING_all_connected.sequences.dictionary.tsv"
    vec_path = "./data/vec5_CTC.txt"

    index_path = "bfs_train_1"
    gnn_model = "ymz_27k/aftgnet/aftgnet_bfs_1/gnn_model_train.ckpt"

    # test_all = "True"
    test_all = "True"

    # test test

    run_func(description, ppi_path, pseq_path, vec_path, index_path, gnn_model, test_all)
