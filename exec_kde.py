"""Calculates kernel density estimation (KDE) of number of (non-zero-padded) particles in the jets 
for a given dataset. Name of dataset must be given when calling from command line, the dataset must 
be stored in folder 'JetNet_datasets'. The calculated KDE will be stored to a .pkl-file in folder
'JetNet_datasets'. 
"""




if __name__ == "__main__":

    import os
    import sys

    from epicgan.data_proc import get_dataset
    from epicgan.utils import calc_kde
    import argparse


    folder = "./JetNet_datasets"

    parser = argparse.ArgumentParser(description="calculates KDE")
    parser.add_argument("dataset_name", help = "specify dataset for which to compute KDE")
    args = parser.parse_args()

    save_path = os.path.join(folder, str(args.dataset_name) + ".pkl")

    try:
        dataset   = get_dataset(str(args.dataset_name))
    except FileNotFoundError:
        print("that file does not exist, program will exit without doing anything")
        sys.exit()


    kde = calc_kde(dataset, file_path = save_path)
