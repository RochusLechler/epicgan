"""Calculates KDE of number of particles in the jets for a given dataset.
Name of dataset must be given when calling from command line.
"""




if __name__ == "__main__":

    import os
    import sys

    from epicgan import data_proc
    from epicgan.utils import calc_kde
    import argparse


    folder = "./JetNet_datasets"

    parser = argparse.ArgumentParser(description="calculates KDE")
    parser.add_argument("dataset_name", help = "specify dataset for which to compute KDE")
    args = parser.parse_args()

    save_path = os.path.join(folder, str(args.dataset_name) + ".pkl")

    try:
        dataset   = data_proc.get_dataset(str(args.dataset_name))
    except FileNotFoundError:
        print("that file does not exist, program will exit without doing anything")
        sys.exit()


    kde = calc_kde(dataset, file_path = save_path)
