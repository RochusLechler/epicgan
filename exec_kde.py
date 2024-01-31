"""calculates KDE of number of particles in the jets for a given dataset
"""




if __name__ == "__main__":

    from epicgan import data_proc
    from epicgan.utils import calc_kde
    import argparse
    import os

    folder = "/home/rochus/Documents/Studium/semester_pisa/cmepda/exam_project/JetNet_datasets"

    parser = argparse.ArgumentParser(description="calculates KDE")
    parser.add_argument("dataset_name", help = "specify dataset for which to compute KDE")
    args = parser.parse_args()

    save_path = os.path.join(folder, str(args.dataset_name) + ".pkl")

    dataset   = data_proc.get_dataset(str(args.dataset_name))

    kde = calc_kde(dataset, file_path = save_path)
