"""actually calling the training and evaluation script
"""

import training
import evaluate_performance



if __name__ == "__main__":

    dataset_names30 = ["gluon30", "quark30", "top30"]
    n_points = 30

    for name in dataset_names30:
        model = training.TrainableModel(name, n_points)
        best_w_distance, best_epoch = model.training()
        print(f"for {name}, best epoch was epoch {best_epoch} with W.-distance {best_w_distance}")

    figures = {}
    for name in dataset_names30:
        result_dict, fig = evaluate_performance.evaluate_performance(name, n_points, make_plots = True)
        print(f"for dataset {name} we have: ")
        print(result_dict)
        figures[name] = fig
