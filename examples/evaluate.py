import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from modelvshuman import Plot, Evaluate
from modelvshuman import constants as c
from plotting_definition import plot_available_models


def run_evaluation(models, datasets):
    params = {"batch_size": 64, "print_predictions": True, "num_workers": 20}
    Evaluate()(models, datasets, **params)


def run_plotting(models, plot_types):
    plotting_def = plot_available_models(models)
    figure_dirname = "example-figures/"
    Plot(plot_types = plot_types, plotting_definition = plotting_def,
         figure_directory_name = figure_dirname)

    # In examples/plotting_definition.py, you can edit
    # plotting_definition_template as desired: this will let
    # the toolbox know which models to plot, and which colours to use etc.


if __name__ == "__main__":

    models = ["resnet50", "bagnet33", "simclr_resnet50x1"]
    datasets = c.DEFAULT_DATASETS # or e.g. ["cue-conflict", "uniform-noise"]
    plot_types = c.DEFAULT_PLOT_TYPES # or e.g. ["accuracy", "shape-bias"]
    
    # 1. evaluate models on out-of-distribution datasets
    run_evaluation(models, datasets)
    # 2. plot the evaluation results
    run_plotting(models, plot_types)
