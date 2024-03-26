import os
import shutil

from . import constants as c
from . import Plot, Evaluate
from .plotting.colors import *
from .plotting.decision_makers import DecisionMaker
from .datasets import list_datasets

'''
mainly exists because I (Jannis) am too stupid to make CLI work
(but also to automatically dump things into my Drive)
'''

DEFAULT_PARAMS = {"batch_size": 64, "print_predictions": True, "num_workers": 20}
DEFAULT_DRIVE_PATH = '/content/drive/My Drive/aoa-data/model-vs-human/'
DEFAULT_DATA_PATH = '/content/model-vs-human/'
DEFAULT_RESULTS_PATH = '/content/drive/My Drive/aoa-results/model-vs-human/'

def plot_available_models(models):
    """ 
    automatically plots the available models for quick iteration
    colors and markers are set to a single default 
    (maybe find some dict-based solution in the future) 
    """
    def decision_maker_fun(df, models=models):
        decision_makers = []
        for model in models:
            decision_makers.append(DecisionMaker(name_pattern=model,
                                   color=rgb(65, 90, 140), marker="o", df=df,
                                   plotting_name=model))
        decision_makers.append(DecisionMaker(name_pattern="subject-*",
                               color=rgb(165, 30, 55), marker="D", df=df,
                               plotting_name="humans"))
        return decision_makers
    return decision_maker_fun



def transfer_data(src_folder, target_folder):
    """
    used to transfer data from and to the Drive which should already be mounted as /content/drive/My Drive/ 
    """
    assert os.path.exists(target_folder), "Target folder does not exist."

    for root, dirs, files in os.walk(src_folder):
        target_root = os.path.join(target_folder, os.path.relpath(root, src_folder))

        assert os.path.exists(target_root), f"Target subfolder '{target_root}' does not exist."

        for file in files:
            src_file = os.path.join(root, file)
            target_file = os.path.join(target_root, file)
            shutil.copy2(src_file, target_file)


def run_evaluation(models, datasets=c.DEFAULT_DATASETS, params=DEFAULT_PARAMS):
    Evaluate()(models, datasets, **params)


def run_plotting(models, plot_types=c.DEFAULT_PLOT_TYPES):
    plotting_def = plot_available_models(models)
    figure_dirname = "example-figures/"
    Plot(plot_types = plot_types, plotting_definition = plotting_def,
         figure_directory_name = figure_dirname)


def analyze_models(models,
                   datasets=c.DEFAULT_DATASETS,
                   plot_types=c.DEFAULT_PLOT_TYPES,
                   eval_params=DEFAULT_PARAMS,
                   drive_path=DEFAULT_DRIVE_PATH,
                   data_path=DEFAULT_DATA_PATH,
                   results_path=DEFAULT_RESULTS_PATH):
    # 1. evaluate models on out-of-distribution datasets
    run_evaluation(models, datasets, eval_params)

    #get the human data
    transfer_data(drive_path + 'raw_data', data_path + 'raw_data')

    # 2. plot the evaluation results
    run_plotting(models, plot_types)

    #transfer data back
    transfer_data(data_path + 'raw_data', results_path + 'raw_data')
    transfer_data(data_path + 'figures', results_path + 'figures')