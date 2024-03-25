
from . import constants as c
from . import Plot, Evaluate
from .plotting.colors import *
from .plotting.decision_makers import DecisionMaker

'''
mainly exists because I (Jannis) am too stupid to make CLI work
(but also to automatically dump things into my Drive)
'''

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


def run_evaluation(models, datasets, params = {"batch_size": 64, "print_predictions": True, "num_workers": 20}):
    Evaluate()(models, datasets, **params)


def run_plotting(models, plot_types):
    plotting_def = plot_available_models(models)
    figure_dirname = "example-figures/"
    Plot(plot_types = plot_types, plotting_definition = plotting_def,
         figure_directory_name = figure_dirname)


def analyze_models(models, datasets, plot_types=c.DEFAULT_PLOT_TYPES, eval_params={"batch_size": 64, "print_predictions": True, "num_workers": 20}):
    # 1. evaluate models on out-of-distribution datasets
    run_evaluation(models, datasets, eval_params)
    # 2. plot the evaluation results
    run_plotting(models, plot_types)

    #TODO: move results to Drive