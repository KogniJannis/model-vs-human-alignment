import os
import shutil

import json
from .helper import plotting_helper as ph
from .datasets.experiments import get_experiments

#output the overall shape bias with error bars for all available models
def shape_bias(RESULTS_DIR=None, DATA_LOCATION=None):
    if RESULTS_DIR == None:
        RESULTS_DIR = 'results'
    if not os.path.exists(RESULTS_DIR):
        print(f"{RESULTS_DIR} not found and created")
        os.makedirs(RESULTS_DIR)

    if DATA_LOCATION == None:
        DATA_LOCATION = 'raw-data/cue-conflict'
    if not os.path.exists(DATA_LOCATION):
        raise Exception(f"Datalocation {DATA_LOCATION} does not exist")
    
    #dataset_names= ["cue-conflict"]
    #datasets = get_experiments(dataset_names)
    #ph.get_experimental_data(datasets)

    for filename in os.listdir(DATA_LOCATION):
        assert filename.endswith('.csv'), "not a csv"
        df = pd.read_csv(os.path.join(DATA_LOCATION, filename))
        subject_name = df['subj'][0]
        assert len(model_name) >= 3, "model_name should be at least three characters long" #sanity check
        print(f"Calculating shape-bias for model: {model_name}")
        class_avgs = df.groupby(["category"]).apply(lambda x: analysis.analysis(df=x)["shape-bias"])
        shape_bias_dict[subject_name]['scores'] = class_avgs.tolist()
        shape_bias_dict[subject_name]['mean'] = sum(class_avgs) / len(class_avgs)
    with open(os.path.join(RESULTS_DIR, 'shapebias_scores.json'), 'w') as json_file:
        json.dump(shape_bias_dict, json_file, indent=4)