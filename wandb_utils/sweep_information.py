import os
from typing import List, Dict

import wandb


def get_top_n_sweep_runs(sweep_id: str, n: int, metric:str) -> List[wandb.apis.public.Run]:
    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    sweep_runs = sweep.runs
    sweep_runs = sorted(sweep_runs, key=lambda x: x.config[metric], reverse=True)
    return sweep_runs[:n]

def get_config_info_about_runs(runs:List[wandb.apis.public.Run], needed_info:List[str]) -> Dict[str, Dict[str, str]]:
    info = {}
    for run in runs:
        info[run.name] = {}
        for n_inf in needed_info:
            info[run.name][n_inf] = run.config.get(n_inf)

    return info

def download_model_from_run(run:wandb.apis.public.Run, output_path:str, model_name:str) -> None:
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    run.file(model_name).download(output_path, replace=True)

def get_sweep_info(sweep_id:str) -> Dict[str, str]:
    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    return sweep.config

if __name__ == '__main__':
    # Example of usage
    sweep_id = 'denisdresvyanskiy/Engagement_recognition_fusion/2rzwky0h'
    info = get_sweep_info(sweep_id)
    print(info)


