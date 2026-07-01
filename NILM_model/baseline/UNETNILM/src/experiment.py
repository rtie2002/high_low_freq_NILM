from __future__ import print_function, division
import warnings
warnings.filterwarnings("ignore")
import os
import torch
import numpy as np
from net.model_pl import NILMnet
from data.load_data import ukdale_appliance_data
from net.utils import DictLogger
from pathlib import Path
import pytorch_lightning as pl
from net.utils import get_latest_checkpoint
from utils.utils import set_seed, get_device
import sys

try:
    from IPython.display import clear_output
except ImportError:
    def clear_output(*args, **kwargs):
        pass
from argparse import ArgumentParser
set_seed(seed=7777)
device =  get_device()


class NILMExperiment(object):

    def __init__(self, params):
        """
        Parameters to be specified for the model
        """
        self.MODEL_NAME = params.get('model_name',"CNNModel")
        self.logs_path =params.get('log_path',"../logs/")
        self.checkpoint_path =params.get('checkpoint_path',"../checkpoints/")
        self.results_path = params.get('results_path',"../results/")
        self.chunk_wise_training = params.get('chunk_wise_training',False)
        self.sequence_length = params.get('sequence_length',99)
        self.n_epochs = params.get('n_epochs', 10 )
        self.batch_size = params.get('batch_size',128)
        self.dropout = params.get('dropout', 0.1)
        self.params = params
        
        #create files
        logs = Path(self.logs_path )
        checkpoints = Path(self.checkpoint_path)
        results = Path(self.results_path)
        logs.mkdir(parents=True, exist_ok=True)
        checkpoints.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)
        results.mkdir(parents=True, exist_ok=True)
        
     

    def fit(self):
        file_name = self.params['file_name']
        self.arch = file_name
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=self.checkpoint_path,
            monitor="val_maF1",
            mode="max",
            save_top_k=1,
        )
        logger = DictLogger(self.logs_path, name=file_name, version=self.params["exp_name"])
        trainer_kwargs = dict(
            logger=logger,
            gradient_clip_val=self.params["clip_value"],
            callbacks=[checkpoint_callback],
            max_epochs=self.params["n_epochs"],
            enable_progress_bar=True,
        )
        if torch.cuda.is_available():
            trainer_kwargs["accelerator"] = "gpu"
            trainer_kwargs["devices"] = 1
        else:
            trainer_kwargs["accelerator"] = "cpu"

        self.hparams = NILMnet.add_model_specific_args()
        self.hparams = vars(self.hparams.parse_args([]))
        self.hparams.update(self.params)
        if "sequence_length" in self.params and "seq_len" not in self.params:
            self.hparams["seq_len"] = self.params["sequence_length"]
        self.hparams["appliances"] = self.params.get(
            "appliances", list(ukdale_appliance_data.keys())
        )

        ckpt = None
        if self.params.get("resume"):
            ckpt = get_latest_checkpoint(self.checkpoint_path)
            if ckpt:
                print(f"Resuming from checkpoint: {ckpt}")
        model = NILMnet(self.hparams)
        print(f"fit model for {file_name}")
        trainer = pl.Trainer(**trainer_kwargs)
        trainer.fit(model, ckpt_path=ckpt)
        trainer.test(model, ckpt_path="best")
        results = getattr(model, "test_results", {})
        clear_output()
        if results:
            print(results.get("app_results", results))
            if "avg_results" in results:
                print("\nAverage metrics (compare maF1 to paper ~0.941):")
                print(results["avg_results"])
        else:
            print("No test results captured.")
        
        
        results_path = f"{self.results_path}{file_name}"
        return results, results_path
        
        
    
   
           
            
    


def run_experiments(model_name="CNN1D", denoise=True,
                     batch_size = 128, epochs = 50,
                    sequence_length =99, sample = None, 
                    dropout = 0.25, data = "ukdale", 
                    benchmark="single-appliance",
                    appliance_id = 0,
                    appliances = ["FRZ"],
                    out_size = 5, quantiles=[0.0025,0.1, 0.5, 0.9, 0.975],
                    data_path="../data/",
                    checkpoint_path="../checkpoints/",
                    results_path="../results/",
                    resume=False):
    exp_name = f"{data}_{model_name}_quantiles" if len(quantiles)>1 else "{data}_{model_name}"
    if benchmark=="single-appliance":
        file_name = f"{exp_name}_single-appliance_{appliances[0]}"
    else:
        file_name = f"{exp_name}_multi-appliance"      
    
    params = {'n_epochs':epochs,'batch_size':batch_size,
                'sequence_length':sequence_length,
                'seq_len': sequence_length if sequence_length % 2 == 0 else sequence_length + 1,
                'model_name':model_name,
                'dropout':dropout,
                'exp_name':exp_name,
                'benchmark':benchmark,
                'clip_value':10,
                'sample':sample,
                'out_size':out_size,
                'appliance_id':appliance_id,
                'appliances':appliances,
                'out_size':len(appliances),
                'data_path':data_path,
                'data':data,
                'quantiles':quantiles,
                "denoise":denoise,
                'file_name':file_name,
                "checkpoint_path": checkpoint_path + file_name + "/",
                "results_path": results_path,
                "resume": resume,
                }
    exp = NILMExperiment(params)
    results, results_path=exp.fit()
   
    return results, results_path

if __name__ == "__main__": 
    sample=None
    epochs=50
    for data in ["ukdale"]:
        for model_name in ["CNN1D", "UNETNiLM"]:
            results = {}
            for idx, app in enumerate(list(ukdale_appliance_data.keys())):
                result, save_path=run_experiments(model_name=model_name, data = data, 
                                sample=sample, epochs=epochs, appliances=[app],
                                appliance_id=idx, benchmark="single-appliance")  
                results[app]=result
            np.save(save_path+"results.npy", results)
            
            
    for data in ["ukdale"]:
        for model_name in ["CNN1D", "UNETNiLM"]:
            results = {}
            result, save_path=run_experiments(model_name=model_name, data = data, 
                                sample=sample, epochs=epochs, appliances=list(ukdale_appliance_data.keys()),
                                appliance_id=None, benchmark="multi-appliance")  
            np.save(save_path+"results.npy", results)                        
            
    
    
