import pytorch_lightning as pl
from tqdm import tqdm
import numpy as np
import sys
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from .modules import CNN1DModel,  UNETNiLM
from net.metrics import  compute_metrics, compute_regress_metrics, get_results_summary
from data.load_data import ukdale_appliance_data 
from data.data_loader import Dataset, load_data, spilit_refit_test
from .utils import ObjectDict, QuantileLoss

from sklearn.metrics import f1_score as sklearn_f1_score


def _batch_f1(pred, target, num_classes=2):
    """F1 for predictions and labels shaped (B, M)."""
    pred = pred.detach().cpu().numpy().reshape(-1)
    target = target.detach().cpu().numpy().reshape(-1)
    if pred.size == 0:
        return torch.tensor(0.0)
    score = sklearn_f1_score(target, pred, average="micro", zero_division=0)
    return torch.tensor(float(score))


class NILMnet(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hp = ObjectDict()
        self.hp.update(hparams.__dict__ if hasattr(hparams, "__dict__") else hparams)
        self._data = None
        self._test_outputs = []
        self.q_criterion = QuantileLoss(self.hp.quantiles)
        if self.hp.model_name== "CNN1D":
            self.model = CNN1DModel(in_size=self.hp.in_size, 
                               output_size=self.hp.out_size,
                               d_model=self.hp.d_model,
                                dropout=self.hp.dropout, 
                               seq_len=self.hp.seq_len,  
                               n_layers=self.hp.n_layers, 
                               n_quantiles=len(self.hp.quantiles),
                               pool_filter=self.hp.pool_filter)
        
        elif self.hp.model_name=="UNETNiLM":
            self.model =  UNETNiLM(in_size=self.hp.in_size, 
                               output_size=self.hp.out_size,
                               features_start=self.hp.d_model//4,
                               seq_len=self.hp.seq_len, 
                               n_layers=self.hp.n_layers,
                               n_quantiles=len(self.hp.quantiles),
                               pool_filter=self.hp.d_model//4
                               )  
        
         
    def forward(self, x):
            return self.model(x)        
        
    def _step(self, batch):
        x, y, z = batch
        if self.hp.benchmark=="single-appliance":
            y = y.unsqueeze(-1)
            z = z.unsqueeze(-1)
        B = x.size(0)
        logits, rmse_logits = self(x)
        logits_ce = logits.permute(0, 2, 1).reshape(-1, 2)
        z_flat = z.reshape(-1)
        prob, pred = torch.max(F.softmax(logits, dim=1), dim=1)
        loss_nll = F.cross_entropy(logits_ce, z_flat)
        if len(self.hp.quantiles)>1:
            prob=prob.unsqueeze(1).expand_as(rmse_logits)
            loss_mse = self.q_criterion(rmse_logits, y)
            mae_score = F.l1_loss(rmse_logits,y.unsqueeze(1).expand_as(rmse_logits))
        else:    
            loss_mse = F.mse_loss(rmse_logits, y)
            mae_score = F.l1_loss(rmse_logits, y)
            
        loss = loss_nll + loss_mse
        
        res = _batch_f1(pred, z)
        logs = {"nlloss":loss_nll, "mseloss":loss_mse,
                 "mae":mae_score, "F1": res}
        return loss, logs
    
    def training_step(self, batch, batch_idx):
        loss, logs = self._step(batch)
        self.log("loss", loss, prog_bar=True)
        for key, value in logs.items():
            self.log(f"tra_{key}", value.item(), prog_bar=(key == "F1"))
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self._step(batch)
        for key, value in logs.items():
            self.log(f"val_{key}", value.item(), on_epoch=True, prog_bar=(key == "F1"))
        return logs

    def on_validation_epoch_end(self):
        pass
    
    def test_step(self, batch,batch_idx):
        x, y, z = batch
        B = x.size(0)
        if self.hp.benchmark=="single-appliance":
            y = y.unsqueeze(-1)
            z = z.unsqueeze(-1)
        logits, pred_power  = self(x)
        
        prob, pred_state = torch.max(F.softmax(logits, dim=1), dim=1)
        if len(self.hp.quantiles) > 1:
            prob = prob.unsqueeze(1).expand_as(pred_power)

        out = {"pred_power": pred_power, "pred_state": pred_state, "power": y, "state": z}
        self._test_outputs.append(out)
        return out

    def on_test_epoch_end(self):
        outputs = self._test_outputs
        self._test_outputs = []
        if not outputs:
            self.test_results = {}
            return
        
        appliance_data = ukdale_appliance_data
        pred_power = torch.cat([x['pred_power'] for x in outputs], 0).cpu().numpy()
        pred_state = torch.cat([x['pred_state'] for x in outputs], 0).cpu().numpy().astype(np.int32)
        power = torch.cat([x['power'] for x in outputs], 0).cpu().numpy()
        state = torch.cat([x['state'] for x in outputs], 0).cpu().numpy().astype(np.int32)
        
        for idx, app in enumerate(self.hp.appliances):
            mean = appliance_data[app]["mean"]
            std = appliance_data[app]["std"]
            power[:, idx] = (power[:, idx] * std) + mean
            if len(self.hp.quantiles) >= 2:
                pred_power[:, :, idx] = (pred_power[:, :, idx] * std) + mean
                pred_power[:, :, idx] = np.where(pred_power[:, :, idx] < 0, 0, pred_power[:, :, idx])
            else:
                pred_power[:, idx] = (pred_power[:, idx] * std) + mean
                pred_power[:, idx] = np.where(pred_power[:, idx] < 0, 0, pred_power[:, idx])
        
        if len(self.hp.quantiles)>=2:
            idx = len(self.hp.quantiles)//2
            y_pred = pred_power[:,idx]
        else:
            y_pred = pred_power 
               
        per_app_results, avg_results = get_results_summary(
            state, pred_state, power, y_pred, self.hp.appliances, self.hp.data
        )
        self.test_results = {
            "pred_power": pred_power,
            "pred_state": pred_state,
            "power": power,
            "state": state,
            "app_results": per_app_results,
            "avg_results": avg_results,
        }
        return self.test_results
        
    def predict(self, model, dataloader):
        outputs = []
        model = model.eval()
        batch_size   = dataloader.batchsize if hasattr(dataloader, 'len') else dataloader.batch_size
        num_batches = len(dataloader)
        values = range(num_batches)
        with tqdm(total=len(values), file=sys.stdout) as pbar:
             with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    logs = self.test_step(batch, batch_idx, model)
                    outputs.append(logs)
                    del  batch
                    pbar.set_description('processed: %d' % (1 + batch_idx))
                    pbar.update(1)
                pbar.close()
        outputs = self.on_test_epoch_end(outputs)
        return outputs
        
        
    def configure_optimizers(self):  
        optim = torch.optim.Adam(self.parameters(),lr=self.hp.learning_rate, betas=(self.hp.beta_1, self.hp.beta_2))
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, patience=self.hp.patience_scheduler, min_lr=1e-6, mode="max"
        )
        scheduler = {'scheduler':sched, 
                     'monitor': 'val_F1',
                     'interval': 'epoch',
                     'frequency': 1}
        return [optim], [scheduler]
        
    
   
    def train_dataloader(self):
        
        data = Dataset(self._get_cache_data()['x_train'], self._get_cache_data()['y_train'], 
                             self._get_cache_data()['z_train'],  seq_len=self.hp.seq_len)
        return torch.utils.data.DataLoader(data,batch_size=self.hp.batch_size,
                                            shuffle=True,pin_memory=True,
                                            num_workers=self.hp.num_workers)
        
    
    def val_dataloader(self):
        
        data = Dataset(self._get_cache_data()['x_val'], self._get_cache_data()['y_val'], 
                             self._get_cache_data()['z_val'],  seq_len=self.hp.seq_len)
        return torch.utils.data.DataLoader(data,batch_size=self.hp.batch_size,
                                            shuffle=False,pin_memory=True,
                                            num_workers=self.hp.num_workers)  
       
    def test_dataloader(self):
        
        data = Dataset(self._get_cache_data()['x_test'], self._get_cache_data()['y_test'], 
                             self._get_cache_data()['z_test'],  seq_len=self.hp.seq_len)
        return torch.utils.data.DataLoader(data,batch_size=self.hp.batch_size,
                                            shuffle=False,pin_memory=True,
                                            num_workers=self.hp.num_workers)        
    
    def _get_cache_data(self):
        if self._data is None:
            x , y , z = load_data(data_path=self.hp.data_path, 
                                                    data_type="test" if self.hp.data=="refit" else "training", 
                                                    sample=self.hp.sample,
                                                     data=self.hp.data,
                                                    denoise=self.hp.denoise) 
            x_train, x_val, x_test = spilit_refit_test(x)
            if self.hp.benchmark=="single-appliance":
                y_train, y_val, y_test = spilit_refit_test(y[:,self.hp.appliance_id][:,None])
                #print(y_train.shape)
                z_train, z_val, z_test = spilit_refit_test(z[:,self.hp.appliance_id][:,None])
            else:       
                y_train, y_val, y_test = spilit_refit_test(y)
                z_train, z_val, z_test = spilit_refit_test(z) 
           
           
            self._data = dict(x_test=x_test, y_test=y_test, z_test=z_test,
                              x_val=x_val, y_val=y_val, z_val=z_val,
                              x_train=x_train, y_train=y_train, z_train=z_train)
            
        return self._data   
    
    
    @staticmethod
    def add_model_specific_args():
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(add_help=False)
        parser.add_argument('--learning_rate', default=1e-4, type=float)
        parser.add_argument('--batch_size', default=4, type=int)
        parser.add_argument('--momentum', default=0.9, type=float)
        parser.add_argument('--beta_1', default=0.999, type=float)
        parser.add_argument('--beta_2', default= 0.98, type=float)
        parser.add_argument('--eps', default=1e-8, type=float)
        parser.add_argument('--patience_scheduler', default=5, type=int)
        parser.add_argument('--weight_decay', default=0.0005, type=float)
        parser.add_argument('--dropout', default=0.25, type=float)
        parser.add_argument('--d_model', default=128, type=int)
        parser.add_argument('--pool_filter', default=8, type=int)
        parser.add_argument('--n_layers', default=5, type=int)
        parser.add_argument('--seq_len', default=100, type=int)
        parser.add_argument('--out_size', default=5, type=int)
        parser.add_argument('--in_size', default=1, type=int)
        parser.add_argument('--denoise', default=False, type=bool)
        parser.add_argument('--num_head', default=8, type=int)
        parser.add_argument('--model_name', default="CNN1D", type=str)
        parser.add_argument('--benchmark', default="Seq2Point", type=str)
        parser.add_argument('--appliance_id', default=0, type=int)
        parser.add_argument('--appliances', default=list(ukdale_appliance_data.keys()), type=list)
        parser.add_argument('--data', default="ukdale", type=str)
        parser.add_argument('--quantiles', default=[0.0025,0.1, 0.5, 0.9, 0.975], type=list)
        parser.add_argument('--num_workers', default=4, type=int)
        #parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
        return parser
