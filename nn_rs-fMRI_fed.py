import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryF1Score, BinaryMatthewsCorrCoef

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import Metrics, Context 
from flwr.simulation import run_simulation
from flwr.client import Client, ClientApp, NumPyClient
from flwr.client.mod import LocalDpMod, fixedclipping_mod, secaggplus_mod
from flwr.server import ServerApp, ServerConfig, ServerAppComponents 
from flwr.server.strategy import DifferentialPrivacyServerSideAdaptiveClipping, DifferentialPrivacyClientSideFixedClipping

from utils import capture_and_store_output, stop_capture_and_restore_output

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features.values, dtype=torch.float32)  
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = self.features[idx].clone().detach().to(dtype=torch.float32)
        label = self.labels[idx].clone().detach().to(dtype=torch.float32)

        return features, label

def fed_BAL_dataloaders(tr_csv, ts_csv):
    tr_df = pd.read_csv(tr_csv, header = None)
    ts_df = pd.read_csv(ts_csv, header = None)

    tr_features = tr_df.iloc[:, 2:]
    tr_labels = tr_df.iloc[:, 0]
    ts_features = ts_df.iloc[:, 2:]
    ts_labels = ts_df.iloc[:, 0]
    pos_count = tr_labels.value_counts().get(1, 0)
    neg_count = tr_labels.value_counts().get(0, 0)
    alpha_focal = pos_count/(pos_count+neg_count)
    class_num_list = [neg_count, pos_count]
  
    input_size = tr_features.shape[1]

    scaler1 = MinMaxScaler()
    norm_tr = pd.DataFrame(scaler1.fit_transform(tr_features), columns=tr_features.columns)
    scaler2 = MinMaxScaler()
    norm_ts = pd.DataFrame(scaler2.fit_transform(ts_features), columns=ts_features.columns)

    tr_dataset = CustomDataset(norm_tr, tr_labels)
    tr_dataloader = DataLoader(tr_dataset, batch_size=BATCH_SIZE, shuffle=True)
    ts_dataset = CustomDataset(norm_ts, ts_labels)
    ts_dataloader = DataLoader(ts_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return input_size, tr_dataloader, ts_dataloader, alpha_focal, class_num_list

def BAL_dataloaders():
    tr1_csv = "data/code_distribution/fed/1_Tokyo/tr_rs-fMRI.csv"
    ts1_csv = "data/code_distribution/fed/1_Tokyo/ts_rs-fMRI.csv"
    input_size, cl1_tr_loader, cl1_ts_loader, cl1_alpha_focal = fed_BAL_dataloaders(tr1_csv, ts1_csv)

    tr2_csv = "data/code_distribution/fed/2_Showa/tr_rs-fMRI.csv"
    ts2_csv = "data/code_distribution/fed/2_Showa/ts_rs-fMRI.csv"
    input_size, cl2_tr_loader, cl2_ts_loader, cl2_alpha_focal = fed_BAL_dataloaders(tr2_csv, ts2_csv)

    tr3_csv = "data/code_distribution/fed/3_Kyoto/tr_rs-fMRI.csv"
    ts3_csv = "data/code_distribution/fed/3_Kyoto/ts_rs-fMRI.csv"
    input_size, cl3_tr_loader, cl3_ts_loader, cl3_alpha_focal = fed_BAL_dataloaders(tr3_csv, ts3_csv)

    tr_loaders = [cl1_tr_loader, cl2_tr_loader, cl3_tr_loader]
    ts_loaders = [cl1_ts_loader, cl2_ts_loader, cl3_ts_loader]
    tr_alpha_focal = [cl1_alpha_focal, cl2_alpha_focal, cl3_alpha_focal]

    return input_size, tr_loaders, ts_loaders, tr_alpha_focal

def BAL_dataloaders_4C():
    tr1_csv = "data/code_distribution/fed/1_Tokyo/tr_rs-fMRI.csv"
    ts1_csv = "data/code_distribution/fed/1_Tokyo/ts_rs-fMRI.csv"
    input_size, cl1_tr_loader, cl1_ts_loader, cl1_alpha_focal, cl1_class_num_list = fed_BAL_dataloaders(tr1_csv, ts1_csv)

    tr2_csv = "data/code_distribution/fed/2_Showa/tr_rs-fMRI.csv"
    ts2_csv = "data/code_distribution/fed/2_Showa/ts_rs-fMRI.csv"
    input_size, cl2_tr_loader, cl2_ts_loader, cl2_alpha_focal, cl2_class_num_list = fed_BAL_dataloaders(tr2_csv, ts2_csv)

    tr3_csv = "data/code_distribution/fed/3_Kyoto/tr_rs-fMRI.csv"
    ts3_csv = "data/code_distribution/fed/3_Kyoto/ts_rs-fMRI.csv"
    input_size, cl3_tr_loader, cl3_ts_loader, cl3_alpha_focal, cl3_class_num_list = fed_BAL_dataloaders(tr3_csv, ts3_csv)

    tr4_csv = "data/code_distribution/fed4/4_KyotoTimTrio/tr_rs-fMRI.csv"
    ts4_csv = "data/code_distribution/fed4/4_KyotoTimTrio/ts_rs-fMRI.csv"
    input_size, cl4_tr_loader, cl4_ts_loader, cl4_alpha_focal, cl4_class_num_list = fed_BAL_dataloaders(tr4_csv, ts4_csv)

    tr_loaders = [cl1_tr_loader, cl2_tr_loader, cl3_tr_loader, cl4_tr_loader]
    ts_loaders = [cl1_ts_loader, cl2_ts_loader, cl3_ts_loader, cl4_ts_loader]
    tr_alpha_focal = [cl1_alpha_focal, cl2_alpha_focal, cl3_alpha_focal, cl4_alpha_focal]
    tr_class_num_list = [cl1_class_num_list, cl2_class_num_list, cl3_class_num_list, cl4_class_num_list]

    return input_size, tr_loaders, ts_loaders, tr_alpha_focal, tr_class_num_list

class BAL_network(nn.Module):
    def __init__(self, input_dim, output_bias=None):
        super(BAL_network, self).__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, 512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

        if output_bias is not None:
            self.fc3.bias.data.fill_(output_bias)

    def forward(self, x):
        x = self.batch_norm(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        x = torch.squeeze(x, dim=1)
        return x

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6, alpha=1.0):
        super().__init__()
        self.smooth = smooth
        self.alpha = alpha  # Weight between Dice and BCE (0.5 = equal)

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        # Dice Loss
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        dice_loss = 1. - dice
        # BCE Loss
        bce_loss = nn.functional.binary_cross_entropy(pred, target, reduction='mean')
        return self.alpha * dice_loss + (1 - self.alpha) * bce_loss
   
def focal_loss(outputs, targets, gamma=2.0, alpha=0.25):
    outputs = torch.clamp(outputs, min=1e-7, max=1-1e-7) # Avoid log(0)
    pt = (targets * outputs) + ((1 - targets) * (1 - outputs))
    at = (targets * alpha) + ((1 - targets) * (1 - alpha))
    focal_weight = at * (1 - pt) ** gamma
    loss = F.binary_cross_entropy(outputs, targets, reduction='mean')
    focal_loss = (focal_weight * loss).mean()
    return focal_loss

def tversky_loss(y_pred, y_true, alpha=0.7, beta=0.3, epsilon=1e-6):
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be non-negative")

    y_true = y_true.float().flatten()
    y_pred = y_pred.float().flatten()

    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)

    true_pos = torch.sum(y_true * y_pred)
    false_pos = torch.sum((1 - y_true) * y_pred)
    false_neg = torch.sum(y_true * (1 - y_pred))

    tversky_index = (true_pos + epsilon) / (
        true_pos + alpha * false_pos + beta * false_neg + epsilon
    )

    return 1 - tversky_index

class DALA(nn.Module):
    def __init__(self, cls_num_list, cls_loss, tau=1, weight=None):
        super(DALA, self).__init__()
        cls_num_list = torch.tensor(cls_num_list, dtype=torch.float, device='cuda')
        cls_p_list = cls_num_list / cls_num_list.sum()
        cls_loss = cls_loss.cuda()
        
        t = cls_p_list / (torch.pow(cls_loss, 0.25)+1e-5)
        m_list = tau * torch.log(t)

        self.m_list = m_list.view(1, -1)
        self.weight = weight

    def forward(self, x, target):
        x_m = x + self.m_list
        return F.cross_entropy(x_m, target, weight=self.weight)

class IntraSCL(nn.Module):
    def __init__(self, cls_num_list, temperature=0.1):
        super(IntraSCL, self).__init__()
        self.temperature = temperature
        self.cls_num_list = torch.tensor(cls_num_list, device='cuda').float()
        self.cls_num_list = self.cls_num_list / self.cls_num_list.sum()

    def forward(self, features, targets):
        device = features.device
        
        if features.dim() == 3:
            features = torch.cat(torch.unbind(features, dim=1), dim=0)
        features = features.view(-1, 256)  # [N, 256]
        
        total_samples = features.shape[0]
        batch_size = total_samples // 2  # Original batch size
        
        targets = targets.contiguous().view(-1).repeat(2)[:total_samples]  
        
        mask = torch.eq(targets.unsqueeze(1), targets.unsqueeze(0)).float()
        
        logits_mask = torch.ones(total_samples, total_samples, device=device) - \
                     torch.eye(total_samples, device=device)

        mask = mask * logits_mask
        
        logits = features.mm(features.T) 
        
        with torch.no_grad():
            weights = self.cls_num_list[targets] 
            temp = torch.sqrt(weights.unsqueeze(0) * weights.unsqueeze(1))
            temp = torch.clamp(temp, min=0.07)
        
        logits = logits / (temp * self.temperature)
        
        logits_max = logits.max(dim=1, keepdim=True)[0].detach()
        logits = logits - logits_max
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        
        return -1 * mean_log_prob_pos.mean()
        
class InterSCL(nn.Module):
    def __init__(self, cls_num_list=None, temperature=0.1):
        super(InterSCL, self).__init__()
        self.temperature = temperature
        self.cls_num_list = cls_num_list

    def forward(self, centers1, features, targets):
        device = features.device  
        
        batch_size = features.shape[0] // 2
        num_centers = centers1.shape[0]

        if features.shape[0] != batch_size * 2: 
            features = features[:batch_size * 2]
            targets = targets[:batch_size * 2]

        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        features = features.view(-1, 256)

        targets = targets.contiguous().view(-1, 1)
        targets_centers = torch.arange(num_centers, device=device).view(-1, 1)
        all_targets = torch.cat([targets.repeat(2, 1), targets_centers], dim=0)
        
        mask = torch.eq(targets[:2*batch_size], targets.T).float().to(device)
        
        logits_mask = torch.ones(2*batch_size, 2*batch_size + num_centers, device=device)
        
        indices = torch.arange(2*batch_size, device=device).view(-1, 1)
        
        logits_mask.scatter_(1, indices, 0)
        logits_mask[:, 2*batch_size:] = 0  # Mask out centers
        
        all_features = torch.cat([features, centers1], dim=0)

        logits = features.mm(all_features.T) / self.temperature
        
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        mean_log_prob_pos = (mask * log_prob[:, :2*batch_size]).sum(1) / (mask.sum(1) + 1e-8)

        loss = -1 * mean_log_prob_pos

        loss = loss.view(2, batch_size).mean()
        
        return loss
     
def iic_loss(features, prototypes, logits, labels, class_num_list, loss_class, k1=1.0, k2=1.0):
    prototypes = F.normalize(prototypes, dim=1).detach().clone()

    intra_cl_criterion = IntraSCL(cls_num_list=class_num_list)
    inter_cl_criterion = InterSCL(cls_num_list=class_num_list)
    ce_criterion = DALA(cls_num_list=class_num_list, cls_loss=loss_class.detach())
    
    with torch.no_grad():
        loss_ce = ce_criterion(logits.detach(), labels)  # Detach if not needed for gradients
    loss_cl_inter = inter_cl_criterion(prototypes, features, labels)
    loss_cl_intra = intra_cl_criterion(features, labels)

    loss = 0.5*loss_ce + k1*loss_cl_intra + k2*loss_cl_inter
    
    return loss

def train_iic(model, class_num_list, loader, resume_epoch=0):
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    model.train()

    len_class_num_list = len(class_num_list)
    cls_loss = torch.zeros(len_class_num_list, device=DEVICE)
    cls_count = torch.zeros(len_class_num_list, device=DEVICE)
    prototypes = torch.zeros(len_class_num_list, 256, device=DEVICE)
    prototype_counts = torch.zeros(len_class_num_list, device=DEVICE)

    for epoch in range(resume_epoch, resume_epoch+LOCAL_EPOCHS):
        all_preds = torch.tensor([], dtype=float, device=DEVICE)
        all_targets = torch.tensor([],dtype=float,device=DEVICE)
        total_loss = 0.0

        for features, targets in loader:
            optimizer.zero_grad()
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            
            with torch.set_grad_enabled(True):
                x = model.batch_norm(features)
                x = F.relu(model.fc1(x))
                x = model.dropout1(x)
                features_256 = F.relu(model.fc2(x))  
                x = model.dropout2(features_256)
                logits = model.fc3(x) 
                preds = torch.sigmoid(logits).squeeze(dim=1) 
            
            with torch.no_grad():
                preds_clamped = torch.clamp(preds, 1e-7, 1-1e-7)
                loss_per_sample = F.binary_cross_entropy(preds_clamped, targets, reduction='none')
                
                for cls_idx in range(len(class_num_list)):
                    mask = (targets == cls_idx)
                    if mask.any():
                        cls_loss[cls_idx] += loss_per_sample[mask].sum()
                        cls_count[cls_idx] += mask.sum()
                        
                        prototypes[cls_idx] += features_256[mask].sum(dim=0)
                        prototype_counts[cls_idx] += mask.sum()
            
            with torch.no_grad():
                current_prototypes = prototypes / (prototype_counts.unsqueeze(1) + 1e-8)
                current_cls_loss = cls_loss / (cls_count + 1e-8)
            
            loss = iic_loss(
                features=features_256,
                prototypes=current_prototypes,
                logits=logits,  # Pass raw logits here
                labels=targets.long(),
                class_num_list=class_num_list,
                loss_class=current_cls_loss
            )
           
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            with torch.no_grad():
                total_loss += loss.item()
                all_preds = torch.cat((all_preds, preds.to(DEVICE)),dim=0)
                all_targets = torch.cat((all_targets, targets),dim=0)
        
        with torch.no_grad():
            if len(all_preds) > 0 and len(all_targets) > 0:
                acc, auc, f1, mcc, conf_matrix = performance(all_targets, all_preds)
                print(conf_matrix)
                
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch+1
                }

    return total_loss/len(loader), acc, auc, f1, mcc

def performance(targets, predictions):
    num_classes = 2
    pred_labels = (predictions >= 0.5).int().to(DEVICE)
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int32, device= DEVICE)
    for t, p in zip(targets.int(), pred_labels):
        conf_matrix[t, p] += 1

    auroc = BinaryAUROC()
    auroc = auroc.to(DEVICE)
    auc_score = auroc(predictions, targets)

    acc = BinaryAccuracy()
    acc = acc.to(DEVICE)
    acc_score = acc(predictions, targets)

    f1 = BinaryF1Score()
    f1 = f1.to(DEVICE)
    f1_score = f1(predictions, targets)

    mcc = BinaryMatthewsCorrCoef()
    mcc = mcc.to(DEVICE)
    mcc_score = mcc(predictions, targets)

    return acc_score, auc_score, f1_score, mcc_score, conf_matrix

def train(model, loader, criterion):  
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    # criterion = lambda outputs, targets: focal_loss(outputs, targets, gamma=2.0, alpha=0.3)
    # criterion = DiceBCELoss()
    model.train()
    for epoch in range(LOCAL_EPOCHS):
        all_preds = torch.tensor([], dtype=float, device=DEVICE)
        all_targets = torch.tensor([],dtype=float,device=DEVICE)
        total_loss = 0.0
        
        for features, targets in loader:
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            preds = model(features)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_preds = torch.cat((all_preds, preds.to(DEVICE)),dim=0)
            all_targets = torch.cat((all_targets, targets),dim=0)
        
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch+1
        }
        # torch.save(state, os.path.join(STORE_DIR, f"train-{epoch}.pth"))

        acc, auc, f1, mcc, conf_matrix = performance(all_targets, all_preds)
        print("in train:")
        print(conf_matrix)

    return total_loss/len(loader), acc, auc, f1, mcc

def eval(model, loader):
    total_loss = 0.0
    all_preds = torch.tensor([], dtype=float, device=DEVICE)
    all_targets = torch.tensor([],dtype=float,device=DEVICE)
    criterion = lambda outputs, targets: tversky_loss(outputs, targets)
    # criterion = lambda outputs, targets: focal_loss(outputs, targets, gamma=2.0, alpha=0.2)
    # criterion = DiceBCELoss()
    model.eval()
    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            preds = model(imgs)
            loss = criterion(preds, targets)

            total_loss += loss.item()
            all_preds = torch.cat((all_preds, preds.to(DEVICE)),dim=0)
            all_targets = torch.cat((all_targets, targets),dim=0)
    
    total_loss = total_loss/len(loader)
    acc, auc, f1, mcc, conf_matrix = performance(all_targets, all_preds)
    
    print("in eval:")
    print(conf_matrix)
    return total_loss, acc, auc, f1, mcc  

def get_parameters(model) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters: List[np.ndarray]):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def save_model(net, server_round):
    state = {'model': net.state_dict(),
             'round': server_round}
    torch.save(state, os.path.join(STORE_DIR, f'train-{server_round}.pth'))

def server_side_evaluation(server_round: int, parameters: fl.common.NDArrays, 
                config: Dict[str, fl.common.Scalar],) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    set_parameters(model, parameters)
    save_model(model, server_round)
 
class schizClient(NumPyClient):
    def __init__(self, cid, net, trainloader, valloader, alpha_focal, class_num_list):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.alpha_focal = alpha_focal
        self.class_num_list = class_num_list
        self.criterion = None

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]
        print(f"[Client {self.cid}, round {server_round}] fit, config: {config}")

        self.set_parameters(parameters)
        print("Training Started...")
        # self.criterion = lambda outputs, targets: focal_loss(outputs, targets, gamma=2.0, alpha=self.alpha_focal)
        # alphaa = (5*self.alpha_focal/6) + (17/30)
        # betaa = 1 - alphaa
        # if self.alpha_focal < 0.25:
        #     alpha, beta = 0.7, 0.3
        # else:
        #     alpha, beta = 0.8, 0.2
        # alpha, beta = 0.2, 0.8
        # self.criterion = lambda outputs, targets: tversky_loss(outputs, targets, alpha= self.alpha_focal, beta= 1-self.alpha_focal)
        self.criterion = lambda outputs, targets: tversky_loss(outputs, targets, alpha= 1-self.alpha_focal, beta= self.alpha_focal)
        # self.criterion = lambda outputs, targets: tversky_loss(outputs, targets)
        # self.criterion = nn.BCELoss()
        # self.criterion = DiceBCELoss()
        loss, acc, auc, f1, mcc = train(self.net, self.trainloader, self.criterion)
        # loss, acc, auc, f1, mcc = train_iic(self.net, self.class_num_list, self.trainloader)
        print("Training Finished.")
        return self.get_parameters(config={}), len(self.trainloader), {"accuracy": float(acc), "loss": float(loss), "auc": float(auc), "f1":float(f1), "mcc":float(mcc), "client train": float(self.cid)}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        self.set_parameters(parameters)
        loss, acc, auc, f1, mcc = eval(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(acc), "loss": float(loss), "auc": float(auc), "f1":float(f1), "mcc":float(mcc), "client evaluate": float(self.cid)}
    
def client_fn(context: Context) -> Client:
    cid = context.node_config["partition-id"]
    return schizClient(cid, model, tr_dataloader[int(cid)], ts_dataloader[int(cid)], tr_alpha_focal[int(cid)],  tr_class_num_list[int(cid)]).to_client()
 
def weighted_fit_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]

    examples = [num_examples for num_examples, _ in metrics]

    num = sum(examples)
    acc = sum(accuracies) / num
    loss = sum(losses) / num

    RESULTS['agg_tr_accuracy_loss'].append([acc, loss])

    return {"accuracy": acc, "loss": loss, "which?":  "fit"}

def fit_config(server_round: int):
    config = {
        "server_round": server_round,  # The current round of federated learning
        "local_epochs": 1 if server_round < 2 else LOCAL_EPOCHS,  #
    }
    return config

def server_fn(context: Context) -> ServerAppComponents:
    #https://github.com/adap/flower/tree/main/src/py/flwr/server/strategy
    strategy = fl.server.strategy.FedAvg(       #FedAvg, QFedAvg, FedAdagrad, FedProx, FedYogi, FedOpt, FedAdam, FedAvgM
        # proximal_mu = 0.5,  #Uncomment in FedProx
        # server_momentum = 0.9, #Uncomment in FedAvgM
        # server_learning_rate = 1.0,
        # q_param = 0.2,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        # min_fit_clients = 3, 
        # min_evaluate_clients = 3, 
        # min_available_clients = 3,
        #fit_metrics_aggregation_fn = weighted_fit_average,
        initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(model)),
        on_fit_config_fn=fit_config,
        evaluate_fn=server_side_evaluation
    )

    config = ServerConfig(num_rounds= NUM_ROUNDS)

    return ServerAppComponents(strategy=strategy, config=config)
 
def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ['TOKENIZERS_PARALLELISM']= 'false'
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
    torch.cuda.empty_cache() 
    warnings.simplefilter('ignore')
    #cudnn stuff
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    global tr_dataloader, ts_dataloader, tr_alpha_focal, tr_class_num_list, NUM_ROUNDS, LOCAL_EPOCHS, AGG_TR_LOSSES
    global RESULTS, LR, model, STORE_DIR, DEVICE, BATCH_SIZE, folder_name
    global WEIGHT_DECAY, DECAY_EPOCH, DECAY_RATIO

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device is {DEVICE}")

    LR = 0.001
    NUM_CLIENTS = 4
    NUM_ROUNDS = 100
    LOCAL_EPOCHS = 1
    BATCH_SIZE = 64
    folder_name = "folder_name"
    resume_dir = ""
    saved_model = ""
    capture_and_store_output(folder_name+".txt")

    STORE_DIR = os.path.join('results', folder_name + " " + datetime.today().strftime('%Y-%m-%d-%H-%M-%S')) 
    os.mkdir(STORE_DIR)
    STORE_DIR = os.path.join(STORE_DIR, datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
    os.mkdir(STORE_DIR)
    RESULTS = {
                'total_rounds': NUM_ROUNDS,
                'batch_size':BATCH_SIZE,
            }

    input_size, tr_dataloader, ts_dataloader, tr_alpha_focal, tr_class_num_list = BAL_dataloaders_4C()

    initial_bias_tensor = torch.tensor( np.log(0.25), dtype=torch.float32)
    model = BAL_network(input_size, output_bias=initial_bias_tensor)
    model = model.to(DEVICE)
    if DEVICE == 'cuda':
        model = torch.nn.DataParallel(model).cuda()

    client = ClientApp(client_fn=client_fn)#, mods=[fixedclipping_mod,],) #, mods=[local_dp_obj])
    server = ServerApp(server_fn=server_fn)

    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}
    if DEVICE.type == "cuda":
        backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}

    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config,
    )

    stop_capture_and_restore_output()

if __name__ == "__main__":
    main()
