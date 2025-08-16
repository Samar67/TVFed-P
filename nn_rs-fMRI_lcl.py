import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryF1Score, BinaryMatthewsCorrCoef

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

def BAL_dataloaders(cl_num):
    if(cl_num == 0):
        tr_csv = "data/cnt/tr_cnt_rs-fMRI.csv"
        ts_csv = "data/cnt/ts_cnt_rs-fMRI.csv"    
    elif(cl_num == 1):
        tr_csv = "data/code_distribution/fed/1_Tokyo/tr_rs-fMRI.csv"
        ts_csv = "data/code_distribution/fed/1_Tokyo/ts_rs-fMRI.csv"
    elif(cl_num == 2):
        tr_csv = "data/code_distribution/fed/2_Showa/tr_rs-fMRI.csv"
        ts_csv = "data/code_distribution/fed/2_Showa/ts_rs-fMRI.csv"
    elif(cl_num == 3):
        tr_csv = "data/code_distribution/fed/3_Kyoto/tr_rs-fMRI.csv"
        ts_csv = "data/code_distribution/fed/3_Kyoto/ts_rs-fMRI.csv"

    tr_df = pd.read_csv(tr_csv, header = None)
    ts_df = pd.read_csv(ts_csv, header = None)

    tr_features = tr_df.iloc[:, 2:]
    tr_labels = tr_df.iloc[:, 0]
    ts_features = ts_df.iloc[:, 2:]
    ts_labels = ts_df.iloc[:, 0]
    pos_count = tr_labels.value_counts().get(1, 0)
    neg_count = tr_labels.value_counts().get(0, 0)

    input_size = tr_features.shape[1]

    scaler1 = MinMaxScaler()
    norm_tr = pd.DataFrame(scaler1.fit_transform(tr_features), columns=tr_features.columns)
    scaler2 = MinMaxScaler()
    norm_ts = pd.DataFrame(scaler2.fit_transform(ts_features), columns=ts_features.columns)

    tr_dataset = CustomDataset(norm_tr, tr_labels)
    tr_dataloader = DataLoader(tr_dataset, batch_size=BATCH_SIZE, shuffle=True)
    ts_dataset = CustomDataset(norm_ts, ts_labels)
    ts_dataloader = DataLoader(ts_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return input_size, pos_count, neg_count, (norm_tr, tr_labels, norm_ts, ts_labels), tr_dataloader, ts_dataloader

def BAL_dataloaders_4C(cl_num):
    if(cl_num == 0):
        tr_csv = "data/cnt/tr_cnt_rs-fMRI.csv"
        ts_csv = "data/cnt/ts_cnt_rs-fMRI.csv"    
    elif(cl_num == 1):
        tr_csv = "data/code_distribution/fed4/1_Tokyo/tr_rs-fMRI.csv"
        ts_csv = "data/code_distribution/fed4/1_Tokyo/ts_rs-fMRI.csv"
    elif(cl_num == 2):
        tr_csv = "data/code_distribution/fed4/2_Showa/tr_rs-fMRI.csv"
        ts_csv = "data/code_distribution/fed4/2_Showa/ts_rs-fMRI.csv"
    elif(cl_num == 3):
        tr_csv = "data/code_distribution/fed4/3_KyotoTrio/tr_rs-fMRI.csv"
        ts_csv = "data/code_distribution/fed4/3_KyotoTrio/ts_rs-fMRI.csv"
    elif(cl_num == 4):
        tr_csv = "data/code_distribution/fed4/4_KyotoTimTrio/tr_rs-fMRI.csv"
        ts_csv = "data/code_distribution/fed4/4_KyotoTimTrio/ts_rs-fMRI.csv"

    tr_df = pd.read_csv(tr_csv, header = None)
    ts_df = pd.read_csv(ts_csv, header = None)

    tr_features = tr_df.iloc[:, 2:]
    tr_labels = tr_df.iloc[:, 0]
    ts_features = ts_df.iloc[:, 2:]
    ts_labels = ts_df.iloc[:, 0]
    pos_count = tr_labels.value_counts().get(1, 0)
    neg_count = tr_labels.value_counts().get(0, 0)

    input_size = tr_features.shape[1]

    scaler1 = MinMaxScaler()
    norm_tr = pd.DataFrame(scaler1.fit_transform(tr_features), columns=tr_features.columns)
    scaler2 = MinMaxScaler()
    norm_ts = pd.DataFrame(scaler2.fit_transform(ts_features), columns=ts_features.columns)

    tr_dataset = CustomDataset(norm_tr, tr_labels)
    tr_dataloader = DataLoader(tr_dataset, batch_size=BATCH_SIZE, shuffle=True)
    ts_dataset = CustomDataset(norm_ts, ts_labels)
    ts_dataloader = DataLoader(ts_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return input_size, pos_count, neg_count, (norm_tr, tr_labels, norm_ts, ts_labels), tr_dataloader, ts_dataloader

class ImbalancedNN_512(nn.Module): ##a7san wa7da fihom ##got 0.7164 f1 and 0.8794 auc epoch 14
    def __init__(self, input_dim, output_bias=None):
        super(ImbalancedNN_512, self).__init__()
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

def weighted_binary_cross_entropy(outputs, targets, pos_weight):
    loss = F.binary_cross_entropy(outputs, targets, reduction='none')
    weights = torch.ones_like(targets)
    weights[targets == 1] = pos_weight
    weighted_loss = (loss * weights).mean()
    return weighted_loss

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6, alpha=1.0):
        super().__init__()
        self.smooth = smooth
        self.alpha = alpha

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

def tversky_loss(y_pred, y_true, alpha=0.3, beta=0.7, epsilon=1e-6):
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

def model_build(mode, pos_count, neg_count, input_size):
    #cudnn stuff
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    total = pos_count + neg_count
    # Calculate initial bias for better starting point
    # initial_bias = np.log(pos_count / neg_count)
    initial_bias =  np.log(0.25)
    initial_bias_tensor = torch.tensor(initial_bias, dtype=torch.float32)

    # Create the model
    model = ImbalancedNN_512(input_size, output_bias=initial_bias_tensor)
    model = model.to(DEVICE)
    if DEVICE == 'cuda':
        model = torch.nn.DataParallel(model).cuda()

    # Calculate positive weight for weighted loss
    # pos_weight = torch.tensor(neg_count / pos_count, dtype=torch.float32) # in case weighted binary cross entrpy were used
    alpha_focal = pos_count / total
    # alpha_focal = 0.3
    # criterion = lambda outputs, targets: weighted_binary_cross_entropy(outputs, targets, pos_weight)
    # criterion = lambda outputs, targets: focal_loss(outputs, targets, gamma=2.0, alpha=alpha_focal)
    # criterion = lambda outputs, targets: focal_loss(outputs, targets, gamma=1.0, alpha=alpha_focal)
    # criterion = DiceBCELoss()
    # alpha = (5*alpha_focal/6) + (17/30)
    # beta = 1 - alpha
    # if alpha_focal < 0.25:
    #     alpha, beta = 0.7, 0.3
    # else:
    #     alpha, beta = 0.8, 0.2
    # criterion = lambda outputs, targets: tversky_loss(outputs, targets, alpha= 0.2, beta= 0.8)
    criterion = lambda outputs, targets: tversky_loss(outputs, targets, alpha= 1-alpha_focal, beta= alpha_focal)
    # criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4) # Added weight decay for L2 regularization effect

    store_dir = ""
    if mode == "train" or mode == "train_iic":
        store_dir = os.path.join('results', folder_name + " " + datetime.today().strftime('%Y-%m-%d-%H-%M-%S')) 
        os.mkdir(store_dir)
        store_dir = os.path.join(store_dir, datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
        os.mkdir(store_dir)

    return model, optimizer, criterion, 0, store_dir

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
        
        targets = targets.contiguous().view(-1).repeat(2)[:total_samples]  # Handle potential truncation
        
        mask = torch.eq(targets.unsqueeze(1), targets.unsqueeze(0)).float()
        
        logits_mask = torch.ones(total_samples, total_samples, device=device) - \
                     torch.eye(total_samples, device=device)
        
        mask = mask * logits_mask
        
        logits = features.mm(features.T)  # [total_samples, total_samples]
        
        with torch.no_grad():
            weights = self.cls_num_list[targets]  # [total_samples]
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
        device = features.device  # Get device from input tensor
        
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
    ce_criterion = DALA(cls_num_list=class_num_list, cls_loss=loss_class.detach())  # Detach loss_class
    
    with torch.no_grad():
        loss_ce = ce_criterion(logits.detach(), labels)  # Detach if not needed for gradients
    loss_cl_inter = inter_cl_criterion(prototypes, features, labels)
    loss_cl_intra = intra_cl_criterion(features, labels)

    loss = 0.5*loss_ce + k1*loss_cl_intra + k2*loss_cl_inter
    
    return loss

def train_iic(model, optimizer, criterion, class_num_list, loader, store_dir, resume_epoch=0):
    model.train()

    len_class_num_list = len(class_num_list)
    cls_loss = torch.zeros(len_class_num_list, device=DEVICE)
    cls_count = torch.zeros(len_class_num_list, device=DEVICE)
    prototypes = torch.zeros(len_class_num_list, 256, device=DEVICE)
    prototype_counts = torch.zeros(len_class_num_list, device=DEVICE)

    for epoch in range(resume_epoch, resume_epoch+EPOCHS):
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
                features_256 = F.relu(model.fc2(x))  # Features for prototypes
                x = model.dropout2(features_256)
                logits = model.fc3(x)  # Get raw logits before sigmoid
                preds = torch.sigmoid(logits).squeeze(dim=1)  # Apply sigmoid here
            
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
                logits=logits, 
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
                
                print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(loader):.4f}, "
                      f"Acc: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}")
                print(conf_matrix)
                
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch+1
                }
                torch.save(state, os.path.join(store_dir, f"train-{epoch}.pth"))

def eval_iic(model, criterion, loader, class_num_list):
    len_class_num_list = len(class_num_list)
    cls_loss = torch.zeros(len_class_num_list, device=DEVICE)
    cls_count = torch.zeros(len_class_num_list, device=DEVICE)
    prototypes = torch.zeros(len_class_num_list, 256, device=DEVICE)
    prototype_counts = torch.zeros(len_class_num_list, device=DEVICE)

    total_loss = 0.0
    all_preds = torch.tensor([], dtype=float, device=DEVICE)
    all_targets = torch.tensor([],dtype=float,device=DEVICE)
    # criterion = nn.BCELoss()
    model.eval()
    with torch.no_grad():
        for features, targets in loader:
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            x = model.batch_norm(features)
            x = F.relu(model.fc1(x))
            x = model.dropout1(x)
            features_256 = F.relu(model.fc2(x))  
            x = model.dropout2(features_256)
            logits = model.fc3(x)
            preds = torch.sigmoid(logits).squeeze(dim=1)
            
            loss_per_sample = F.binary_cross_entropy(preds, targets, reduction='none')
            
            for cls_idx in range(len(class_num_list)):
                mask = (targets == cls_idx)
                if mask.any():
                    cls_loss[cls_idx] += loss_per_sample[mask].sum()
                    cls_count[cls_idx] += mask.sum()
                    
                    prototypes[cls_idx] += features_256[mask].sum(dim=0)
                    prototype_counts[cls_idx] += mask.sum()
            
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
            
            total_loss += loss.item()
            all_preds = torch.cat((all_preds, preds.to(DEVICE)),dim=0)
            all_targets = torch.cat((all_targets, targets),dim=0)
    
    total_loss = total_loss/len(loader)
    acc, auc, f1, mcc, conf_matrix = performance(all_targets, all_preds)
    print(conf_matrix)

    return total_loss, acc, f1, auc, mcc       

def train(model, optimizer, criterion, loader, store_dir, resume_epoch=0):  
    model.train()  
    for epoch in range(resume_epoch, resume_epoch+EPOCHS):
        all_preds = torch.tensor([], dtype=float, device=DEVICE)
        all_targets = torch.tensor([],dtype=float,device=DEVICE)
        total_loss = 0.0
        for features, targets in loader:
            optimizer.zero_grad()
            features, targets = features.to(DEVICE), targets.to(DEVICE)
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
        torch.save(state, os.path.join(store_dir, f"train-{epoch}.pth"))

        acc, auc, f1, mcc, conf_matrix = performance(all_targets, all_preds)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(loader)}, acc: {acc}, auc: {auc}, f1: {f1}, mcc: {mcc}")
        print(conf_matrix)

def eval(model, criterion, loader):
    total_loss = 0.0
    all_preds = torch.tensor([], dtype=float, device=DEVICE)
    all_targets = torch.tensor([],dtype=float,device=DEVICE)
    model.eval()
    with torch.no_grad():
        for features, targets in loader:
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            preds = model(features)
            loss = criterion(preds, targets)
            total_loss += loss.item()
            all_preds = torch.cat((all_preds, preds.to(DEVICE)),dim=0)
            all_targets = torch.cat((all_targets, targets),dim=0)
    
    total_loss = total_loss/len(loader)
    acc, auc, f1, mcc, conf_matrix = performance(all_targets, all_preds)
    print(conf_matrix)

    return total_loss, acc, f1, auc, mcc       
            
def main():
    global BATCH_SIZE, LR, EPOCHS, DEVICE
    global criterion, optimizer
    global folder_name

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    EPOCHS = 100
    LR = 0.001
    BATCH_SIZE = 64
    folder_name = "ts_cl1_4Cs_512nn_rsFMRI_focal_alpEqPosOtot"
    capture_and_store_output(folder_name+".txt")

    #Uncomment the following block if You are training
    # mode = "train"
    # resume_dir = ""
    # saved_model = ""
    # test_saving_file = ""

    #Uncomment the following block if You are testing    
    # mode = "test"
    # resume_dir = "results/folderName/100Es/"
    # saved_model = ""
    # test_saving_file = "cl1_4Cs_512nn_rsFMRI_focal_alpEqPosOtot"

    # mode = "attribution"
    # resume_dir = "results/cl2_515nn_rsFMRI_focal/100Es/"
    # saved_model = "train-5.pth"
    # test_saving_file = ""

    # mode = "lasso"
    # resume_dir = ""
    # saved_model = ""
    # test_saving_file = ""

    # mode = "train_iic"
    # resume_dir = ""
    # saved_model = ""
    # test_saving_file = ""

    # mode = "test_iic"
    # resume_dir = "results/2cl2_fedIIC_4Cs_512nn_rsFMRI/100Es/"
    # saved_model = ""
    # test_saving_file = "2cl2_fedIIC_4Cs_512nn_rsFMRI"
    
    input_size, pos_count, neg_count, tr_ts_dfs, tr_dataloader, ts_dataloader = BAL_dataloaders_4C(1)
    model, optimizer, criterion, resume_epoch, store_dir = model_build(mode, pos_count, neg_count, input_size)
    if saved_model:
        checkpoint = torch.load(resume_dir+saved_model)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        resume_epoch = checkpoint['epoch']
      
    if mode == "train":
        train(model, optimizer, criterion, tr_dataloader, store_dir, resume_epoch)
    elif mode == "test": 
        models_list = os.listdir(resume_dir)
        models_list.sort()
        loss, acc, f1, auc, mcc = dict(), dict(), dict(), dict(), dict()
        for model_file in models_list:
            #print(model_file)
            loss[model_file] = []
            acc[model_file] = []
            f1[model_file] = []
            auc[model_file] = []
            mcc[model_file] = []

            checkpoint = torch.load(resume_dir+model_file)
            model.load_state_dict(checkpoint['model']) 
            ts_loss, ts_acc, ts_f1, ts_auc, ts_mcc= eval(model, criterion, ts_dataloader)

            loss[model_file].append(ts_loss)
            acc[model_file].append(ts_acc)
            f1[model_file].append(ts_f1)
            auc[model_file].append(ts_auc)
            mcc[model_file].append(ts_mcc)

        loss_df = pd.DataFrame.from_dict(loss)
        loss_df.to_csv(test_saving_file+"_loss.csv", index=False)

        acc = {key: ( x.cpu().numpy() for x in value) for key, value in acc.items()}
        acc_df = pd.DataFrame.from_dict(acc)
        acc_df.to_csv(test_saving_file+"_acc.csv", index=False)

        f1 = {key: ( x.cpu().numpy() for x in value) for key, value in f1.items()}
        f1_df = pd.DataFrame.from_dict(f1)
        f1_df.to_csv(test_saving_file+"_f1.csv", index=False)

        auc = {key: ( x.cpu().numpy() for x in value) for key, value in auc.items()}
        auc_df = pd.DataFrame.from_dict(auc)
        auc_df.to_csv(test_saving_file+"_auc.csv", index=False)

        mcc = {key: ( x.cpu().numpy() for x in value) for key, value in mcc.items()}
        mcc_df = pd.DataFrame.from_dict(mcc)
        mcc_df.to_csv(test_saving_file+"_mcc.csv", index=False)
    elif mode == "attribution":
        model.eval()
        baseline = torch.zeros(25, 9730)
        test_batch = next(iter(ts_dataloader)) 
        features, targets = test_batch
        ig = IntegratedGradients(model)
        attributions, delta = ig.attribute(
            features.to(DEVICE),
            baselines=baseline.to(DEVICE),
            return_convergence_delta=True
        )
        importance = attributions.abs().sum(dim=0).cpu().detach().numpy()
        imp_n = importance.argsort()[::-1][:1000]
        print(','.join(map(str, imp_n)))

        # # Plot top N features
        # top_n = 500  # Adjust as needed
        # top_indices = np.argsort(importance)[-top_n:][::-1]
        # top_values = importance[top_indices]

        # plt.figure(figsize=(10, 6))
        # plt.barh(range(top_n), top_values, align='center')
        # plt.yticks(range(top_n), top_indices)
        # plt.gca().invert_yaxis()  # Highest importance at top
        # plt.xlabel("Feature Importance (Absolute Attribution Sum)")
        # plt.title("Top 500 Important Features")
        # plt.show()
    elif mode == "lasso":
        # lasso_cv = LassoCV(alphas=[0.01,0.1,1.0], cv=5)
        lasso_model = LogisticRegression(penalty='l1', solver='liblinear', class_weight={0: 1.25, 1: 5.0}, C=1.0, random_state=42)
        print(tr_ts_dfs[0].shape)
        print(tr_ts_dfs[1].shape)
        lasso_model.fit(tr_ts_dfs[0], tr_ts_dfs[1])

        # print("Best alpha:", lasso_cv.alpha_)
        selected_features = np.where(lasso_model.coef_[0] != 0)[0]
        print(f"Selected {len(selected_features)} features:", selected_features)
        print(','.join(map(str, selected_features)))

        f1_scores = cross_val_score(lasso_model,tr_ts_dfs[0], tr_ts_dfs[1], cv=5, scoring='f1')
        print(f"Mean F1-score: {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")
    elif mode == "train_iic":
        class_num_list = [neg_count, pos_count]
        train_iic(model, optimizer, criterion, class_num_list, tr_dataloader, store_dir, resume_epoch)
    elif mode == "test_iic": 
        class_num_list = [neg_count, pos_count]
        models_list = os.listdir(resume_dir)
        models_list.sort()
        loss, acc, f1, auc, mcc = dict(), dict(), dict(), dict(), dict()
        for model_file in models_list:
            #print(model_file)
            loss[model_file] = []
            acc[model_file] = []
            f1[model_file] = []
            auc[model_file] = []
            mcc[model_file] = []

            checkpoint = torch.load(resume_dir+model_file)
            model.load_state_dict(checkpoint['model']) 
            ts_loss, ts_acc, ts_f1, ts_auc, ts_mcc= eval_iic(model, criterion, ts_dataloader, class_num_list)

            loss[model_file].append(ts_loss)
            acc[model_file].append(ts_acc)
            f1[model_file].append(ts_f1)
            auc[model_file].append(ts_auc)
            mcc[model_file].append(ts_mcc)

        loss_df = pd.DataFrame.from_dict(loss)
        loss_df.to_csv(test_saving_file+"_loss.csv", index=False)

        acc = {key: ( x.cpu().numpy() for x in value) for key, value in acc.items()}
        acc_df = pd.DataFrame.from_dict(acc)
        acc_df.to_csv(test_saving_file+"_acc.csv", index=False)

        f1 = {key: ( x.cpu().numpy() for x in value) for key, value in f1.items()}
        f1_df = pd.DataFrame.from_dict(f1)
        f1_df.to_csv(test_saving_file+"_f1.csv", index=False)

        auc = {key: ( x.cpu().numpy() for x in value) for key, value in auc.items()}
        auc_df = pd.DataFrame.from_dict(auc)
        auc_df.to_csv(test_saving_file+"_auc.csv", index=False)

        mcc = {key: ( x.cpu().numpy() for x in value) for key, value in mcc.items()}
        mcc_df = pd.DataFrame.from_dict(mcc)
        mcc_df.to_csv(test_saving_file+"_mcc.csv", index=False)

    stop_capture_and_restore_output()

if __name__ == "__main__":
    main()

