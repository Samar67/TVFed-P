import os
import sys
import numpy as np
import pandas as pd
# import seaborn as sns
from datetime import datetime
# import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
# from captum.attr import IntegratedGradients
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
# from models import OneDCNNClassifier

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

    # important_1000_features = np.array([5170,842,4383,9431,444,2367,2403,3474,2366,4463,7524,9122,9187,4980,2352,8432,9072,1030,1840,9658,1675,2492,2417,542,308,2407,6464,2523,9512,3752,7392,3502,4255,1291,9071,4165,9670,2408,9087,5976,5984,7600,9421,8641,2537,4981,800,2542,1926,3591,4166,3945,1801,2621,1591,2264,5966,8909,764,2212,5859,9396,7168,4565,3821,9382,9112,110,1795,8587,9516,7531,8638,8676,1847,3576,5977,934,8408,3552,2354,266,8942,2282,8989,2439,8058,1972,1751,1885,3898,4161,9588,4678,9414,6666,7969,7525,4322,1239,1110,9446,3094,5059,5022,9088,251,5066,7869,797,3937,1876,1240,6865,1884,4890,5325,933,9447,2739,9077,3984,3637,6196,7048,8000,4262,4521,3176,6008,5116,9480,3817,7317,9237,9505,6454,6822,4359,2309,1714,8304,9695,8059,8691,9669,9130,3397,7175,4384,1008,4485,5774,1721,6978,9491,4510,898,490,5720,6873,6467,7008,1031,668,4218,1848,9439,8602,9571,7788,8222,2562,89,3506,8986,9095,3170,2995,2675,5076,2210,9066,2818,2447,4109,3543,7459,802,4966,1161,8827,3491,2486,9265,9327,158,8549,4562,5110,3249,9483,3836,3126,4366,6070,3445,3278,4517,3561,406,2771,4511,2724,3944,9664,9333,6086,7163,1624,9511,1504,205,8252,2540,1657,9702,796,5020,1608,4792,3864,9653,103,7577,2493,9416,5465,5828,5960,2541,8086,8819,6975,5587,7664,5852,3789,4115,2521,6654,5077,675,4652,5561,6626,9107,7322,1548,4733,8908,8733,7990,2297,3751,7421,1477,9398,5791,4210,9094,5706,2172,712,3712,1509,3955,4496,7980,2404,535,9455,2991,7087,2129,9129,7866,9537,126,8303,1965,9012,7789,7951,8812,9106,9671,200,575,8978,5546,1844,6736,1176,3306,1877,5967,8458,6002,3790,1930,5718,1695,2249,5519,2153,5471,6394,9538,3058,2121,2146,536,4989,1303,601,2422,7991,1587,5877,6796,8016,2544,1459,9579,2723,9176,9415,628,5444,288,2832,685,3829,4360,4321,7779,9065,8677,2843,1269,5167,8147,4090,1765,3559,1929,7104,8907,4677,865,8528,2637,1480,6063,2353,1253,133,4156,5591,1669,2440,3160,5165,4012,2585,6661,2853,8013,3509,3508,9562,4495,67,8523,9585,5450,1105,2283,9504,2109,2709,6062,4513,3754,9581,5067,2154,4752,2050,7166,9509,7419,6634,1100,2506,5797,2388,4572,9334,6685,7877,6833,4327,6141,708,3925,9423,8902,6630,3934,8554,897,4209,309,1426,5000,7704,4977,142,399,3473,5804,7161,9614,3999,927,1089,2058,6648,5707,4415,1635,9536,1634,6795,8935,2296,8223,100,3362,380,3929,8485,9481,686,4868,88,3902,1299,1279,1142,5080,5307,732,8168,688,7328,599,7248,7189,4419,5246,2606,6847,5254,4118,9359,6355,1270,7103,9123,2954,4867,6282,5162,1875,8088,8519,953,1803,6334,2828,8167,6453,5860,2294,6307,1874,9495,3869,4061,4371,5688,2874,9490,9397,9513,9611,8148,6789,7494,157,7897,8981,5955,5989,2790,8906,8649,74,8006,9642,3831,2443,9019,4050,321,7232,6677,3629,204,9568,7469,1598,6888,9529,2211,7688,6646,7565,5858,1032,1292,9006,6781,310,8031,3511,1380,1200,9689,1752,1677,4616,3462,9428,1190,8464,6362,6300,8531,4486,6607,281,3844,7259,6385,3973,1362,7848,4713,3390,2421,8220,6962,5459,2749,7732,5612,4507,8312,7376,102,1027,4002,1466,2842,87,3286,2250,9211,1595,8030,979,5550,7946,8194,987,1076,9198,54,7088,2662,2160,5595,7239,3083,6135,9586,5206,2130,1594,4427,9374,163,2219,7599,6306,7145,8988,4406,585,6539,5593,1869,4676,7244,2433,5922,8670,3577,7642,3484,2252,2791,2949,6872,9605,4105,2526,6655,3480,7320,7402,2000,5610,538,9226,3632,5605,6697,9613,2550,250,2295,2975,1636,2397,2424,5119,3551,9572,7914,7433,2525,3134,843,5474,7375,4542,550,7094,6776,4516,8320,7500,280,25,3942,0,1505,6283,634,4973,6759,3454,7105,9336,4509,6568,8047,886,7850,7119,58,5485,4660,8722,1120,9301,4769,857,1729,4177,4779,1979,4875,3446,7223,51,9643,5630,947,5681,496,4782,3240,1414,9569,9389,5841,4281,1981,7045,7569,9386,7691,7151,446,9492,7936,3101,9699,2605,9181,3135,4005,6712,61,2074,4505,7237,1223,7890,941,919,709,630,1084,1855,7271,543,4083,3901,8237,7588,4374,7830,1841,4108,9345,9558,7543,3943,2241,6221,189,2474,2955,906,2809,6760,8655,5770,3455,5404,3091,9180,3261,2232,7794,2201,2851,8634,8044,1802,7944,4569,6294,3649,484,994,4877,3045,1334,8824,7162,5060,2700,839,6689,3594,4638,3936,3935,8626,7654,7520,433,2999,6835,5842,2334,5472,1547,7875,8665,8896,9667,1749,8714,4470,4778,5023,4960,5096,7188,96,2333,6631,2636,9263,2315,7530,6757,9559,2530,1973,6358,123,7576,404,2483,9262,2473,667,6280,592,1389,8769,9666,1976,1359,2559,154,6134,5813,111,442,5330,9632,9426,8447,9100,5437,2394,97,2538,5488,9194,134,7846,8726,3086,8171,9445,9257,2900,3558,2410,7815,5954,5456,5921,2418,32,7777,9589,9565,9207,7347,2990,6124,2386,9701,856,1199,694,4529,6152,8435,424,3548,6683,9557,7123,5617,5175,9085,8122,801,4649,590,3974,4062,8484,6038,5983,9250,7012,2527,929,7514,1484,4314,5346,9598,8020,3897,6016,1058,6942,1832,4506,6892,5361,8381,3167,8949,141,6198,7306,981,4550,7635,6463,6052,2921,3059,2251,5429,8155,160,1451,8279,7208,8872,7415,2487,390,845,7976,8971,6277,7587,2213,8563,6153,2238,8396,5710,4320,725,273,9041,680,143,460,959,1432,3472,9049,9004,3220,5536,8980,6348,4886,7019,2622,5558,342,5021,7293,516,552,1710,4828,1075])
    # important_lasso_cnt_features_218 = np.array([10,209,241,312,355,435,498,505,507,542,555,571,574,577,580,581,621,629,631,685,690,726,754,756,869,870,882,888,989,991,1138,1189,1206,1253,1358,1359,1400,1446,1453,1459,1502,1558,1560,1593,1719,1795,1855,1928,1949,1951,1991,2029,2139,2212,2272,2276,2277,2279,2433,2434,2444,2504,2536,2595,2641,2648,2652,2728,2759,2767,2821,2833,2851,2939,2947,2949,2956,3006,3008,3178,3216,3329,3332,3386,3418,3441,3444,3506,3603,3608,3615,3643,3712,3719,3725,3727,3829,3830,3944,4012,4105,4157,4182,4183,4218,4234,4261,4372,4450,4470,4495,4540,4578,4589,4634,4659,4660,4662,4669,4762,4781,4867,4880,4993,5053,5110,5167,5169,5192,5195,5210,5254,5265,5266,5272,5282,5286,5324,5335,5385,5477,5522,5550,5596,5608,5622,5637,5666,5749,5817,5932,5977,6106,6177,6190,6248,6379,6386,6411,6435,6494,6535,6536,6602,6705,6793,6800,6826,6842,6962,6973,6995,7000,7010,7028,7145,7151,7166,7288,7317,7338,7393,7404,7416,7543,7576,7577,7832,7838,7899,7973,8012,8055,8113,8192,8285,8389,8463,8542,8654,8788,8950,9088,9141,9257,9265,9323,9393,9422,9462,9505,9520,9537,9579,9614,9621,9663,9681,9714])
    # filtered_tr_features = tr_features.iloc[:, important_lasso_cnt_features_218]
    # filtered_ts_features = ts_features.iloc[:, important_lasso_cnt_features_218]

    # input_size = filtered_tr_features.shape[1]
    input_size = tr_features.shape[1]

    scaler1 = MinMaxScaler()
    # scaler1 = StandardScaler()
    # norm_tr = pd.DataFrame(scaler1.fit_transform(filtered_tr_features), columns=filtered_tr_features.columns)
    norm_tr = pd.DataFrame(scaler1.fit_transform(tr_features), columns=tr_features.columns)
    scaler2 = MinMaxScaler()
    # scaler2 = StandardScaler()
    # norm_ts = pd.DataFrame(scaler2.fit_transform(filtered_ts_features), columns=filtered_ts_features.columns)
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

    # important_1000_features = np.array([5170,842,4383,9431,444,2367,2403,3474,2366,4463,7524,9122,9187,4980,2352,8432,9072,1030,1840,9658,1675,2492,2417,542,308,2407,6464,2523,9512,3752,7392,3502,4255,1291,9071,4165,9670,2408,9087,5976,5984,7600,9421,8641,2537,4981,800,2542,1926,3591,4166,3945,1801,2621,1591,2264,5966,8909,764,2212,5859,9396,7168,4565,3821,9382,9112,110,1795,8587,9516,7531,8638,8676,1847,3576,5977,934,8408,3552,2354,266,8942,2282,8989,2439,8058,1972,1751,1885,3898,4161,9588,4678,9414,6666,7969,7525,4322,1239,1110,9446,3094,5059,5022,9088,251,5066,7869,797,3937,1876,1240,6865,1884,4890,5325,933,9447,2739,9077,3984,3637,6196,7048,8000,4262,4521,3176,6008,5116,9480,3817,7317,9237,9505,6454,6822,4359,2309,1714,8304,9695,8059,8691,9669,9130,3397,7175,4384,1008,4485,5774,1721,6978,9491,4510,898,490,5720,6873,6467,7008,1031,668,4218,1848,9439,8602,9571,7788,8222,2562,89,3506,8986,9095,3170,2995,2675,5076,2210,9066,2818,2447,4109,3543,7459,802,4966,1161,8827,3491,2486,9265,9327,158,8549,4562,5110,3249,9483,3836,3126,4366,6070,3445,3278,4517,3561,406,2771,4511,2724,3944,9664,9333,6086,7163,1624,9511,1504,205,8252,2540,1657,9702,796,5020,1608,4792,3864,9653,103,7577,2493,9416,5465,5828,5960,2541,8086,8819,6975,5587,7664,5852,3789,4115,2521,6654,5077,675,4652,5561,6626,9107,7322,1548,4733,8908,8733,7990,2297,3751,7421,1477,9398,5791,4210,9094,5706,2172,712,3712,1509,3955,4496,7980,2404,535,9455,2991,7087,2129,9129,7866,9537,126,8303,1965,9012,7789,7951,8812,9106,9671,200,575,8978,5546,1844,6736,1176,3306,1877,5967,8458,6002,3790,1930,5718,1695,2249,5519,2153,5471,6394,9538,3058,2121,2146,536,4989,1303,601,2422,7991,1587,5877,6796,8016,2544,1459,9579,2723,9176,9415,628,5444,288,2832,685,3829,4360,4321,7779,9065,8677,2843,1269,5167,8147,4090,1765,3559,1929,7104,8907,4677,865,8528,2637,1480,6063,2353,1253,133,4156,5591,1669,2440,3160,5165,4012,2585,6661,2853,8013,3509,3508,9562,4495,67,8523,9585,5450,1105,2283,9504,2109,2709,6062,4513,3754,9581,5067,2154,4752,2050,7166,9509,7419,6634,1100,2506,5797,2388,4572,9334,6685,7877,6833,4327,6141,708,3925,9423,8902,6630,3934,8554,897,4209,309,1426,5000,7704,4977,142,399,3473,5804,7161,9614,3999,927,1089,2058,6648,5707,4415,1635,9536,1634,6795,8935,2296,8223,100,3362,380,3929,8485,9481,686,4868,88,3902,1299,1279,1142,5080,5307,732,8168,688,7328,599,7248,7189,4419,5246,2606,6847,5254,4118,9359,6355,1270,7103,9123,2954,4867,6282,5162,1875,8088,8519,953,1803,6334,2828,8167,6453,5860,2294,6307,1874,9495,3869,4061,4371,5688,2874,9490,9397,9513,9611,8148,6789,7494,157,7897,8981,5955,5989,2790,8906,8649,74,8006,9642,3831,2443,9019,4050,321,7232,6677,3629,204,9568,7469,1598,6888,9529,2211,7688,6646,7565,5858,1032,1292,9006,6781,310,8031,3511,1380,1200,9689,1752,1677,4616,3462,9428,1190,8464,6362,6300,8531,4486,6607,281,3844,7259,6385,3973,1362,7848,4713,3390,2421,8220,6962,5459,2749,7732,5612,4507,8312,7376,102,1027,4002,1466,2842,87,3286,2250,9211,1595,8030,979,5550,7946,8194,987,1076,9198,54,7088,2662,2160,5595,7239,3083,6135,9586,5206,2130,1594,4427,9374,163,2219,7599,6306,7145,8988,4406,585,6539,5593,1869,4676,7244,2433,5922,8670,3577,7642,3484,2252,2791,2949,6872,9605,4105,2526,6655,3480,7320,7402,2000,5610,538,9226,3632,5605,6697,9613,2550,250,2295,2975,1636,2397,2424,5119,3551,9572,7914,7433,2525,3134,843,5474,7375,4542,550,7094,6776,4516,8320,7500,280,25,3942,0,1505,6283,634,4973,6759,3454,7105,9336,4509,6568,8047,886,7850,7119,58,5485,4660,8722,1120,9301,4769,857,1729,4177,4779,1979,4875,3446,7223,51,9643,5630,947,5681,496,4782,3240,1414,9569,9389,5841,4281,1981,7045,7569,9386,7691,7151,446,9492,7936,3101,9699,2605,9181,3135,4005,6712,61,2074,4505,7237,1223,7890,941,919,709,630,1084,1855,7271,543,4083,3901,8237,7588,4374,7830,1841,4108,9345,9558,7543,3943,2241,6221,189,2474,2955,906,2809,6760,8655,5770,3455,5404,3091,9180,3261,2232,7794,2201,2851,8634,8044,1802,7944,4569,6294,3649,484,994,4877,3045,1334,8824,7162,5060,2700,839,6689,3594,4638,3936,3935,8626,7654,7520,433,2999,6835,5842,2334,5472,1547,7875,8665,8896,9667,1749,8714,4470,4778,5023,4960,5096,7188,96,2333,6631,2636,9263,2315,7530,6757,9559,2530,1973,6358,123,7576,404,2483,9262,2473,667,6280,592,1389,8769,9666,1976,1359,2559,154,6134,5813,111,442,5330,9632,9426,8447,9100,5437,2394,97,2538,5488,9194,134,7846,8726,3086,8171,9445,9257,2900,3558,2410,7815,5954,5456,5921,2418,32,7777,9589,9565,9207,7347,2990,6124,2386,9701,856,1199,694,4529,6152,8435,424,3548,6683,9557,7123,5617,5175,9085,8122,801,4649,590,3974,4062,8484,6038,5983,9250,7012,2527,929,7514,1484,4314,5346,9598,8020,3897,6016,1058,6942,1832,4506,6892,5361,8381,3167,8949,141,6198,7306,981,4550,7635,6463,6052,2921,3059,2251,5429,8155,160,1451,8279,7208,8872,7415,2487,390,845,7976,8971,6277,7587,2213,8563,6153,2238,8396,5710,4320,725,273,9041,680,143,460,959,1432,3472,9049,9004,3220,5536,8980,6348,4886,7019,2622,5558,342,5021,7293,516,552,1710,4828,1075])
    # important_lasso_cnt_features_218 = np.array([10,209,241,312,355,435,498,505,507,542,555,571,574,577,580,581,621,629,631,685,690,726,754,756,869,870,882,888,989,991,1138,1189,1206,1253,1358,1359,1400,1446,1453,1459,1502,1558,1560,1593,1719,1795,1855,1928,1949,1951,1991,2029,2139,2212,2272,2276,2277,2279,2433,2434,2444,2504,2536,2595,2641,2648,2652,2728,2759,2767,2821,2833,2851,2939,2947,2949,2956,3006,3008,3178,3216,3329,3332,3386,3418,3441,3444,3506,3603,3608,3615,3643,3712,3719,3725,3727,3829,3830,3944,4012,4105,4157,4182,4183,4218,4234,4261,4372,4450,4470,4495,4540,4578,4589,4634,4659,4660,4662,4669,4762,4781,4867,4880,4993,5053,5110,5167,5169,5192,5195,5210,5254,5265,5266,5272,5282,5286,5324,5335,5385,5477,5522,5550,5596,5608,5622,5637,5666,5749,5817,5932,5977,6106,6177,6190,6248,6379,6386,6411,6435,6494,6535,6536,6602,6705,6793,6800,6826,6842,6962,6973,6995,7000,7010,7028,7145,7151,7166,7288,7317,7338,7393,7404,7416,7543,7576,7577,7832,7838,7899,7973,8012,8055,8113,8192,8285,8389,8463,8542,8654,8788,8950,9088,9141,9257,9265,9323,9393,9422,9462,9505,9520,9537,9579,9614,9621,9663,9681,9714])
    # filtered_tr_features = tr_features.iloc[:, important_lasso_cnt_features_218]
    # filtered_ts_features = ts_features.iloc[:, important_lasso_cnt_features_218]

    # input_size = filtered_tr_features.shape[1]
    input_size = tr_features.shape[1]

    scaler1 = MinMaxScaler()
    # scaler1 = StandardScaler()
    # norm_tr = pd.DataFrame(scaler1.fit_transform(filtered_tr_features), columns=filtered_tr_features.columns)
    norm_tr = pd.DataFrame(scaler1.fit_transform(tr_features), columns=tr_features.columns)
    scaler2 = MinMaxScaler()
    # scaler2 = StandardScaler()
    # norm_ts = pd.DataFrame(scaler2.fit_transform(filtered_ts_features), columns=filtered_ts_features.columns)
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
    """
    Computes weighted binary cross-entropy loss.

    Args:
        outputs (torch.Tensor): Predicted probabilities (after sigmoid).
        targets (torch.Tensor): Binary target labels (0 or 1).
        pos_weight (float): Weight for the positive (minority) class.

    Returns:
        torch.Tensor: The weighted binary cross-entropy loss.
    """
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
    """
    Computes the focal loss for imbalanced binary classification.

    Args:
        outputs (torch.Tensor): Predicted probabilities (after sigmoid).
        targets (torch.Tensor): Binary target labels (0 or 1).
        gamma (float): Focusing parameter (typically 2).
        alpha (float): Weight for the positive class (between 0 and 1).

    Returns:
        torch.Tensor: The focal loss.
    """
    # criterion = torch.nn.BCEWithLogitsLoss(weight=alpha, reduction='none')
    # focal_loss = lambda pred, target: (criterion(pred, target)) * (1 - torch.exp(-criterion(pred, target))).mean()

    outputs = torch.clamp(outputs, min=1e-7, max=1-1e-7) # Avoid log(0)
    pt = (targets * outputs) + ((1 - targets) * (1 - outputs))
    at = (targets * alpha) + ((1 - targets) * (1 - alpha))
    focal_weight = at * (1 - pt) ** gamma
    loss = F.binary_cross_entropy(outputs, targets, reduction='mean')
    focal_loss = (focal_weight * loss).mean()
    return focal_loss

    # outputs = torch.clamp(outputs, min=1e-7, max=1-1e-7)  # Avoid log(0)
    # pt = (targets * outputs) + ((1 - targets) * (1 - outputs))  # p_t
    # at = (targets * alpha) + ((1 - targets) * (1 - alpha))  # alpha_t
    # loss = -at * (1 - pt) ** gamma * torch.log(pt)  # Focal Loss
    # return loss.mean()

def tversky_loss(y_pred, y_true, alpha=0.3, beta=0.7, epsilon=1e-6):
    """
    Compute the Tversky Loss for binary classification using PyTorch tensors.
    
    Args:
        y_true (torch.Tensor): Ground truth labels, shape (N,) or (N, 1)
        y_pred (torch.Tensor): Predicted probabilities, shape (N,) or (N, 1)
        alpha (float): Weight for false positives (default: 0.7)
        beta (float): Weight for false negatives (default: 0.3)
        epsilon (float): Small value to avoid division by zero (default: 1e-6)
        
    Returns:
        torch.Tensor: Scalar Tversky Loss value
    """
    # Input validation
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be non-negative")

    # Ensure tensors are float and flatten
    y_true = y_true.float().flatten()
    y_pred = y_pred.float().flatten()

    # Clip predictions to avoid numerical issues
    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)

    # Compute true positives, false positives, false negatives
    true_pos = torch.sum(y_true * y_pred)
    false_pos = torch.sum((1 - y_true) * y_pred)
    false_neg = torch.sum(y_true * (1 - y_pred))

    # Compute Tversky index
    tversky_index = (true_pos + epsilon) / (
        true_pos + alpha * false_pos + beta * false_neg + epsilon
    )

    # Return Tversky loss
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
    criterion = lambda outputs, targets: focal_loss(outputs, targets, gamma=2.0, alpha=alpha_focal)
    # criterion = lambda outputs, targets: focal_loss(outputs, targets, gamma=1.0, alpha=alpha_focal)
    # criterion = DiceBCELoss()
    # alpha = (5*alpha_focal/6) + (17/30)
    # beta = 1 - alpha
    # if alpha_focal < 0.25:
    #     alpha, beta = 0.7, 0.3
    # else:
    #     alpha, beta = 0.8, 0.2
    # criterion = lambda outputs, targets: tversky_loss(outputs, targets, alpha= 0.2, beta= 0.8)
    # criterion = lambda outputs, targets: tversky_loss(outputs, targets, alpha= 1-alpha_focal, beta= alpha_focal)
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
        
        # t = cls_p_list / (torch.pow(cls_loss, args.d)+1e-5)
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
        
        # Ensure proper feature dimensions [batch_size, 2, 256] -> [2*batch_size, 256]
        if features.dim() == 3:
            features = torch.cat(torch.unbind(features, dim=1), dim=0)
        features = features.view(-1, 256)  # [N, 256]
        
        # Calculate actual batch size (after any potential flattening)
        total_samples = features.shape[0]
        batch_size = total_samples // 2  # Original batch size
        
        # Create targets for both views
        targets = targets.contiguous().view(-1).repeat(2)[:total_samples]  # Handle potential truncation
        
        # Create proper mask dimensions
        mask = torch.eq(targets.unsqueeze(1), targets.unsqueeze(0)).float()
        
        # Create logits_mask with CORRECT dimensions
        logits_mask = torch.ones(total_samples, total_samples, device=device) - \
                     torch.eye(total_samples, device=device)
        
        # Apply mask
        mask = mask * logits_mask
        
        # Compute similarity matrix
        logits = features.mm(features.T)  # [total_samples, total_samples]
        
        # Compute class-aware temperature scaling
        with torch.no_grad():
            weights = self.cls_num_list[targets]  # [total_samples]
            temp = torch.sqrt(weights.unsqueeze(0) * weights.unsqueeze(1))
            temp = torch.clamp(temp, min=0.07)
        
        # Apply temperature scaling
        logits = logits / (temp * self.temperature)
        
        # Numerical stability
        logits_max = logits.max(dim=1, keepdim=True)[0].detach()
        logits = logits - logits_max
        
        # Compute probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # Compute mean log prob for positives
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        
        return -1 * mean_log_prob_pos.mean()
        
class InterSCL(nn.Module):
    def __init__(self, cls_num_list=None, temperature=0.1):
        super(InterSCL, self).__init__()
        self.temperature = temperature
        self.cls_num_list = cls_num_list

    def forward(self, centers1, features, targets):
        device = features.device  # Get device from input tensor
        # print(device)
        # Process features

        
        batch_size = features.shape[0] // 2
        # print(f"batch_size = {batch_size}")
        num_centers = centers1.shape[0]
        # print(f"num_centers = {num_centers}")

        if features.shape[0] != batch_size * 2:  # 2 views per sample
            features = features[:batch_size * 2]
            targets = targets[:batch_size * 2]

        features = torch.cat(torch.unbind(features, dim=1), dim=0)  # [2*batch_size, feature_dim]
        # print(f"features b3d el unbind w cat {features.shape}")
        features = features.view(-1, 256)
        # print(f"features b3d el view{features.shape}")

        
        # Create targets including centers
        targets = targets.contiguous().view(-1, 1)
        targets_centers = torch.arange(num_centers, device=device).view(-1, 1)
        all_targets = torch.cat([targets.repeat(2, 1), targets_centers], dim=0)
        
        # Create mask for original samples
        mask = torch.eq(targets[:2*batch_size], targets.T).float().to(device)
        
        # Create logits mask (on correct device)
        logits_mask = torch.ones(2*batch_size, 2*batch_size + num_centers, device=device)
        
        # Create indices on the same device
        indices = torch.arange(2*batch_size, device=device).view(-1, 1)
        
        # Perform scatter on correct device
        logits_mask.scatter_(1, indices, 0)
        logits_mask[:, 2*batch_size:] = 0  # Mask out centers
        
        # Combine features and centers
        all_features = torch.cat([features, centers1], dim=0)
        
        # Compute similarity matrix
        logits = features.mm(all_features.T) / self.temperature
        
        # Numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        # Compute probabilities
        # exp_logits = torch.exp(logits) * logits_mask
        # print(f"exp_logits = {exp_logits}")

        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # print(f"exp_logits = {exp_logits}")

        
        # # Compute mean log prob for positives
        # mean_log_prob_pos = (mask * log_prob[:, :2*batch_size]).sum(1) / (mask.sum(1) + 1e-8)
        # print(f"mean_log_prob_pos = {mean_log_prob_pos}")

        # lossss = -1 * mean_log_prob_pos.mean()  
        # print(lossss) 
        # loss = loss.view(2, batch_size).mean()


        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob[:, :2*batch_size]).sum(1) / (mask.sum(1) + 1e-8)
        # print(f"mean_log_prob_pos  = {mean_log_prob_pos}")

        loss = -1 * mean_log_prob_pos
        # print(f"loss abl el view = {loss}")

        loss = loss.view(2, batch_size).mean()
        # print(f"loss b3d el view = {loss}")

        
        return loss
     
def iic_loss(features, prototypes, logits, labels, class_num_list, loss_class, k1=1.0, k2=1.0):
    # Detach prototypes to prevent gradient flow
    prototypes = F.normalize(prototypes, dim=1).detach().clone()
    # print(f"prototypes shape and value {prototypes.shape} \n {prototypes}")
    
    # Initialize criteria
    intra_cl_criterion = IntraSCL(cls_num_list=class_num_list)
    inter_cl_criterion = InterSCL(cls_num_list=class_num_list)
    ce_criterion = DALA(cls_num_list=class_num_list, cls_loss=loss_class.detach())  # Detach loss_class
    
    # Compute losses
    with torch.no_grad():
        loss_ce = ce_criterion(logits.detach(), labels)  # Detach if not needed for gradients
    loss_cl_inter = inter_cl_criterion(prototypes, features, labels)
    loss_cl_intra = intra_cl_criterion(features, labels)

    # print(f"loss_ce = {loss_ce}")
    # print(f"loss_cl_inter = {loss_cl_inter}")
    # print(f"loss_cl_intra = {loss_cl_intra}")
    
    # Combine losses
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
            # print("batch gdeeedaaaaa")
            optimizer.zero_grad()
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            
            # Forward pass with feature extraction
            with torch.set_grad_enabled(True):
                # print(f"Features shape: {features.shape} \n features = {features}")
                x = model.batch_norm(features)
                # print(f"b3d el batchnorm: {x.shape}  \n x = {x}")
                x = F.relu(model.fc1(x))
                # print(f"b3d el relu fc1: {x.shape} \n x = {x}")
                x = model.dropout1(x)
                # print(f"b3d el dropout1: {x.shape} \n x = {x}")
                features_256 = F.relu(model.fc2(x))  # Features for prototypes
                # print("features_256 shape  value")
                # print(features_256.shape)
                # print(features_256)
                x = model.dropout2(features_256)
                # print(f"b3d el dropout2: {x.shape}")
                logits = model.fc3(x)  # Get raw logits before sigmoid
                # print("logits (abl el sigmoid)")
                # print(logits)
                preds = torch.sigmoid(logits).squeeze(dim=1)  # Apply sigmoid here
                # print("predssss (b3d el sigmoid)")
                # print(preds)
            
            # Calculate class-wise loss (no grad)
            with torch.no_grad():
                preds_clamped = torch.clamp(preds, 1e-7, 1-1e-7)
                loss_per_sample = F.binary_cross_entropy(preds_clamped, targets, reduction='none')
                
                for cls_idx in range(len(class_num_list)):
                    mask = (targets == cls_idx)
                    if mask.any():
                        cls_loss[cls_idx] += loss_per_sample[mask].sum()
                        cls_count[cls_idx] += mask.sum()
                        
                        # Update prototypes (no grad)
                        prototypes[cls_idx] += features_256[mask].sum(dim=0)
                        prototype_counts[cls_idx] += mask.sum()
            
            # Normalize prototypes and losses
            with torch.no_grad():
                current_prototypes = prototypes / (prototype_counts.unsqueeze(1) + 1e-8)
                current_cls_loss = cls_loss / (cls_count + 1e-8)
            
            # Calculate IIC loss - use logits (before sigmoid) for the loss calculation
            loss = iic_loss(
                features=features_256,
                prototypes=current_prototypes,
                logits=logits,  # Pass raw logits here
                labels=targets.long(),
                class_num_list=class_num_list,
                loss_class=current_cls_loss
            )
            # print(f"batch losssss {loss}")
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Store predictions
            with torch.no_grad():
                total_loss += loss.item()
                all_preds = torch.cat((all_preds, preds.to(DEVICE)),dim=0)
                all_targets = torch.cat((all_targets, targets),dim=0)
        
        # Save checkpoint and print metrics
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
            
            # Calculate IIC loss - use logits (before sigmoid) for the loss calculation
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
    # optimizer = optim.AdamW(model.mobilenet.classifier.parameters(), lr=LR)
    # criterion = nn.BCELoss()
    model.train()  
    for epoch in range(resume_epoch, resume_epoch+EPOCHS):
        all_preds = torch.tensor([], dtype=float, device=DEVICE)
        all_targets = torch.tensor([],dtype=float,device=DEVICE)
        total_loss = 0.0
        for features, targets in loader:
            # print(targets.shape)
            # print(targets)
            optimizer.zero_grad()
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            preds = model(features)
            # print(preds.shape)
            # print(preds)
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
    # criterion = nn.BCELoss()
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
    
    # mode = "train"
    # resume_dir = ""
    # saved_model = ""
    # test_saving_file = ""

    mode = "test"
    resume_dir = "results/cl1_4Cs_512nn_rsFMRI_focal_alpEqPosOtot/100Es/"
    saved_model = ""
    test_saving_file = "cl1_4Cs_512nn_rsFMRI_focal_alpEqPosOtot"

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
