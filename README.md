# DGNN-for-Session-based-Recommendation

This is a Pytorch implementation for our WSDM 2023 paper:

> Zihao Li, Xianzhi Wang, Chao Yang, Lina Yao, Julian McAuley, Guandong Xu. 2023. Exploiting Explicit and Implicit Item relationships for Session-based Recommendation. In Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining (WSDM '23).

Contributors: Zihao Li.

## Overview

We propose to decouple the modeling of explicit dependencies and implicit correlations among items for session-based recommendation.
To this end, we present a dual graph neural network (DGNN), where a GNN with a single gate (SG-GNN) captures the explicit dependencies as reflected by the ordering of items in sessions, and an adaptive GNN (A-GNN) learns implicit correlations between any two items adaptively with a self-learning strategy. Our model works as below,

![framework](DGNNs.png)

## Environment Requirement

- Install Python, Pytorch(>=1.8). Our code has been tested running under a Linux desktop (NVIDIA Quadro RTX 6000), Python 3.6.8, Pytorch 1.10.1.  

## Datasets

The preprocessed datasets are included in the repo (e.g. datasets/yoochoose1_64/train.txt), where each line contains a session id and item id list (starting from 1) meaning the interactions (sorted by timestamp).

The data pre-processing script (i.e., preprocess_Dig_Yoo) is also included. For example, You can also download the raw Yoochoose-click data from [here](https://www.kaggle.com/datasets/chadgostopp/recsys-challenge-2015?select=yoochoose-clicks.dat) and run preprocess_Dig_Yoo.py to obtatin the preprocessed dataset for model training.

## Quick Start

Download this repository and run the below command on the terminal for model training.

```
python code/main.py --dataset yoochoose1_64
```

Output log :

```
2022/11/26 22:06:49 - __main__ - INFO - 65 - main - Namespace(batchSize=100, dataset='yoochoose1_64', dropout_global_att=0.5, dropout_global_ffn=0.5, epoch=50, fuse_A=False, global_att_block_nums=5, global_att_head_nums=4, hiddenSize=100, l2=1e-05, len_max=70, log_file='log/', lr=0.001, lr_dc=0.5, lr_dc_step=3, mt=0.9, nonhybrid=False, patience=5, random_seed=2023, step_global=2, valid_portion=0.1, validation=False)
2022/11/26 22:07:07 - __main__ - INFO - 135 - main - -------------------------------------------------------
2022/11/26 22:07:07 - __main__ - INFO - 137 - main - epoch: 0
2022/11/26 22:07:07 - __main__ - INFO - 208 - model - start training:2022-11-26 22:07:07.385682
2022/11/26 22:07:10 - __main__ - INFO - 224 - model - [0/3699] Loss: 10.6094
2022/11/26 22:11:35 - __main__ - INFO - 224 - model - [740/3699] Loss: 6.7706
2022/11/26 22:16:30 - __main__ - INFO - 224 - model - [1480/3699] Loss: 7.7551
2022/11/26 22:20:54 - __main__ - INFO - 224 - model - [2220/3699] Loss: 6.7143
2022/11/26 22:25:55 - __main__ - INFO - 224 - model - [2960/3699] Loss: 5.5899
2022/11/26 22:30:57 - __main__ - INFO - 226 - model - 	Loss:	25870.801
2022/11/26 22:30:57 - __main__ - INFO - 228 - model - start predicting: 2022-11-26 22:30:57.564232
2022/11/26 22:32:05 - __main__ - INFO - 153 - main - Best Result:
2022/11/26 22:32:05 - __main__ - INFO - 154 - main - 	Recall@20:	62.1274	MMR@20:	27.7101	Epoch:	0,	0
2022/11/26 22:32:05 - __main__ - INFO - 135 - main - -------------------------------------------------------
2022/11/26 22:32:05 - __main__ - INFO - 137 - main - epoch: 1
2022/11/26 22:32:05 - __main__ - INFO - 208 - model - start training:2022-11-26 22:32:05.198368
2022/11/26 22:32:05 - __main__ - INFO - 224 - model - [0/3699] Loss: 5.5067
2022/11/26 22:37:04 - __main__ - INFO - 224 - model - [740/3699] Loss: 4.7762
2022/11/26 22:41:58 - __main__ - INFO - 224 - model - [1480/3699] Loss: 5.9290
2022/11/26 22:46:58 - __main__ - INFO - 224 - model - [2220/3699] Loss: 4.4842
2022/11/26 22:52:07 - __main__ - INFO - 224 - model - [2960/3699] Loss: 4.4392
2022/11/26 22:57:16 - __main__ - INFO - 226 - model - 	Loss:	16842.912
2022/11/26 22:57:16 - __main__ - INFO - 228 - model - start predicting: 2022-11-26 22:57:16.714422
2022/11/26 22:58:33 - __main__ - INFO - 153 - main - Best Result:
2022/11/26 22:58:33 - __main__ - INFO - 154 - main - 	Recall@20:	75.0528	MMR@20:	40.1226	Epoch:	1,	1
2022/11/26 22:58:33 - __main__ - INFO - 135 - main - -------------------------------------------------------
2022/11/26 22:58:33 - __main__ - INFO - 137 - main - epoch: 2
2022/11/26 22:58:33 - __main__ - INFO - 208 - model - start training:2022-11-26 22:58:33.401552
2022/11/26 22:58:34 - __main__ - INFO - 224 - model - [0/3699] Loss: 4.3191
2022/11/26 23:03:38 - __main__ - INFO - 224 - model - [740/3699] Loss: 3.6664
2022/11/26 23:08:45 - __main__ - INFO - 224 - model - [1480/3699] Loss: 4.6060
2022/11/26 23:13:29 - __main__ - INFO - 224 - model - [2220/3699] Loss: 3.3848
2022/11/26 23:18:11 - __main__ - INFO - 224 - model - [2960/3699] Loss: 3.4304
2022/11/26 23:22:55 - __main__ - INFO - 226 - model - 	Loss:	13206.345
2022/11/26 23:22:55 - __main__ - INFO - 228 - model - start predicting: 2022-11-26 23:22:55.298667
2022/11/26 23:24:02 - __main__ - INFO - 153 - main - Best Result:
2022/11/26 23:24:02 - __main__ - INFO - 154 - main - 	Recall@20:	79.3857	MMR@20:	46.1336	Epoch:	2,	2
2022/11/26 23:24:02 - __main__ - INFO - 135 - main - -------------------------------------------------------
2022/11/26 23:24:02 - __main__ - INFO - 137 - main - epoch: 3
2022/11/26 23:24:02 - __main__ - INFO - 208 - model - start training:2022-11-26 23:24:02.707588
2022/11/26 23:24:03 - __main__ - INFO - 224 - model - [0/3699] Loss: 3.5217
2022/11/26 23:28:43 - __main__ - INFO - 224 - model - [740/3699] Loss: 3.3724
2022/11/26 23:33:26 - __main__ - INFO - 224 - model - [1480/3699] Loss: 4.1205
2022/11/26 23:38:09 - __main__ - INFO - 224 - model - [2220/3699] Loss: 3.1378
2022/11/26 23:42:53 - __main__ - INFO - 224 - model - [2960/3699] Loss: 3.1469
2022/11/26 23:47:35 - __main__ - INFO - 226 - model - 	Loss:	11808.760
2022/11/26 23:47:35 - __main__ - INFO - 228 - model - start predicting: 2022-11-26 23:47:35.565350
2022/11/26 23:48:42 - __main__ - INFO - 153 - main - Best Result:
2022/11/26 23:48:42 - __main__ - INFO - 154 - main - 	Recall@20:	80.6272	MMR@20:	48.0348	Epoch:	3,	3
2022/11/26 23:48:42 - __main__ - INFO - 135 - main - -------------------------------------------------------
2022/11/26 23:48:42 - __main__ - INFO - 137 - main - epoch: 4
2022/11/26 23:48:42 - __main__ - INFO - 208 - model - start training:2022-11-26 23:48:42.843138
2022/11/26 23:48:43 - __main__ - INFO - 224 - model - [0/3699] Loss: 3.2575
2022/11/26 23:53:25 - __main__ - INFO - 224 - model - [740/3699] Loss: 3.1841
2022/11/26 23:58:06 - __main__ - INFO - 224 - model - [1480/3699] Loss: 3.7155
2022/11/27 00:02:50 - __main__ - INFO - 224 - model - [2220/3699] Loss: 2.8898
2022/11/27 00:07:33 - __main__ - INFO - 224 - model - [2960/3699] Loss: 2.9705
2022/11/27 00:12:15 - __main__ - INFO - 226 - model - 	Loss:	10858.662
2022/11/27 00:12:15 - __main__ - INFO - 228 - model - start predicting: 2022-11-27 00:12:15.927006
2022/11/27 00:13:23 - __main__ - INFO - 153 - main - Best Result:
2022/11/27 00:13:23 - __main__ - INFO - 154 - main - 	Recall@20:	81.2122	MMR@20:	49.1387	Epoch:	4,	4
2022/11/27 00:13:23 - __main__ - INFO - 135 - main - -------------------------------------------------------
2022/11/27 00:13:23 - __main__ - INFO - 137 - main - epoch: 5
2022/11/27 00:13:23 - __main__ - INFO - 208 - model - start training:2022-11-27 00:13:23.293859
2022/11/27 00:13:23 - __main__ - INFO - 224 - model - [0/3699] Loss: 2.9813
2022/11/27 00:18:03 - __main__ - INFO - 224 - model - [740/3699] Loss: 2.9054
2022/11/27 00:22:46 - __main__ - INFO - 224 - model - [1480/3699] Loss: 3.2117
2022/11/27 00:27:30 - __main__ - INFO - 224 - model - [2220/3699] Loss: 2.5550
2022/11/27 00:32:13 - __main__ - INFO - 224 - model - [2960/3699] Loss: 2.6129
2022/11/27 00:36:56 - __main__ - INFO - 226 - model - 	Loss:	9661.092
2022/11/27 00:36:56 - __main__ - INFO - 228 - model - start predicting: 2022-11-27 00:36:56.264429
2022/11/27 00:38:03 - __main__ - INFO - 153 - main - Best Result:
2022/11/27 00:38:03 - __main__ - INFO - 154 - main - 	Recall@20:	81.7364	MMR@20:	50.3430	Epoch:	5,	5
2022/11/27 00:38:03 - __main__ - INFO - 135 - main - -------------------------------------------------------
2022/11/27 00:38:03 - __main__ - INFO - 137 - main - epoch: 6
2022/11/27 00:38:03 - __main__ - INFO - 208 - model - start training:2022-11-27 00:38:03.449805
2022/11/27 00:38:04 - __main__ - INFO - 224 - model - [0/3699] Loss: 2.6974
2022/11/27 00:42:43 - __main__ - INFO - 224 - model - [740/3699] Loss: 2.8027
2022/11/27 00:47:29 - __main__ - INFO - 224 - model - [1480/3699] Loss: 2.9811
2022/11/27 00:52:11 - __main__ - INFO - 224 - model - [2220/3699] Loss: 2.4527
2022/11/27 00:56:56 - __main__ - INFO - 224 - model - [2960/3699] Loss: 2.5373
2022/11/27 01:01:39 - __main__ - INFO - 226 - model - 	Loss:	9223.630
2022/11/27 01:01:39 - __main__ - INFO - 228 - model - start predicting: 2022-11-27 01:01:39.799956
2022/11/27 01:02:47 - __main__ - INFO - 153 - main - Best Result:
2022/11/27 01:02:47 - __main__ - INFO - 154 - main - 	Recall@20:	81.7453	MMR@20:	50.5832	Epoch:	6,	6
2022/11/27 01:02:47 - __main__ - INFO - 135 - main - -------------------------------------------------------
2022/11/27 01:02:47 - __main__ - INFO - 137 - main - epoch: 7
2022/11/27 01:02:47 - __main__ - INFO - 208 - model - start training:2022-11-27 01:02:47.634704
2022/11/27 01:02:48 - __main__ - INFO - 224 - model - [0/3699] Loss: 2.5754
2022/11/27 01:07:29 - __main__ - INFO - 224 - model - [740/3699] Loss: 2.7265
2022/11/27 01:12:09 - __main__ - INFO - 224 - model - [1480/3699] Loss: 2.7870
2022/11/27 01:16:50 - __main__ - INFO - 224 - model - [2220/3699] Loss: 2.3763
2022/11/27 01:21:34 - __main__ - INFO - 224 - model - [2960/3699] Loss: 2.4636
2022/11/27 01:26:18 - __main__ - INFO - 226 - model - 	Loss:	8902.054
2022/11/27 01:26:18 - __main__ - INFO - 228 - model - start predicting: 2022-11-27 01:26:18.832268
2022/11/27 01:27:26 - __main__ - INFO - 153 - main - Best Result:
2022/11/27 01:27:26 - __main__ - INFO - 154 - main - 	Recall@20:	81.7453	MMR@20:	50.6519	Epoch:	6,	7
2022/11/27 01:27:26 - __main__ - INFO - 135 - main - -------------------------------------------------------
2022/11/27 01:27:26 - __main__ - INFO - 137 - main - epoch: 8
2022/11/27 01:27:26 - __main__ - INFO - 208 - model - start training:2022-11-27 01:27:26.228596
2022/11/27 01:27:26 - __main__ - INFO - 224 - model - [0/3699] Loss: 2.4433
2022/11/27 01:32:09 - __main__ - INFO - 224 - model - [740/3699] Loss: 2.6179
2022/11/27 01:36:51 - __main__ - INFO - 224 - model - [1480/3699] Loss: 2.5475
2022/11/27 01:41:33 - __main__ - INFO - 224 - model - [2220/3699] Loss: 2.2534
2022/11/27 01:46:16 - __main__ - INFO - 224 - model - [2960/3699] Loss: 2.3013
2022/11/27 01:51:00 - __main__ - INFO - 226 - model - 	Loss:	8381.315
2022/11/27 01:51:00 - __main__ - INFO - 228 - model - start predicting: 2022-11-27 01:51:00.094379
2022/11/27 01:52:07 - __main__ - INFO - 153 - main - Best Result:
2022/11/27 01:52:07 - __main__ - INFO - 154 - main - 	Recall@20:	81.8813	MMR@20:	51.3188	Epoch:	8,	8
2022/11/27 01:52:07 - __main__ - INFO - 135 - main - -------------------------------------------------------
2022/11/27 01:52:07 - __main__ - INFO - 137 - main - epoch: 9
2022/11/27 01:52:07 - __main__ - INFO - 208 - model - start training:2022-11-27 01:52:07.699221
2022/11/27 01:52:08 - __main__ - INFO - 224 - model - [0/3699] Loss: 2.3394
2022/11/27 01:56:48 - __main__ - INFO - 224 - model - [740/3699] Loss: 2.5563
2022/11/27 02:01:30 - __main__ - INFO - 224 - model - [1480/3699] Loss: 2.4477
2022/11/27 02:06:13 - __main__ - INFO - 224 - model - [2220/3699] Loss: 2.1658
2022/11/27 02:10:58 - __main__ - INFO - 224 - model - [2960/3699] Loss: 2.2818
2022/11/27 02:15:40 - __main__ - INFO - 226 - model - 	Loss:	8168.046
2022/11/27 02:15:40 - __main__ - INFO - 228 - model - start predicting: 2022-11-27 02:15:40.729680
2022/11/27 02:16:48 - __main__ - INFO - 153 - main - Best Result:
2022/11/27 02:16:48 - __main__ - INFO - 154 - main - 	Recall@20:	81.8813	MMR@20:	51.3188	Epoch:	8,	8
2022/11/27 02:16:48 - __main__ - INFO - 135 - main - -------------------------------------------------------
2022/11/27 02:16:48 - __main__ - INFO - 137 - main - epoch: 10
2022/11/27 02:16:48 - __main__ - INFO - 208 - model - start training:2022-11-27 02:16:48.338916
2022/11/27 02:16:48 - __main__ - INFO - 224 - model - [0/3699] Loss: 2.2518
2022/11/27 02:21:30 - __main__ - INFO - 224 - model - [740/3699] Loss: 2.5044
2022/11/27 02:26:13 - __main__ - INFO - 224 - model - [1480/3699] Loss: 2.3628
2022/11/27 02:30:55 - __main__ - INFO - 224 - model - [2220/3699] Loss: 2.1500
2022/11/27 02:35:41 - __main__ - INFO - 224 - model - [2960/3699] Loss: 2.2438
2022/11/27 02:40:26 - __main__ - INFO - 226 - model - 	Loss:	8019.341
2022/11/27 02:40:26 - __main__ - INFO - 228 - model - start predicting: 2022-11-27 02:40:26.342581
2022/11/27 02:41:33 - __main__ - INFO - 153 - main - Best Result:
2022/11/27 02:41:33 - __main__ - INFO - 154 - main - 	Recall@20:	81.8813	MMR@20:	51.3188	Epoch:	8,	8
2022/11/27 02:41:33 - __main__ - INFO - 135 - main - -------------------------------------------------------
2022/11/27 02:41:33 - __main__ - INFO - 137 - main - epoch: 11
2022/11/27 02:41:33 - __main__ - INFO - 208 - model - start training:2022-11-27 02:41:33.858091
2022/11/27 02:41:34 - __main__ - INFO - 224 - model - [0/3699] Loss: 2.1917
2022/11/27 02:46:14 - __main__ - INFO - 224 - model - [740/3699] Loss: 2.4591
2022/11/27 02:50:57 - __main__ - INFO - 224 - model - [1480/3699] Loss: 2.2420
2022/11/27 02:55:41 - __main__ - INFO - 224 - model - [2220/3699] Loss: 2.0421
2022/11/27 03:00:25 - __main__ - INFO - 224 - model - [2960/3699] Loss: 2.1140
2022/11/27 03:05:15 - __main__ - INFO - 226 - model - 	Loss:	7767.702
2022/11/27 03:05:15 - __main__ - INFO - 228 - model - start predicting: 2022-11-27 03:05:15.717379
2022/11/27 03:06:23 - __main__ - INFO - 153 - main - Best Result:
2022/11/27 03:06:23 - __main__ - INFO - 154 - main - 	Recall@20:	81.8813	MMR@20:	51.5090	Epoch:	8,	11
2022/11/27 03:06:23 - __main__ - INFO - 135 - main - -------------------------------------------------------
2022/11/27 03:06:23 - __main__ - INFO - 137 - main - epoch: 12
2022/11/27 03:06:23 - __main__ - INFO - 208 - model - start training:2022-11-27 03:06:23.415982
2022/11/27 03:06:24 - __main__ - INFO - 224 - model - [0/3699] Loss: 2.1085
2022/11/27 03:11:04 - __main__ - INFO - 224 - model - [740/3699] Loss: 2.4391
2022/11/27 03:15:46 - __main__ - INFO - 224 - model - [1480/3699] Loss: 2.2109
2022/11/27 03:20:28 - __main__ - INFO - 224 - model - [2220/3699] Loss: 2.0273
2022/11/27 03:25:11 - __main__ - INFO - 224 - model - [2960/3699] Loss: 2.1130
2022/11/27 03:29:59 - __main__ - INFO - 226 - model - 	Loss:	7657.962
2022/11/27 03:29:59 - __main__ - INFO - 228 - model - start predicting: 2022-11-27 03:29:59.050492
2022/11/27 03:31:06 - __main__ - INFO - 153 - main - Best Result:
2022/11/27 03:31:06 - __main__ - INFO - 154 - main - 	Recall@20:	81.8813	MMR@20:	51.5090	Epoch:	8,	11
2022/11/27 03:31:06 - __main__ - INFO - 135 - main - -------------------------------------------------------
2022/11/27 03:31:06 - __main__ - INFO - 137 - main - epoch: 13
2022/11/27 03:31:06 - __main__ - INFO - 208 - model - start training:2022-11-27 03:31:06.564809
2022/11/27 03:31:07 - __main__ - INFO - 224 - model - [0/3699] Loss: 2.0643
2022/11/27 03:35:48 - __main__ - INFO - 224 - model - [740/3699] Loss: 2.4242
2022/11/27 03:40:30 - __main__ - INFO - 224 - model - [1480/3699] Loss: 2.1579
2022/11/27 03:45:12 - __main__ - INFO - 224 - model - [2220/3699] Loss: 1.9723
2022/11/27 03:49:56 - __main__ - INFO - 224 - model - [2960/3699] Loss: 2.0893
2022/11/27 03:54:39 - __main__ - INFO - 226 - model - 	Loss:	7582.233
2022/11/27 03:54:39 - __main__ - INFO - 228 - model - start predicting: 2022-11-27 03:54:39.424032
2022/11/27 03:55:46 - __main__ - INFO - 153 - main - Best Result:
2022/11/27 03:55:46 - __main__ - INFO - 154 - main - 	Recall@20:	81.8813	MMR@20:	51.5090	Epoch:	8,	11
2022/11/27 03:55:46 - __main__ - INFO - 135 - main - -------------------------------------------------------
2022/11/27 03:55:46 - __main__ - INFO - 137 - main - epoch: 14
2022/11/27 03:55:46 - __main__ - INFO - 208 - model - start training:2022-11-27 03:55:46.940110
2022/11/27 03:55:47 - __main__ - INFO - 224 - model - [0/3699] Loss: 2.0338
2022/11/27 04:00:28 - __main__ - INFO - 224 - model - [740/3699] Loss: 2.3926
2022/11/27 04:05:10 - __main__ - INFO - 224 - model - [1480/3699] Loss: 2.1152
2022/11/27 04:09:52 - __main__ - INFO - 224 - model - [2220/3699] Loss: 1.9182
2022/11/27 04:14:37 - __main__ - INFO - 224 - model - [2960/3699] Loss: 2.0222
2022/11/27 04:19:22 - __main__ - INFO - 226 - model - 	Loss:	7459.330
2022/11/27 04:19:22 - __main__ - INFO - 228 - model - start predicting: 2022-11-27 04:19:22.695413
2022/11/27 04:20:30 - __main__ - INFO - 153 - main - Best Result:
2022/11/27 04:20:30 - __main__ - INFO - 154 - main - 	Recall@20:	81.8813	MMR@20:	51.5587	Epoch:	8,	14
2022/11/27 04:20:30 - __main__ - INFO - 135 - main - -------------------------------------------------------
2022/11/27 04:20:30 - __main__ - INFO - 137 - main - epoch: 15
2022/11/27 04:20:30 - __main__ - INFO - 208 - model - start training:2022-11-27 04:20:30.478942
2022/11/27 04:20:31 - __main__ - INFO - 224 - model - [0/3699] Loss: 1.9998
2022/11/27 04:25:11 - __main__ - INFO - 224 - model - [740/3699] Loss: 2.3841
2022/11/27 04:29:52 - __main__ - INFO - 224 - model - [1480/3699] Loss: 2.0717
2022/11/27 04:34:37 - __main__ - INFO - 224 - model - [2220/3699] Loss: 1.8840
2022/11/27 04:39:20 - __main__ - INFO - 224 - model - [2960/3699] Loss: 2.0199
2022/11/27 04:44:04 - __main__ - INFO - 226 - model - 	Loss:	7402.988
2022/11/27 04:44:04 - __main__ - INFO - 228 - model - start predicting: 2022-11-27 04:44:04.551159
2022/11/27 04:45:11 - __main__ - INFO - 153 - main - Best Result:
2022/11/27 04:45:11 - __main__ - INFO - 154 - main - 	Recall@20:	81.8813	MMR@20:	51.5587	Epoch:	8,	14
2022/11/27 04:45:11 - __main__ - INFO - 135 - main - -------------------------------------------------------
2022/11/27 04:45:11 - __main__ - INFO - 137 - main - epoch: 16
2022/11/27 04:45:11 - __main__ - INFO - 208 - model - start training:2022-11-27 04:45:11.830304
2022/11/27 04:45:12 - __main__ - INFO - 224 - model - [0/3699] Loss: 1.9716
2022/11/27 04:49:52 - __main__ - INFO - 224 - model - [740/3699] Loss: 2.3714
2022/11/27 04:54:34 - __main__ - INFO - 224 - model - [1480/3699] Loss: 2.0471
2022/11/27 04:59:15 - __main__ - INFO - 224 - model - [2220/3699] Loss: 1.8756
2022/11/27 05:04:01 - __main__ - INFO - 224 - model - [2960/3699] Loss: 2.0065
2022/11/27 05:08:43 - __main__ - INFO - 226 - model - 	Loss:	7361.528
2022/11/27 05:08:43 - __main__ - INFO - 228 - model - start predicting: 2022-11-27 05:08:43.955933
2022/11/27 05:09:51 - __main__ - INFO - 153 - main - Best Result:
2022/11/27 05:09:51 - __main__ - INFO - 154 - main - 	Recall@20:	81.8813	MMR@20:	51.5587	Epoch:	8,	14
2022/11/27 05:09:51 - __main__ - INFO - 135 - main - -------------------------------------------------------
2022/11/27 05:09:51 - __main__ - INFO - 137 - main - epoch: 17
2022/11/27 05:09:51 - __main__ - INFO - 208 - model - start training:2022-11-27 05:09:51.716705
2022/11/27 05:09:52 - __main__ - INFO - 224 - model - [0/3699] Loss: 1.9626
2022/11/27 05:14:33 - __main__ - INFO - 224 - model - [740/3699] Loss: 2.3705
2022/11/27 05:19:15 - __main__ - INFO - 224 - model - [1480/3699] Loss: 2.0243
2022/11/27 05:23:58 - __main__ - INFO - 224 - model - [2220/3699] Loss: 1.8271
2022/11/27 05:28:45 - __main__ - INFO - 224 - model - [2960/3699] Loss: 1.9932
2022/11/27 05:33:29 - __main__ - INFO - 226 - model - 	Loss:	7304.998
2022/11/27 05:33:29 - __main__ - INFO - 228 - model - start predicting: 2022-11-27 05:33:29.305075
2022/11/27 05:34:37 - __main__ - INFO - 153 - main - Best Result:
2022/11/27 05:34:37 - __main__ - INFO - 154 - main - 	Recall@20:	81.8813	MMR@20:	51.5587	Epoch:	8,	14
2022/11/27 05:34:37 - __main__ - INFO - 135 - main - -------------------------------------------------------
2022/11/27 05:34:37 - __main__ - INFO - 137 - main - epoch: 18
2022/11/27 05:34:37 - __main__ - INFO - 208 - model - start training:2022-11-27 05:34:37.149351
2022/11/27 05:34:37 - __main__ - INFO - 224 - model - [0/3699] Loss: 1.9465
2022/11/27 05:39:18 - __main__ - INFO - 224 - model - [740/3699] Loss: 2.3530
2022/11/27 05:43:59 - __main__ - INFO - 224 - model - [1480/3699] Loss: 2.0044
2022/11/27 05:48:40 - __main__ - INFO - 224 - model - [2220/3699] Loss: 1.7885
2022/11/27 05:53:24 - __main__ - INFO - 224 - model - [2960/3699] Loss: 1.9837
2022/11/27 05:58:08 - __main__ - INFO - 226 - model - 	Loss:	7275.407
2022/11/27 05:58:08 - __main__ - INFO - 228 - model - start predicting: 2022-11-27 05:58:08.791308
2022/11/27 05:59:16 - __main__ - INFO - 153 - main - Best Result:
2022/11/27 05:59:16 - __main__ - INFO - 154 - main - 	Recall@20:	81.8813	MMR@20:	51.5587	Epoch:	8,	14
2022/11/27 05:59:16 - __main__ - INFO - 135 - main - -------------------------------------------------------
2022/11/27 05:59:16 - __main__ - INFO - 137 - main - epoch: 19
2022/11/27 05:59:16 - __main__ - INFO - 208 - model - start training:2022-11-27 05:59:16.271363
2022/11/27 05:59:16 - __main__ - INFO - 224 - model - [0/3699] Loss: 1.9227
2022/11/27 06:03:57 - __main__ - INFO - 224 - model - [740/3699] Loss: 2.3561
2022/11/27 06:08:38 - __main__ - INFO - 224 - model - [1480/3699] Loss: 1.9811
2022/11/27 06:13:20 - __main__ - INFO - 224 - model - [2220/3699] Loss: 1.7916
2022/11/27 06:18:05 - __main__ - INFO - 224 - model - [2960/3699] Loss: 1.9706
2022/11/27 06:22:49 - __main__ - INFO - 226 - model - 	Loss:	7252.976
2022/11/27 06:22:49 - __main__ - INFO - 228 - model - start predicting: 2022-11-27 06:22:49.021106
2022/11/27 06:23:56 - __main__ - INFO - 153 - main - Best Result:
2022/11/27 06:23:56 - __main__ - INFO - 154 - main - 	Recall@20:	81.8813	MMR@20:	51.5587	Epoch:	8,	14
2022/11/27 06:23:56 - __main__ - INFO - 159 - main - -------------------------------------------------------
2022/11/27 06:23:56 - __main__ - INFO - 162 - main - Run time: 29809.133901 s
```

You can also config other experimental settsing for your requirements
```
usage: main.py [-h] [--dataset DATASET] [--random_seed RANDOM_SEED] [--len_max LEN_MAX] [--batchSize BATCHSIZE]
               [--hiddenSize HIDDENSIZE] [--fuse_A] [--global_att_block_nums GLOBAL_ATT_BLOCK_NUMS]
               [--global_att_head_nums GLOBAL_ATT_HEAD_NUMS] [--dropout_global_att DROPOUT_GLOBAL_ATT]
               [--dropout_global_ffn DROPOUT_GLOBAL_FFN] [--epoch EPOCH] [--lr LR] [--mt MT] [--lr_dc LR_DC]
               [--lr_dc_step LR_DC_STEP] [--l2 L2] [--step_global STEP_GLOBAL] [--nonhybrid] [--patience PATIENCE]
               [--validation] [--valid_portion VALID_PORTION] [--log_file LOG_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     dataset name: diginetica/yoochoose1_4/yoochoose1_64/Gowalla/LastFM/sample
  --random_seed RANDOM_SEED
                        random_seed
  --len_max LEN_MAX     max lenghth of sequences
  --batchSize BATCHSIZE
                        input batch size
  --hiddenSize HIDDENSIZE
                        hidden state size
  --fuse_A              whether to fuse an auxiliary adjacent matrix via correlation or self-attention
  --global_att_block_nums GLOBAL_ATT_BLOCK_NUMS
                        the number of global attention blocks
  --global_att_head_nums GLOBAL_ATT_HEAD_NUMS
                        the number of multi-heads for global attention
  --dropout_global_att DROPOUT_GLOBAL_ATT
                        dropout of global attention
  --dropout_global_ffn DROPOUT_GLOBAL_FFN
                        dropout of ffn in global attention block
  --epoch EPOCH         the number of epochs to train for
  --lr LR               learning rate
  --mt MT               the momentum of SGD
  --lr_dc LR_DC         learning rate decay rate
  --lr_dc_step LR_DC_STEP
                        the number of steps after which the learning rate decay
  --l2 L2               l2 penalty
  --step_global STEP_GLOBAL
                        global gnn propogation steps
  --nonhybrid           only use the global preference to predict
  --patience PATIENCE   the number of epoch to wait before early stop
  --validation          validation
  --valid_portion VALID_PORTION
                        split the portion of training set as validation set
  --log_file LOG_FILE   log dir path
```

## Acknowledgement

Our code is developed based on [SR-GNN](https://github.com/CRIPAC-DIG/SR-GNN). Thanks to the authors for their contributions.
