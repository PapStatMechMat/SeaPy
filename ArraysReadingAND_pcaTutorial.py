# Authors: Stefanos Papanikolaou <stephanos.papanikolaou@gmail.com>
# BSD 2-Clause License
# Copyright (c) 2021, PapStatMechMat
# All rights reserved.
import AutomationTools as AT
import unsupervised_learning_tools as UMLT
import MachineLearningTools as ML

in_dir,out_dir,method_choice, file_choice=AT.GetDirectories()
filetype=file_choice
max_num=80
print(file_choice)
if method_choice == 'PCA Analysis':
    fs,labs=AT.GetFiles(in_dir,filetype,max_num)
    dt=ML.PrepareFeatureMatrix(fs,filetype)
    x,y,per=UMLT.ApplyPCA(dt,out_dir,'Total_')
    ML.PlotPCA(x,y,per,labs,out_dir)
