# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 20:16:09 2021

@author: 24233
"""
import mne
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from mne import Epochs, pick_types, events_from_annotations
from mne.decoding import CSP


def load_path(fname):
    names=os.listdir(fname)
    fpath=[]
    for i in names:
        fpath.append(fname+'/'+i)
    return fpath

def load_dataset(fpon):
    if type(fpon)==str:
        fpath=load_path(fpon)
    elif type(fpon)==list:
        fpath=fpon
    dataset=[]
    for i in fpath:
        dataset.append(mne.io.read_raw_edf(i,preload=True))
    return dataset

def pretreat(dataset):
    h={'Fc5.': 'FC5',
     'Fc3.': 'FC3',
     'Fc1.': 'FC1',
     'Fcz.': 'FCz',
     'Fc2.': 'FC2',
     'Fc4.': 'FC4',
     'Fc6.': 'FC6',
     'C5..': 'C5',
     'C3..': 'C3',
     'C1..': 'C1',
     'Cz..': 'Cz',
     'C2..': 'C2',
     'C4..': 'C4',
     'C6..': 'C6',
     'Cp5.': 'CP5',
     'Cp3.': 'CP3',
     'Cp1.': 'CP1',
     'Cpz.': 'CPz',
     'Cp2.': 'CP2',
     'Cp4.': 'CP4',
     'Cp6.': 'CP6',
     'Fp1.': 'Fp1',
     'Fpz.': 'Fpz',
     'Fp2.': 'Fp2',
     'Af7.': 'AF7',
     'Af3.': 'AF3',
     'Afz.': 'AFz',
     'Af4.': 'AF4',
     'Af8.': 'AF8',
     'F7..': 'F7',
     'F5..': 'F5',
     'F3..': 'F3',
     'F1..': 'F1',
     'Fz..': 'Fz',
     'F2..': 'F2',
     'F4..': 'F4',
     'F6..': 'F6',
     'F8..': 'F8',
     'Ft7.': 'FT7',
     'Ft8.': 'FT8',
     'T7..': 'T7',
     'T8..': 'T8',
     'T9..': 'T9',
     'T10.': 'T10',
     'Tp7.': 'TP7',
     'Tp8.': 'TP8',
     'P7..': 'P7',
     'P5..': 'P5',
     'P3..': 'P3',
     'P1..': 'P1',
     'Pz..': 'Pz',
     'P2..': 'P2',
     'P4..': 'P4',
     'P6..': 'P6',
     'P8..': 'P8',
     'Po7.': 'PO7',
     'Po3.': 'PO3',
     'Poz.': 'POz',
     'Po4.': 'PO4',
     'Po8.': 'PO8',
     'O1..': 'O1',
     'Oz..': 'Oz',
     'O2..': 'O2',
     'Iz..': 'Iz'}
    #ch_names=['FC3','FCz','FC4','T7','C3','Cz','C4','T8','P5','Pz','P6']
    ch_names=['C3','Cz','C4']
    length=len(dataset)
    for i in range(length):
        dataset[i].rename_channels(h)
        dataset[i].set_montage('standard_1020')
        dataset[i].set_eeg_reference('average', projection=True)
        dataset[i].pick_channels(ch_names)
        dataset[i].filter(0.5,30)
    epoch=[]
    for i in dataset:
        events,events_id=mne.events_from_annotations(i)
        epoch.append(mne.Epochs(i,events,events_id,tmin=0, tmax=4.1-1/160,baseline=(None, None)))
    return dataset,epoch,events_id

def cut_dataset(epoch,events_id):
    length=len(epoch)
    T0=epoch[0]['T0']
    T1=epoch[0]['T1']
    T2=epoch[0]['T2']
    for i in range(1,length):
        T0=mne.concatenate_epochs([T0,epoch[i]['T0']])
        T1=mne.concatenate_epochs([T1,epoch[i]['T1']])
        T2=mne.concatenate_epochs([T2,epoch[i]['T2']])
    return T0,T1,T2
    

if __name__ == '__main__':
    dataset=load_dataset('./Lab/Data/data/edf')
    dataset,epoch,events_id=pretreat(dataset)
    T0,T1,T2=cut_dataset(epoch,events_id)
    epochs=mne.concatenate_epochs([T0[0:75],T1])
    epochs=mne.concatenate_epochs([epochs,T2])
    labels = epochs.events[:, -1] - 1
    t=0
    for i in range(len(epochs)):
        label=labels[i]
        for j in range(1):
            b=np.array([epochs[i].get_data()[0][:,j*164:(j+1)*164].reshape(1,3,164),label])
            print(b[1],t)
            np.save('./原始数据-3-3-640/{0}.npy'.format(t),b)
            t=t+1