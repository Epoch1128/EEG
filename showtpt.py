# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 18:43:46 2021

@author: 24233
"""
import mne  #version0.23
sampling_freq=500
ch_names=['FC3','FCz','FC4','T7','C3','Cz','C4','T8','P5','Pz','P6']
ch_num=[2,4,6,41,9,11,13,42,48,51,54]
ch_types = ['eeg']*11
info=mne.create_info(ch_names,ch_types=ch_types,sfreq=sampling_freq) #创建info
info.set_montage('standard_1020')
