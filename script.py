import torch
import copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from torch import nn, optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import posixpath
import wfdb
from sklearn import preprocessing as ps
###############
from numpy import sin, cos, pi, linspace
from numpy.random import randn
from scipy import signal
from scipy.signal import lfilter, lfilter_zi, filtfilt, butter
from matplotlib.pyplot import plot, legend, show, grid, figure, savefig, xlim
import numpy as np
###############

class Encoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder, self).__init__()
    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    
    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )
  def forward(self, x):
    x = x.reshape((1, self.seq_len, self.n_features))
    x, (_, _) = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)
    return hidden_n.reshape((self.n_features, self.embedding_dim))
class Decoder(nn.Module):
  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(Decoder, self).__init__()
    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features
    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True
    )
    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    self.output_layer = nn.Linear(self.hidden_dim, n_features)
  def forward(self, x):
    x = x.repeat(self.seq_len, self.n_features)
    x = x.reshape((self.n_features, self.seq_len, self.input_dim))
    x, (hidden_n, cell_n) = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)
    x = x.reshape((self.seq_len, self.hidden_dim))
    return self.output_layer(x)
class RecurrentAutoencoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(RecurrentAutoencoder, self).__init__()
    self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
    self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x
  
def create_dataset(df):
  sequences = df.astype(np.float32).to_numpy().tolist()
  dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
  n_seq, seq_len, n_features = torch.stack(dataset).shape
  return dataset, seq_len, n_features


device = torch.device('cpu')
model = torch.load('C:/Users/FreeComp/Desktop/stem_projects/ecg_processing(abdo_saad)/ecg_processing/anomaly/model1.pth',map_location=torch.device('cpu'))
model = model.to(device)

##########
def plot_filter_graphs(data, order):
    lowcut = 1
    highcut = 35
    nyq = 0.5 * 300
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')

    # Apply the filter to data. Use lfilter_zi to choose the initial condition of the filter.
    z = lfilter(b, a, data)

    # Use filtfilt to apply the filter.
    y = filtfilt(b, a, data)
    y = np.flipud(y)
    y = signal.lfilter(b, a, y)
    y = np.flipud(y)

    # Make the plot.
    

    # Return the filtered signal
    return y
##########




THRESHOLD = 75
def predict(model, dataset):
  predictions, losses = [], []
  criterion = nn.L1Loss(reduction='sum').to(device)
  with torch.no_grad():
    model = model.eval()
    for seq_true in dataset:
      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)

      loss = criterion(seq_pred, seq_true)

      predictions.append(seq_pred.cpu().numpy().flatten())
      losses.append(loss.item())
  return predictions, losses


def testPredict(num_of_pulses):
  THRESHOLD = 75
  path='C:/Users/FreeComp/Desktop/stem_projects/ecg_processing(abdo_saad)/ecg_processing/anomaly/signalDataTest/12726'
  record = wfdb.rdrecord(path,sampfrom=0, sampto=250)
  testsignal= ps.scale(np.nan_to_num(record.p_signal[:,0])).tolist()
  i=0
  start=250
  end=500
  df=pd.DataFrame([testsignal])
  while i<num_of_pulses:
      record = wfdb.rdrecord(path,sampfrom=start, sampto=end)
      new_record= ps.scale(np.nan_to_num(record.p_signal[:,0])).tolist()
      y=plot_filter_graphs(new_record,2)
      new_record=y.tolist()
      df=pd.concat([df,pd.DataFrame([new_record])], ignore_index=True)
      start=end
      end=end+250
      i=i+1

  testSet,_,_=create_dataset(df)
  _, losses = predict(model, testSet)
  return losses


x=testPredict(25)
# Define the list
# Calculate the average
average = sum(x) / len(x)
print("losses is :",average) 