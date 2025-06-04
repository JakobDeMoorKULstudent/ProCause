from torch import nn, Tensor
import torch
from typing import Tuple
import torch.nn.functional as F
import numpy as np



class CNN_earlycasevar(nn.Module):

  def __init__(self, input_size_case=1, input_size_process=1, length=5,
               nr_cnn_layers=2, nr_out_channels=10, kernel_size=2, stride=1,
               nr_dense_layers=1, dense_width=20, p=0.1, nr_outputs=1):
    super(CNN, self).__init__()
    self.input_size_case = input_size_case
    self.input_size_process = input_size_process
    self.length = length
    self.nr_cnn_layers = nr_cnn_layers
    self.nr_out_channels = nr_out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.nr_dense_layers = nr_dense_layers
    self.p = p
    self.dense_width = dense_width
    self.nr_outputs = nr_outputs

    def conv1d_size_out(size, kernel_size=self.kernel_size, stride=self.stride):
      return (size - 1 * (kernel_size - 1) - 1) // stride + 1

    # cnn layers
    self.cnn_layers = nn.ModuleDict()
    for nr in range(self.nr_cnn_layers):
      if nr == 0:
        self.cnn_layers[str(nr)] = nn.Sequential(nn.Conv1d(in_channels=self.input_size_process + 1 + self.input_size_case, out_channels=self.nr_out_channels,
                         kernel_size=self.kernel_size, stride=self.stride, dilation=1), nn.BatchNorm1d(self.nr_out_channels))
        conv_size = conv1d_size_out(self.length)
      else:
        self.cnn_layers[str(nr)] = nn.Sequential(nn.Conv1d(in_channels=self.nr_out_channels, out_channels=self.nr_out_channels,
                         kernel_size=self.kernel_size, stride=self.stride, dilation=1), nn.BatchNorm1d(self.nr_out_channels))
        conv_size = conv1d_size_out(conv_size)

    assert conv_size > 0, "too many convolutional layers, too large kernel sizes or strides"

    # compute size of flattened vector
    linear_input_size = conv_size * nr_out_channels

    # dense layers
    self.dense_layers = nn.ModuleDict()
    for nr in range(self.nr_dense_layers):
      if nr == 0:
        self.dense_layers[str(nr)] = nn.Linear(linear_input_size, self.dense_width)
        #self.batchnorm1 = nn.BatchNorm1d(width, affine=False)
      else:
        self.dense_layers[str(nr)] = nn.Linear(self.dense_width, self.dense_width)

    # outputs
    self.last_layer = nn.Linear(self.dense_width, 10)
    self.output_mean = nn.Linear(10, self.nr_outputs)
    self.output_logvar = nn.Linear(10, 1)

    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(self.p)

  def forward(self, x_case, x_process, t=None):
    '''Forward pass'''

    t_reshaped = t.reshape((x_process.shape[0], 1, x_process.shape[2]))
    x = torch.cat((x_process, t_reshaped), 1)

    x_case_reshaped = x_case.repeat(1, 1,  x_process.shape[2]).reshape((x_process.shape[0], 1, x_process.shape[2]))
    x = torch.cat((x, x_case_reshaped), 1)

    x = nn.Sequential(self.cnn_layers[str(0)], self.relu)(x)

    for nr in range(1, self.nr_cnn_layers):
      x = nn.Sequential(self.cnn_layers[str(nr)], self.relu)(x)
      #outputs = self.dropout(outputs)

    x = x.view(x.size(0), -1)

    #x_concat = torch.cat((x_case, x), 1)

    # calculate dense layers
    if self.nr_dense_layers > 0:
      hidden = nn.Sequential(self.dropout, self.dense_layers[str(0)], self.relu)(x)
      for nr in range(1, self.nr_dense_layers):
        hidden = nn.Sequential(self.dropout, self.dense_layers[str(nr)], self.relu)(hidden)
        # x = nn.Sequential(self.dropout, self.layer1, self.relu)(x) #, self.batchnorm1)(x

    # calculate outputs
    last = nn.Sequential(self.dropout, self.last_layer, self.relu)(hidden)
    mean = nn.Sequential(self.dropout, self.output_mean)(last)
    logvar = nn.Sequential(self.dropout, self.output_logvar)(last)

    return mean, logvar


class CNN(nn.Module):

  def __init__(self, input_size_case=1, input_size_process=1, length=5,
               nr_cnn_layers=2, nr_out_channels=10, kernel_size=2, stride=1,
               nr_dense_layers=1, dense_width=20, p=0.1, nr_outputs=1):
    super(CNN, self).__init__()
    self.input_size_case = input_size_case
    self.input_size_process = input_size_process
    self.length = length
    self.nr_cnn_layers = nr_cnn_layers
    self.nr_out_channels = nr_out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.nr_dense_layers = nr_dense_layers
    self.p = p
    self.dense_width = dense_width
    self.nr_outputs = nr_outputs

    def conv1d_size_out(size, kernel_size=self.kernel_size, stride=self.stride):
      return (size - 1 * (kernel_size - 1) - 1) // stride + 1

    # cnn layers
    self.cnn_layers = nn.ModuleDict()
    for nr in range(self.nr_cnn_layers):
      if nr == 0:
        self.cnn_layers[str(nr)] = nn.Sequential(nn.Conv1d(in_channels=self.input_size_process + 1, out_channels=self.nr_out_channels,
                         kernel_size=self.kernel_size, stride=self.stride, dilation=1), nn.BatchNorm1d(self.nr_out_channels))
        conv_size = conv1d_size_out(self.length)
      else:
        self.cnn_layers[str(nr)] = nn.Sequential(nn.Conv1d(in_channels=self.nr_out_channels, out_channels=self.nr_out_channels,
                         kernel_size=self.kernel_size, stride=self.stride, dilation=1), nn.BatchNorm1d(self.nr_out_channels))
        conv_size = conv1d_size_out(conv_size)

    assert conv_size > 0, "too many convolutional layers, too large kernel sizes or strides"

    # compute size of flattened vector
    linear_input_size = conv_size * nr_out_channels

    # dense layers
    self.dense_layers = nn.ModuleDict()
    for nr in range(self.nr_dense_layers):
      if nr == 0:
        self.dense_layers[str(nr)] = nn.Linear(self.input_size_case + linear_input_size, self.dense_width)
        #self.batchnorm1 = nn.BatchNorm1d(width, affine=False)
      else:
        self.dense_layers[str(nr)] = nn.Linear(self.dense_width, self.dense_width)

    # outputs
    self.last_layer = nn.Linear(self.dense_width, 10)
    self.output_mean = nn.Linear(10, self.nr_outputs)
    self.output_logvar = nn.Linear(10, 1)

    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(self.p)

  def forward(self, x_case, x_process, t=None):
    '''Forward pass'''

    t_reshaped = t.reshape((x_process.shape[0], 1, x_process.shape[2]))
    x = torch.cat((x_process, t_reshaped), 1)

    x = nn.Sequential(self.cnn_layers[str(0)], self.relu)(x)

    for nr in range(1, self.nr_cnn_layers):
      x = nn.Sequential(self.cnn_layers[str(nr)], self.relu)(x)
      #outputs = self.dropout(outputs)

    x = x.view(x.size(0), -1)

    x_concat = torch.cat((x_case, x), 1)

    # calculate dense layers
    if self.nr_dense_layers > 0:
      hidden = nn.Sequential(self.dropout, self.dense_layers[str(0)], self.relu)(x_concat)
      for nr in range(1, self.nr_dense_layers):
        hidden = nn.Sequential(self.dropout, self.dense_layers[str(nr)], self.relu)(hidden)
        # x = nn.Sequential(self.dropout, self.layer1, self.relu)(x) #, self.batchnorm1)(x

    # calculate outputs
    last = nn.Sequential(self.dropout, self.last_layer, self.relu)(hidden)
    mean = nn.Sequential(self.dropout, self.output_mean)(last)
    logvar = nn.Sequential(self.dropout, self.output_logvar)(last)

    return mean, logvar

###################################################################################################################################
###################################################################################################################################


class LSTM(nn.Module):
  '''
    Multilayer Perceptron.
  '''

  def __init__(self, input_size_case=1, input_size_process=1,
               nr_lstm_layers=1, lstm_size=1,
               nr_dense_layers=1, dense_width=20, p=0.0, nr_outputs=1, masked=True):
    super().__init__()
    self.input_size_case = input_size_case
    self.input_size_process = input_size_process
    self.nr_lstm_layers = nr_lstm_layers
    self.lstm_size = lstm_size
    self.nr_dense_layers = nr_dense_layers
    self.dense_width = dense_width
    self.p = p         # dropout probability
    self.nr_outputs = nr_outputs
    self.masked = masked

    # INPUT_SIZE_PROCESS CONTAINS THE TREATMENT IF IT IS INCLUDED

    # lstm layers
    if self.nr_lstm_layers > 0:
      self.lstm_layers = nn.ModuleDict()

      for nr in range(self.nr_lstm_layers):
        if nr == 0:
          # self.lstm_layers[str(nr)] = nn.LSTM(self.input_size_process + 1, self.lstm_size, 1)
          self.lstm_layers[str(nr)] = nn.LSTM(self.input_size_process, self.lstm_size, 1)
        else:
          self.lstm_layers[str(nr)] = nn.LSTM(self.lstm_size, self.lstm_size, 1)
    else:
      self.lstm_size = 0


    # dense layers
    self.dense_layers = nn.ModuleDict()
    for nr in range(self.nr_dense_layers):
      if nr == 0:
        self.dense_layers[str(nr)] = nn.Linear(self.input_size_case + self.lstm_size, self.dense_width)
        #self.batchnorm1 = nn.BatchNorm1d(width, affine=False)
      else:
        self.dense_layers[str(nr)] = nn.Linear(self.dense_width, self.dense_width)

    # outputs
    self.last_layer = nn.Linear(self.dense_width, 10)
    self.output_mean = nn.Linear(10, self.nr_outputs)
    # self.output_logvar = nn.Linear(10, 1)

    # parameter-free layers
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(self.p)


  def forward(self, x_case, x_process, prefix_len=None, t=None):
    '''Forward pass'''

    # X_PROCESS CONTAINS THE TREATMENT IF IT IS INCLUDED
    x = x_process

    # lstm layers
    if self.nr_lstm_layers > 0:
      x = x.transpose(0, 2)
      x = x.transpose(1, 2)
      if self.masked:
        # Sort and remember the original sorting
        x = nn.utils.rnn.pack_padded_sequence(x, lengths=prefix_len, enforce_sorted=False)

      outputs, (h, c) = self.lstm_layers[str(0)](x)
      ### !!! ### this dropout is only for standard LSTMs
      # outputs = self.dropout(outputs)

      for nr in range(1, self.nr_lstm_layers):
        outputs, (h, c) = self.lstm_layers[str(nr)](outputs, (h, c))
        # outputs = self.dropout(outputs)

      outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
      # grab the corresponding correct outputs (with prefix_len)
      outputs = outputs[prefix_len.long() - 1, np.arange(len(prefix_len))]

      # concatenate lstm output with case variables
      x_concat = torch.cat((x_case, outputs), 1)
    else:
      x_concat = x_case

      x_concat = self.relu(x_concat)

    # calculate dense layers
    if self.nr_dense_layers > 0:
      hidden = nn.Sequential(self.dropout, self.dense_layers[str(0)], self.relu)(x_concat)
      for nr in range(1, self.nr_dense_layers):
        hidden = nn.Sequential(self.dropout, self.dense_layers[str(nr)], self.relu)(hidden)
        # x = nn.Sequential(self.dropout, self.layer1, self.relu)(x) #, self.batchnorm1)(x

    # calculate outputs
    last = nn.Sequential(self.dropout, self.last_layer, self.relu)(hidden)
    mean = nn.Sequential(self.dropout, self.output_mean)(last)
    # logvar = nn.Sequential(self.dropout, self.output_logvar)(last)

    return mean
    # return mean, logvar

class Vanilla_NN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.relu = nn.ReLU()
        self.model = self._build_model()

    def _build_model(self):
        layers = [nn.Linear(self.input_size, self.hidden_size), self.relu]
        for _ in range(self.num_layers - 1):
            layers += [nn.Linear(self.hidden_size, self.hidden_size), self.relu]
        layers += [nn.Linear(self.hidden_size, self.output_size)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
   
class LSTM_VAE(nn.Module):
  '''
    Multilayer perceptron LSTM VAE.
  '''

  def __init__(self, input_size_case=1, input_size_process=1,
               nr_lstm_layers=1, lstm_size=1, latent_size=1,
               nr_dense_layers=1, dense_width=20, p=0.0, nr_outputs=1, masked=True):
    super().__init__()
    self.input_size_case = input_size_case
    self.input_size_process = input_size_process
    self.nr_lstm_layers = nr_lstm_layers
    self.lstm_size = lstm_size
    self.latent_size = latent_size
    self.nr_dense_layers = nr_dense_layers
    self.dense_width = dense_width
    self.p = p         # dropout probability
    self.nr_outputs = nr_outputs
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

     # Encoder Part
    self.encoder_lstm = torch.nn.LSTM(input_size= self.input_size_process , hidden_size= self.latent_size, batch_first=True, num_layers= self.nr_lstm_layers)
    self.mean = torch.nn.Linear(in_features=self.latent_size + self.input_size_case, out_features=self.latent_size)
    self.log_variance = torch.nn.Linear(in_features=self.latent_size + self.input_size_case, out_features=self.latent_size)

    # Decoder Part                                   
    self.init_hidden_decoder = torch.nn.Linear(in_features=self.latent_size, out_features= self.lstm_size * nr_lstm_layers)
    self.decoder_lstm = torch.nn.LSTM(input_size= self.input_size_process, hidden_size= self.lstm_size, batch_first = True, num_layers = self.nr_lstm_layers)
    
    self.output =torch.nn.Linear(in_features=self.lstm_size, out_features= self.nr_dense_layers)

    # dense layers
    self.dense_layers = nn.ModuleDict()
    for nr in range(self.nr_dense_layers):
      if nr == 0:
        self.dense_layers[str(nr)] = nn.Linear(self.lstm_size + self.input_size_case, self.dense_width)
        #self.batchnorm1 = nn.BatchNorm1d(width, affine=False)
      else:
        self.dense_layers[str(nr)] = nn.Linear(self.dense_width, self.dense_width)

    # outputs
    self.last_layer = nn.Linear(self.dense_width, 10)
    self.output_mean = nn.Linear(10, self.nr_outputs)
    # self.output_logvar = nn.Linear(10, 1)

    # parameter-free layers
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(self.p)

  def get_batch(self, batch):
        sentences_length = torch.tensor([batch.shape[1]]*batch.shape[0])
        return sentences_length
  
  def init_hidden(self, batch_size):
    hidden_cell = torch.zeros(self.nr_lstm_layers, batch_size, self.latent_size).to(self.device)
    state_cell = torch.zeros(self.nr_lstm_layers, batch_size, self.latent_size).to(self.device)
    return (hidden_cell, state_cell)

  def encoder(self, packed_x_embed, x_case, hidden_encoder, total_padding_length):

      packed_output_encoder, hidden_encoder = self.encoder_lstm(packed_x_embed, hidden_encoder)

      final_hidden_state = hidden_encoder[0][-1]  # Shape: (batch_size, hidden_size)
      combined_features = torch.cat([final_hidden_state, x_case], dim=1)  # (batch_size, hidden_size + x_case_size)

      # Compute mean and variance
      mean = self.mean(combined_features)
      log_var = self.log_variance(combined_features)
      std = torch.exp(0.5 * log_var)

      # Reparameterization trick
      batch_size = final_hidden_state.size(0)
      noise = torch.randn(batch_size, self.latent_size).to(self.device)
      z = noise * std + mean

      return z, mean, log_var

  def decoder(self, z, packed_x_embed, x_case, total_padding_length=None):
      # Concatenate z with static features

      hidden_decoder = self.init_hidden_decoder(z)

      # Reshape to (nr_lstm_layers, batch_size, hidden_size)
      hidden_decoder = hidden_decoder.view(self.nr_lstm_layers, z.size(0), self.lstm_size)
      hidden_decoder = (hidden_decoder, hidden_decoder)

      packed_output_decoder, _ = self.decoder_lstm(packed_x_embed,hidden_decoder) 

      output_decoder, _ = torch.nn.utils.rnn.pad_packed_sequence(
          packed_output_decoder, batch_first=True, total_length=total_padding_length
      )

      lstm_output = output_decoder[:, -1, :]  # (batch_size, hidden_size)

      # Concatenate x_case back into decoder outputs
      output_decoder = torch.cat([lstm_output, x_case], dim=1)  # (batch_size, seq_len, lstm_size + x_case_dim)

      return output_decoder


  def forward(self, x_case, x_process, prefix_len, states=None):
    '''Forward pass
        
    x_cases: batch_size x n_features
    x_process: batch_size (26) x input_features (18) x sequence_length (13)
    '''

    # X_PROCESS CONTAINS THE TREATMENT IF IT IS INCLUDED
    x = x_process

    states = self.init_hidden(x_process.size(0))
    states = states[0].detach(), states[1].detach()

    # lstm layers
    if self.nr_lstm_layers > 0:
      #x = x.transpose(0, 2)in
      x = x.transpose(1, 2)

      maximum_padding_length = x.size(1)

      packed_x_embed = torch.nn.utils.rnn.pack_padded_sequence(input= x, lengths= prefix_len, batch_first=True, enforce_sorted=False)

      # Encoder
      z, mean, log_var = self.encoder(packed_x_embed, x_case, states, maximum_padding_length)

      # Decoder
      output_decoder = self.decoder(z, packed_x_embed, x_case, maximum_padding_length)

      outputs = self.dropout(output_decoder)

      outputs = self.relu(outputs)

    # calculate dense layers
    if self.nr_dense_layers > 0:
      hidden = nn.Sequential(self.dropout, self.dense_layers[str(0)], self.relu)(outputs)
      for nr in range(1, self.nr_dense_layers):
        hidden = nn.Sequential(self.dropout, self.dense_layers[str(nr)], self.relu)(hidden)

    # calculate outputs
    last = nn.Sequential(self.dropout, self.last_layer, self.relu)(hidden)
    mean = nn.Sequential(self.dropout, self.output_mean)(last)

    return mean