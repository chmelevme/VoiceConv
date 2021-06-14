from torch import nn
import torch


class ConvBlock(nn.Module):
  def __init__(self, c_in, c_h, subsample=1):
    super(ConvBlock, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv1d(c_in, c_h, 3, padding=1),
        nn.BatchNorm1d(c_h),
        nn.LeakyReLU(),
        nn.Conv1d(c_h, c_in, 3, padding=1, stride=subsample)
    )
    self.sub = subsample
    
  
  def forward(self, x):
    y = self.conv(x)
    
    if self.sub > 1:
      x = torch.nn.functional.avg_pool1d(x, kernel_size=self.sub)

    return x+y

class DecConvBlock(nn.Module):
  def __init__(self, c_in, c_h, upsample=1):
    super(DecConvBlock, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv1d(c_in, c_h, 3, padding=1),
        nn.BatchNorm1d(c_h),
        nn.LeakyReLU(),
        nn.Conv1d(c_h, c_in, 3, padding=1)
    )
    self.ups = upsample
  
  def forward(self, x):
    y = self.conv(x)
    if self.ups >1:
      x = torch.nn.functional.interpolate(x, scale_factor = self.ups)
      y = torch.nn.functional.interpolate(y, scale_factor = self.ups)
      
    return x+y

class Encoder(nn.Module):
  def __init__(self, c_in, c_out, n_conv_blocks, c_h, subsample):
    super(Encoder, self).__init__()
    self.conv_first = nn.Conv1d(c_in, c_h, 3, padding=1)
    self.conv_blocks = nn.ModuleList(
        [ConvBlock(c_h, c_h, subsample) for _, subsample in zip(range(n_conv_blocks),subsample)]
    )
    self.conv_last = nn.Conv1d(c_h, c_out, 3,padding=1)
    self.IN = InstanceNorm()

  def forward(self, x):
    x = self.conv_first(x)
    mns = []
    stds = []

    for block in self.conv_blocks:
      x = block(x)
      x, mean, std = self.IN(x, True)
      mns.append(mean)
      stds.append(std)

    y = self.conv_last(x)

    return (y, mns, stds)

class AdaIN(nn.Module):
  def __init__(self):
    super(AdaIN, self).__init__()
    self.IN = InstanceNorm()
  
  def forward(self, x, mn, std):
    x = self.IN(x)
    y = std*x+mn
    
    return y

class Decoder(nn.Module):
  def __init__(self,c_in, c_out, n_conv_blocks, c_h, upsample=1):
    super(Decoder, self).__init__()
    self.conv = nn.Conv1d(c_in, c_h, 3, padding=1)
    self.conv_blocks = nn.ModuleList(
        [DecConvBlock(c_h, c_h, ups) for _, ups in zip(range(n_conv_blocks), upsample)]
    )
    self.gru = nn.GRU(c_h, c_h)
    self.Linear = nn.Linear(c_h, c_out)
    self.inorm = InstanceNorm()
    self.act = nn.LeakyReLU()
    self.AdaIN = AdaIN()
  
  def forward(self, src, trg, return_c=False):
    y1, _, _ = src
    y2, mns, stds = trg
    mn, sd = self.inorm.get_mean_std(y2)
    c_affine = self.AdaIN(y1, mn, sd)
    y = self.conv(c_affine)
    y = self.act(y)
    for block, mn, std  in zip(self.conv_blocks, mns[::-1], stds[::-1]):
      y = block(y)
      y = self.AdaIN(y, mn, std)
    y = y.transpose(1,2)
    y, _ = self.gru(y)
    y = self.Linear(y)
    y = y.transpose(1,2)

    if return_c:
      return y, c
    
    return y

class InstanceNorm(nn.Module):
  def __init__(self, eps=1e-5):
    super(InstanceNorm, self).__init__()
    self.eps = eps

  def get_mean_std(self, x, mask=None):
        B, C = x.shape[:2]

        mn = x.view(B, C, -1).mean(-1)
        sd = (x.view(B, C, -1).var(-1) + self.eps).sqrt()
        mn = mn.view(B, C, 1)
        sd = sd.view(B, C, 1)
        return mn, sd


  def forward(self, x, get_mean_std=False):
      mean, std = self.get_mean_std(x)
      x = (x - mean) / std
      if get_mean_std:
          return x, mean, std
      else:
          return x

class Activation(nn.Module):
  def __init__(self, a=0.1):
    super(Activation, self).__init__()
    self.a = a

  def forward(self, x):
    x = 1/(1+torch.exp(-self.a*x))
    return x

class AgainVC(nn.Module):
  def __init__(self, encoder_params, decoder_params, activation_params):
    super(AgainVC, self).__init__()
    self.encoder = Encoder(**encoder_params)
    self.decoder = Decoder(**decoder_params)
    self.act = Activation(**activation_params)

  def forward(self, x, target=None):
    if target is None:
      target = torch.clone(x)
    
    enc, mns_enc, sds_enc = self.encoder(x)
    cond, mns_cond, sds_cond = self.encoder(target)
    
    enc = (self.act(enc), mns_enc, sds_enc)
    cond = (self.act(cond), mns_cond, sds_cond)
    y = self.decoder(enc, cond)

    return y


