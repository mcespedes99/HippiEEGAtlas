#    Author: Mauricio Cespedes Tenorio <mcespedes99@gmail.com>
#    This is the Python version of the code from: 
#    http://audition.ens.fr/adc/NoiseTools/src/NoiseTools/nt_zapline.m
#    which is an implementation of the proposed algorithm in,
#    Alain de Cheveigné,
#    ZapLine: A simple and effective method to remove power line artifacts,
#    NeuroImage, Volume 207, 2020, 116356, ISSN 1053-8119,
#    https://doi.org/10.1016/j.neuroimage.2019.116356.
import numpy as np
import scipy.signal
def square_filt(x,T,nIterations=1):
  #y=nt_smooth(x,T,nIterations,nodelayflag) - smooth by convolution with square window
  #
  #  y: smoothed data
  # 
  #  x: data to smooth
  #  T: samples, size of window (can be fractionary)
  #  nIterations: number of iterations of smoothing operation (large --> gaussian kernel)
  #

  integ=int(np.floor(T))
  frac=T-integ

  # if integ>=size(x,1);
  #     x=repmat(mean(x),[size(x,1),1,1,1]);
  #     return;
  # end

  # remove onset step
  mn = np.mean(x[0:integ+1,:], axis=0)
  x= x - mn

  if nIterations==1 and frac==0:
      # faster
      x=np.cumsum(x);
      x[T:,:]=x[T:,:]-x[0:-T,:]
      x=x/T
  else:
      # filter kernel
      B = np.concatenate((np.ones(integ), [frac]))/T
      for k in np.arange(1,nIterations):
          B=np.convolve(B, B)
          print('aqui')
      x=scipy.signal.lfilter(B, 1, x, axis=0)

  # restore DC
  x=x+mn
  return x

# Possible alternative to square filter
import numpy as np
import scipy.signal
import scipy.fftpack
def square_notch_filt(x, fline, srate, nHarmonics, plotting=False):
  #y=nt_smooth(x,T,nIterations,nodelayflag) - smooth by convolution with square window
  #
  #  y: smoothed data
  # 
  #  x: data to smooth
  #  T: samples, size of window (can be fractionary) 
  #  nIterations: number of iterations of smoothing operation (large --> gaussian kernel)
  #

  # if integ>=size(x,1);
  #     x=repmat(mean(x),[size(x,1),1,1,1]);
  #     return;
  # end
  fline = fline*srate
  # remove onset step
  mn = np.mean(x, axis=0)
  x= x - mn
  # Apply filter
  lower_trans = .1
  upper_trans = .1
  norder = 24
  filtorder = norder*np.round(srate/55)+1
  for n in np.arange(1, nHarmonics+1):
    if n>1:
      filtorder = int(filtorder/(n*0.6)) # Change 0.6 to a parameter
      if filtorder % 2 == 0:
        filtorder += 1
    f_harmonic = n*fline
    lower_bnd = f_harmonic-5
    upper_bnd = f_harmonic+5
    # filter kernel
    filter_shape = [ 1,1,0,0,1,1 ]
    filter_freqs = [ 0, lower_bnd*(1-lower_trans), lower_bnd, upper_bnd, \
                    upper_bnd+upper_bnd*upper_trans, srate/2 ]
    filter_kern = scipy.signal.firls(filtorder,filter_freqs,filter_shape,fs=srate)
    # Apply filter:
    x = scipy.signal.filtfilt(filter_kern,1,x, axis=0)
    if plotting:
      fig,ax2 = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
      # Power spectrum
      hz = np.linspace(0,srate/2,int(np.floor(len(filter_kern)/2)+1))
      filterpow = np.abs(scipy.fftpack.fft(filter_kern))**2
      ax2.plot(hz,filterpow[:len(hz)], 'ks-')
      plt.plot(filter_freqs,filter_shape,'ro-')
      ax2.set_xlim([0,srate/2])
      ax2.set_xlabel('Frequency (Hz)')
      ax2.set_ylabel('Filter gain')
      ax2.set_title('Frequency response')
      plt.show()
  # Plot to check
  # if plotting: # Needs fix

    # Filter kernel
    # fig,[ax1,ax2] = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
    # ax1.plot(filter_kern)
    # ax1.set_xlabel('Time points')
    # ax1.set_title('Filter kernel (firls)')
    # # Power spectrum
    # hz = np.linspace(0,srate/2,int(np.floor(len(filter_kern)/2)+1))
    # filterpow = np.abs(scipy.fftpack.fft(filter_kern))**2
    # ax2.plot(hz,filterpow[:len(hz)],'ks-')
    # ax2.set_xlim([0,srate/2])
    # ax2.set_xlabel('Frequency (Hz)')
    # ax2.set_ylabel('Filter gain')
    # ax2.set_title('Frequency response')
    # plt.show()
  # restore DC
  x=x+mn
  return x

import scipy.fftpack
def bias_fft(x,freq,nfft):
  #[c0,c1]=nt_bias_fft(x,freq,nfft) - covariance with and w/o filter bias
  #
  # x: data 
  # freq: row vector of normalized frequencies to keep (wrt sr)
  # nfft: fft size
  #
  # The filter has zeros at all frequencies except those immediately inferior
  # or superior to values in vector freq.
  # 
  # If freq has two rows, keep frequencies between corresponding values on
  # first and second row.
  #
  # NoiseTools

  if max(freq)>0.5: 
    raise Exception('frequencies should be <= 0.5')
  if nfft > x.shape[0]: 
    raise Exception('nfft too large')

  # Here the filter is built in the freq domain, which has a reflection
  # Left half of the filter
  filt=np.zeros(int(np.floor(nfft/2)+1))
  for k in np.arange(0,len(freq)):
      idx=int(freq[k]*nfft+0.5)
      filt[idx]=1

  filt=np.concatenate((filt,np.flipud(filt[0:-1])))

  ## now for convolution
  n = x.shape[0]
  k = len(filt)
  nConv = n + k -1
  half_kern = int( np.floor(k/2) )
  filt = scipy.fftpack.fft(scipy.fftpack.ifft(filt),nConv)

  # FFTs
  dataX = scipy.fftpack.fft(x, nConv, axis=0)

  # IFFT
  x_filt = np.multiply(dataX, filt.reshape(len(filt),1))
  x_filt = np.real( scipy.fftpack.ifft( x_filt , axis=0))
  x_filt = x_filt[half_kern:-half_kern,:]
  
  c0 = np.cov(x.T)
  c1 = np.cov(x_filt.T)

  # return x_filt
  return c0, c1


from numpy.linalg import eig
def eigen(A):
  # Calculates ordered real part of eigenvalues and eigenvectors
  [eigvals,eigvecs]=eig(A)
  eigvecs = np.real(eigvecs)
  eigvals = np.real(eigvals)
  idx = np.flipud(np.argsort(eigvals))
  eigvals = np.flipud(np.sort(eigvals))
  eigvecs = eigvecs[:,idx]
  return eigvals, eigvecs

import numpy as np
def dss(c0,c1):
  # Refer to de Cheveigné A, Parra LC. Joint decorrelation, a versatile
  # tool for multichannel data analysis. Neuroimage. 2014 Sep;98:487-505.
  # doi: 10.1016/j.neuroimage.2014.05.068. Epub 2014 Jun 2. PMID: 24990357.
  #[todss,pwr1,pwr2]=nt_dss0(c0,c1,keep1,keep2) - dss from covariance
  #
  # todss: matrix to convert data to normalized DSS components
  # pwr0: power per component (baseline)
  # pwr1: power per component (biased)
  #
  # c0: baseline covariance
  # c1: biased covariance
  # keep1: number of PCs to retain (default: all)
  # keep2: ignore PCs smaller than keep2 (default: 10.^-9)
  #

  if c0.shape != c1.shape: 
    raise Exception('C0 and C1 should have same size')
  if c0.shape[0] != c0.shape[1]:
    raise Exception('C0 should be square')

  # if any(find(isnan(c0)))
  #     error('NaN in c0');
  # end
  # if any(find(isnan(c1)))
  #     error('NaN in c1');
  # end
  # if any(find(isinf(c0)))
  #     error('INF in c0');
  # end
  # if any(find(isinf(c1)))
  #     error('INF in c1');
  # end
  # Eig vals and vecs from the unbiased covariance
  [eigvals0,eigvecs0] = eigen(c0)
  eigvals0 = np.abs(eigvals0)

  # apply PCA and whitening to the biased covariance
  N = np.diag(np.sqrt(1/(eigvals0)))    
  c2 = np.transpose(N) @ np.transpose(eigvecs0) @ c1 @ eigvecs0 @ N

  # matrix to convert PCA-whitened data to DSS
  [eigvals1,eigvecs1]= eigen(c2)

  # DSS matrix (raw data to normalized DSS)
  todss = eigvecs0*N*eigvecs1
  N2 = np.diag(np.transpose(todss) @ c0 @ todss)
  todss=todss*np.diag(1/np.sqrt(N2)) # adjust so that components are normalized
  return todss


def crosscov(x,y):
  c = np.transpose(x) @ y
  return c


def denoise_PCA(x, ref):
  mnx = np.mean(x, axis=0)
  x = x - mnx
  

  mnref = np.mean(ref)
  ref = ref - mnref

  # print(len(x))
  cref = ref.T @ ref
  cref = cref / len(x)

  # The crosscov matrix would be just a way of measuring how each channel of x
  # related to the reference signal
  cxref = crosscov(x,ref)
  cxref = cxref/len(x)


  # regression matrix of x on ref
  # PCA of regressor
  if cref.size>1:
    print('lolo')
    [eigenvalues, eigvecs]=eigen(cref)
    # cross-covariance between data and regressor PCs
    cxref = np.transpose(cxref)
    r = np.transpose(eigvecs) @ cxref
    eigenvalues = np.reshape(eigenvalues, (len(eigenvalues),1))
    r = np.multiply(r, 1/eigenvalues)
    r = eigvecs @ r
  else:
    [eigenvalues, eigvecs] = [1, 1]
    # cross-covariance between data and regressor PCs
    cxref = np.transpose(cxref)
    # print(cxref)
    r = cxref

  # TSPCA
  # Then here r is just how each channel from x (which represents the original noise) variates 
  # compare to ref (which is just a mx1 signal that represents the best the PL
  # noise accross channels). So it just means the weights of ref to 'reconstruct'
  # each channel from x. The result of ref*r would be constructing a signal per channel
  # that represents the best x based only in ref, which is the PL noise!!
  z = ref @ r
  # z = z/2
  y = x-z
  mny = np.mean(y, axis=0)
  y = y-mny
  return y


import warnings
import numpy as np
from sklearn.decomposition import PCA
def zapline(x, fline, srate, nremove=1, p={}, filt=2):
  #[y,yy]=nt_zapline(x,fline,nremove,p,plotflag) - remove power line artifact
  #
  #  y: denoised data
  #  yy: artifact
  #
  #  x: data
  #  fline: line frequency (normalized to sr)
  #  nremove: number of components to remove [default: 1]
  #  p: additional parameters:
  #    p.nfft: size of FFT [default:1024]
  #    p.nkeep: number of components to keep in DSS [default: all]
  #    p.niterations: number of iterations for smoothing filter
  #    p.fig1: figure to use for DSS score [default: 100]
  #    p.fig2: figure to use for results [default: 101]
  #  plotflag: plot
  #
  #Examples:
  #  nt_zapline(x,60/1000) 
  #    apply to x, assuming line frequency=60Hz and sampling rate=1000Hz, plot results
  #  nt_zapline(x,60/1000,4)
  #    same, removing 4 line-dominated components 
  #  p=[];p.nkeep=30; nt_zapline(x,60/1000,4,p);
  #    same, truncating PCs beyond the 30th to avoid overfitting
  #  [y,yy]=nt_zapline(x,60/1000)
  #    return cleaned data in y, noise in yy, don't plot
  #
  if p=={}:
    p['nfft'] = 1024
    p['nkeep'] = []
    p['niterations'] = 1

  # Handling arguments
  if x.size == 0:
    raise Exception("x data cannot be an empty array")
  # Assuming a shape nxm for x:
  if nremove>=x.shape[0]:
    raise Exception("Number of components cannot be larger than lenght of each signal")
  if fline>1/2:
    raise Exception('fline should be less than Nyquist')
  if x.shape[0]<p['nfft']:
    warnings.warn(f'reducing nfft to {str(x.shape[0])}')
    p['nfft']=2*np.floor(x.shape[0]/2)

  if filt==1:
    xx=square_filt(x,1/fline,p['niterations']) # cancels line_frequency and harmonics, light lowpass
  elif filt==2:
    xx=square_notch_filt(x,fline, srate, 3)
  if p['nkeep']==[]: 
    try:
      p['nkeep']=x.shape[1]
    except:
      p['nkeep'] = 1
  # reduce dimensionality to avoid overfitting
  x_rem = x-xx
  c = np.cov(x_rem.T)
  [eigenvalues, eigvecs]=eigen(c)
  # In python, each column of eigvecs represent an eigenvector. So you have to 
  # multiple x * eigvecs. Easy way to say it, a chunk version of eigvecs could
  # be nxk so the only way to multiply it is x*eigvecs
  # This just rotates the data according to the principal components
  xxxx = (x_rem) @ (eigvecs) 
  
  print('ca')
  # DSS to isolate line components from residual:
  nHarmonics=np.floor((1/2)/fline);
  [c0,c1]=bias_fft(xxxx, fline*np.arange(1,nHarmonics+1), p['nfft']);
  # print('c0')
  # print(c0)
  # print('c1')
  # print(c1)
  print('2')
  todss = dss(c0,c1);
  print('3')
  # This would be the projection of the noise to the main component of the biased
  # noise, which should represent the line noise.
  xxxx= xxxx @ todss[:,0:nremove] # line-dominated components. 
  # return xxxx
  # Denoise
  xxx =denoise_PCA(x-xx,xxxx); # project them out
  del xxxx

  # reconstruct clean signal
  y=xx+xxx
  del xx
  # del xxx
  yy=x-y

  return y