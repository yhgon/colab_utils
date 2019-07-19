import IPython.display as ipd
import librosa
from librosa.display import specshow as specshow
import matplotlib.pyplot as plt
import numpy as np

def display_audio(filename, srt=22050, mono=True, sec=3):
    y_22k, sr_22k = librosa.load(filename, sr=22050, mono=mono)
    y_22k=y_22k[0:sec*sr_22k]
    ipd.display(ipd.Audio(y_22k, rate=sr_22k) )
    return y_22k, sr_22k
    
    
def print_mel_basis(y, sr=22050,  n_mels=80,  ms=10,  fmin=0, fmax=8000):
    melfb = librosa.filters.mel(sr=sr, n_fft=int(ms*sr/1000), n_mels=n_mels, fmin=fmin, fmax=fmax  )
    plt.figure(figsize=(4, 2))
    librosa.display.specshow(melfb, x_axis='linear')
    plt.ylabel('Mel filter')
    plt.title('Mel filter bank')
    plt.colorbar()
    plt.tight_layout()
    plt.show()    
    
def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom', interpolation='none')
        
def compare_plot(A,B):
    plot_data([A,B])

def show_plot(S, fsize=(8,8) ):
    plt.figure(figsize=fsize )
    print(type(S), S.shape )
    plt.imshow(S , origin='bottom' )
    plt.show()    

def preemphasis_np(x, factor=0.97): 
    return np.append(x[0], x[1:] - factor * x[:-1]) 

def preemphasis_scipy(x, factor=0.97):
    import scipy
    return scipy.signal.lfilter([1, -factor], [1], x)

def inv_preemphasis_scipy(x, factor=0.97):
    import scipy
    return scipy.signal.lfilter([1], [1, -factor], x)

def amp_to_dB(S):
    constant=20.
    dB = constant* np.log10(S)
    return dB

def inv_amp_to_dB(dB):
    constant=20.
    S = np.power(10, dB/constant )
    return S

def spec_normalize(S, min_level_db = -100):
    return np.clip((S -  min_level_db) / - min_level_db, 0, 1)

def inv_spec_normalize(S, min_level_db = -100 ):
    return (np.clip(S, 0, 1) * - min_level_db) +  min_level_db

def stft_set_parameters():
    n_fft = my_ms1
    hop_length = my_ms2
    win_length = my_ms1 
    return n_fft, hop_length, win_length 

def _stft(y):
    n_fft, hop_length, win_length = stft_set_parameters()
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y):
    _, hop_length, win_length = stft_set_parameters()
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)

def _griffin_lim(S):
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles)
    for i in range(my_niters):
        angles = np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)
    return y

def _torch_griffin_lim(magnitudes, stft_fn, n_iters=60):

    angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitudes.size())))
    angles = angles.astype(np.float32)
    angles = torch.autograd.Variable(torch.from_numpy(angles))
    signal = stft_fn.inverse(magnitudes, angles).squeeze(1)

    for i in range(n_iters):
        _, angles = stft_fn.transform(signal)
        signal = stft_fn.inverse(magnitudes, angles).squeeze(1)
    return signal
    


def get_mel_taco(y, sr=22050, n_mels=80, ms1=25,  ms2=10, fmin=0, fmax=8000):
    frs1=int(ms1*sr/1000)
    frs2=int(ms2*sr/1000)

    y_t = torch.from_numpy( np.array([y]))    
    y_t = y_t / torch.max(torch.abs(y_t))
        
    my_stft = TacotronSTFT(filter_length=frs1, hop_length=frs2,
                           win_length=frs1, sampling_rate=sr, n_mel_channels=n_mels,
                           mel_fmin=fmin, mel_fmax=fmax)
    
    S = my_stft.mel_spectrogram(y_t).squeeze()
    #S = my_stft.mel_spectrogram(y_t).cuda().half()  #  for waveglow
    print( " ref: {}k, length : {}s shape: {} frame_length : {}ms n_fft: {} ". format( sr/1000, len(y)/sr , S.shape, ms,  frs   ) ) 
    show_plot(S, fsize=(2,4) )
    return S
    
def get_mel_manual(y, sr=22050,  n_mels=80,  ms1=25, ms2=10,  fmin=0, fmax=8000, power=1.7):
    frs1=int(ms1*sr/1000) # for windows
    frs2=int(ms2*sr/1000) # for hop
    
    y = preemphasis_scipy(y)
    
    mag=  np.abs(librosa.stft(y, n_fft=frs1, hop_length=frs2))    
    mag_power =  mag**power
    print(mag_power.shape)
    mel_basis  = librosa.filters.mel(sr=sr, n_fft=frs1, n_mels=n_mels, fmin=fmin, fmax=fmax  ) # linear mag   
    S = np.dot(mel_basis, mag_power  ) # Mel
    
    S_dB=amp_to_dB(S)

    S_norm=spec_normalize(S_dB)
    
    print( " ref: {}k, length : {}s shape: {} win {}ms n_fft: {}frame hop {}ms {}frame overlap  ". format( sr/1000, len(y)/sr , S.shape, ms1, frs1, ms2, frs2   ) ) 
  
    show_plot(S_norm, fsize=(16,8) )
    return S_norm  
    
def float_to_pcm16(sig, dtype='int16'):
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")
    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)  

def save_wave_file(file_name, srt, data):
    from scipy.io import wavfile as wf
    data_float=data 
    data_int=float_to_pcm16(data_float)
    wf.write(filename=file_name, rate=srt, data=data_int.T) # scipy
    
def mag_to_wav(S_norm, sr=22050,  n_mels=80,  ms1=25, ms2=10,  fmin=0, fmax=8000, power=1.7):
    frs1=int(ms1*sr/1000) # for windows
    frs2=int(ms2*sr/1000) # for hop
    
    S_dB=inv_spec_normalize(S_norm)
    
    S=inv_amp_to_dB(S_dB)
    
    mel_basis  = librosa.filters.mel(sr=sr, n_fft=frs1, n_mels=n_mels, fmin=fmin, fmax=fmax  ) # linear mag   
    mag = np.dot( S.T, mel_basis) # Mel
    print(mag.shape)
    #y=griffin_lim(mag.T )
    
    my_stft = TacotronSTFT(filter_length=frs1, hop_length=frs2,
                           win_length=frs1, sampling_rate=sr, n_mel_channels=n_mels,
                           mel_fmin=fmin, mel_fmax=fmax)

    mag=np.array([mag.T]) 
    print(mag.shape)
    # synthesize with griffin lim
    mag=torch.from_numpy(mag).float()
    print(mag.shape)
    
    waveform = griffin_lim(torch.autograd.Variable(mag*100 ), my_stft.stft_fn, 60)
    
    
    waveform = waveform / torch.max(torch.abs(waveform))
    waveform = inv_preemphasis_scipy(waveform.cpu())
    save_wave_file(file_name=save_filename, srt=sr , data= waveform )

def get_mag_manual(y, sr=22050,  n_mels=80,  ms=10,   fmin=0, fmax=8000):
    frs=int(ms*sr/1000)
    
    y = y / max( abs(y))
    
    mag= np.abs(librosa.stft(y, n_fft=frs, hop_length=frs))**2
    #mag= np.abs(librosa.stft(y, n_fft=frs, hop_length=frs))
    
    print("mag",mag.shape)
    show_plot(mag.squeeze(), fsize=(8,8) )
    
    
    mel_basis  = librosa.filters.mel(sr=sr, n_fft=frs, n_mels=n_mels, fmin=fmin, fmax=fmax  ) # linear mag
    
    S = np.dot(mel_basis, mag) # Mel
    
    print("S",S.shape)
    show_plot(S.squeeze(), fsize=(8,8) )
    
    
    #S = librosa.feature.melspectrogram(y=y , sr=sr, n_mels=n_mels, n_fft=frs, hop_length=frs)

    
    my_stft = TacotronSTFT(filter_length=frs, hop_length=frs,
                           win_length=frs, sampling_rate=sr, n_mel_channels=n_mels,
                           mel_fmin=fmin, mel_fmax=fmax)
    
    S_gpu=torch.from_numpy( np.array([S]) )
    
    S_gpu=my_stft.spectral_normalize(S_gpu)

    print( " ref: {}k, length : {}s shape: {} frame_length : {}ms n_fft: {} ". format( sr/1000, len(y)/sr , S.shape, ms,  frs   ) ) 
    #print(S)
    S_cpu=S
    print("S_gpu",S_gpu.shape)
    show_plot(S_gpu.squeeze(), fsize=(8,8) )
    return mag

def get_mag_inverse_manual(S, sr=22050,  n_mels=80,  ms=10,   fmin=0, fmax=8000):
    frs=int(ms*sr/1000)
    #show_plot(S)
    my_stft = TacotronSTFT(filter_length=frs, hop_length=frs,
                           win_length=frs, sampling_rate=sr, n_mel_channels=n_mels,
                           mel_fmin=fmin, mel_fmax=fmax)

    S_gpu=torch.from_numpy( np.array(S) )
    
    S_cpu=my_stft.spectral_de_normalize(S_gpu)
    print("S_cpu",S_cpu.shape)
    #show_plot(S_cpu)  
    
    mel =   S_cpu.float().transpose( 0, 1)
    print("mel",mel.shape)
    #show_plot(mel)  

    mel_basis1  = librosa.filters.mel(sr=sr, n_fft=frs, n_mels=n_mels, fmin=fmin, fmax=fmax  )  
    mel_basis2  = my_stft.mel_basis
    print("mel_basis2.T",mel_basis1.T.shape)
    
    mag = np.dot( mel_basis1.T, mel.transpose(0,1) ) 
    #mag = torch.mm( mel,  mel_basis1.float()  )
    print("mag",mag.shape)
    #show_plot(mag)  
       
    mag_sqrt = np.sqrt(mag)
    print("mag_sqrt",mag_sqrt.shape)
    show_plot(mag_sqrt) # identical 
    
    mag_abs=np.abs(mag_sqrt) 
    
    return   mag_abs 


