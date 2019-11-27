import os 
import sys 
import librosa 
import numpy as np 
import matplotlib.pyplot as plt
import librosa.display
from gfile import  download_list as download_main  ########## assume gfile.py

def load_resample(filename, sr_org = 22050, sr_target = 16000 ):
    data, sr = librosa.load(filename , sr = sr_org )
    data_16k_polyphase  = librosa.core.resample(data, sr_org, sr_target, res_type='polyphase', fix=True, scale=False ) # fast and high quality
    return data_16k_polyphase

def display_fileinfo(check_data, check_sr ):
    data_mins = int(   len(check_data) / (check_sr) / (60) )  
    data_secs =  len(check_data) / (check_sr)  
    data_ksecs = round( len(check_data) / (check_sr) /(1000) , 2) 
    data_msams = round(   len(check_data) / (1000000) , 2) 
    data_ksams = int(   len(check_data) / (1000) )
    print( " duration : {} Min,   {} K {} sec,   {}M  {}K samples with sr {} ".format( data_mins,  data_ksecs, data_secs , data_msams ,data_ksams , sr )  )

def show_wave(data, sr , dur=10):   
    plt.figure( figsize=(16,1) )
    librosa.display.waveplot(data[0:sr*dur] , sr=sr, alpha=1) 
    plt.show()
    #ipd.display(ipd.Audio(data[0:sr*dur] , rate=sr, normalize=False )) #with 7.7.0
    
def gain_scaler(dB):
    constant=20.
    scaler = np.power(10, dB/constant )
    return scaler

def gain_dB(value):
    constant=20.
    dB = constant* np.log10(value)
    return dB

def volume_slider(signal, dB):
    signal = signal*gain_scaler(dB)
    return signal

def chunk_step1_v1(fname='/content/wavs_22khz/piracy_09_female_Linda_Johnson.wav', sr=22050, chunk_top_db=60, chunk_frame_length=512, chunk_hop_length=32):
    data, sr = librosa.load(fname, sr = sr  )

    plt.figure( figsize=(16,1) )
    librosa.display.waveplot(data[0:sr*10] , sr=sr, alpha=1) 
    plt.show()
    ipd.display(ipd.Audio(data[0:sr*10] , rate=sr, normalize=False ))

    y_result_level_1  = librosa.effects.split(y=data, top_db=chunk_top_db, frame_length=chunk_frame_length, hop_length=chunk_hop_length)
    #print ( y_result_level_1.shape, np.round( y_result_level_1 / sr, 2)  )   

    data = np.array( y_result_level_1 )
    dataset = pd.DataFrame({'vo_s': data[:, 0], 'vo_e': data[:, 1]})
    dataset['vo_d']     = dataset['vo_e'] - dataset['vo_s']
    dataset['vo_s_sec'] = dataset['vo_s']/sr
    dataset['vo_e_sec'] = dataset['vo_e']/sr  
    dataset['vo_d_sec'] = dataset['vo_d']/sr 

    new_data = dataset.query('vo_d_sec > 0.24  '  ).reset_index(drop=True) # 0.3/758, 0.4/743,  0.5/720,  0.6/704 , 0.7/687 
    #print(new_data.shape)

    new_data['si_s']     = new_data['vo_e'].shift(periods=1, fill_value=0)+1
    new_data['si_e']     = new_data['vo_s']-1
    new_data['si_d']     = new_data['si_e'] - new_data['si_s']

    new_data['si_s_sec'] = new_data['si_s']/sr
    new_data['si_e_sec'] = new_data['si_e']/sr  
    new_data['si_d_sec'] = new_data['si_d']/sr 
    #print(new_data.shape)

    new_data2 = new_data.query('si_d_sec > 0.4'  ).reset_index(drop=True) 
    #print(new_data2.shape)
    new_data2.drop(['vo_s','vo_e','vo_d','vo_s_sec','vo_e_sec','vo_d_sec','si_s_sec', 'si_d','si_e_sec'],axis=1)
    new_data2['vo_s'] = new_data2['si_e']+1
    new_data2['vo_e'] = new_data2['si_s'].shift(periods=-1,  fill_value=None)-1
    new_data2['vo_d'] = new_data2['vo_e'] - new_data2['vo_s']

    new_data2['vo_s_sec'] = new_data2['vo_s']/sr
    new_data2['vo_e_sec'] = new_data2['vo_e']/sr  
    new_data2['vo_d_sec'] = new_data2['vo_d']/sr 
    print(new_data2['vo_d_sec'].describe() )

    plt.hist = new_data2['vo_d_sec'].hist(bins=200)
    plt.title('duration of each voice')
    plt.show()
    plt.hist = new_data2['si_d_sec'].hist(bins=200)
    plt.title('duration of each silence')
    plt.show()
    return new_data2

def chunk_step1_v2(data , sr=16000, chunk_top_db=60, chunk_frame_length=512, chunk_hop_length=32, csv_filename='csv_example.csv', vo_query_value = 0.24,  si_query_value = 0.4):
    y_result_level_1  = librosa.effects.split( y=data, top_db=chunk_top_db, frame_length=chunk_frame_length, hop_length=chunk_hop_length)
    #print ( y_result_level_1.shape, np.round( y_result_level_1 / sr, 2)  )   

    data = np.array( y_result_level_1 )
    dataset = pd.DataFrame({'vo_s': data[:, 0], 'vo_e': data[:, 1]})
    dataset['vo_d']     = dataset['vo_e'] - dataset['vo_s']
    dataset['vo_s_sec'] = dataset['vo_s']/sr
    dataset['vo_e_sec'] = dataset['vo_e']/sr  
    dataset['vo_d_sec'] = dataset['vo_d']/sr 

    new_data = dataset.query('vo_d_sec > {}'.format(vo_query_value)  ).reset_index(drop=True) # 0.3/758, 0.4/743,  0.5/720,  0.6/704 , 0.7/687 
    #print(new_data.shape)

    new_data['si_s']     = new_data['vo_e'].shift(periods=1, fill_value=0)+1
    new_data['si_e']     = new_data['vo_s']-1
    new_data['si_d']     = new_data['si_e'] - new_data['si_s']

    new_data['si_s_sec'] = new_data['si_s']/sr
    new_data['si_e_sec'] = new_data['si_e']/sr  
    new_data['si_d_sec'] = new_data['si_d']/sr 
    #print(new_data.shape)

    new_data2 = new_data.query('si_d_sec > {}'.format(si_query_value)  ).reset_index(drop=True) 
    #print(new_data2.shape)
    new_data2.drop(['vo_s','vo_e','vo_d','vo_s_sec','vo_e_sec','vo_d_sec','si_s_sec', 'si_d','si_e_sec'],axis=1)
    new_data2['vo_s'] = new_data2['si_e']+1
    new_data2['vo_e'] = new_data2['si_s'].shift(periods=-1,  fill_value=None)-1
    new_data2['vo_d'] = new_data2['vo_e'] - new_data2['vo_s']

    new_data2['vo_s_sec'] = new_data2['vo_s']/sr
    new_data2['vo_e_sec'] = new_data2['vo_e']/sr  
    new_data2['vo_d_sec'] = new_data2['vo_d']/sr 
    print(new_data2['vo_d_sec'].describe() )

    plt.hist = new_data2['vo_d_sec'].hist(bins=200)
    plt.title('duration of each voice')
    plt.show()
    plt.hist = new_data2['si_d_sec'].hist(bins=200)
    plt.title('duration of each silence')
    plt.show()
    new_data2.to_csv(csv_filename, mode='w')
    return new_data2

def data_step1(filename, prefix='librivox_chunk_', max_len=1, work_dir = '/content/'):

    str_prefix  = "https://drive.google.com/file/d/"
    str_postfix = "/view?usp=sharing"
    str_space   = " "
    str_enter   = "\n"    
    with open(filename, 'r') as f_handle:
        i = 1
        for line in f_handle:
            if (i > max_len ) : 
                break
            id=line.replace(str_prefix, '').replace(str_postfix, '').replace(str_space, '').replace(str_enter, '')
            print(i, id, end ='' )
            str_filename_input = "" + str(line) + "" 
            str_filename_output = str(prefix) + str(i).zfill(6) 
            str_filename_output_wav_ext = str_filename_output +'.wav'
            str_filename_output_csv_ext = str_filename_output +'.csv'
            save_filename = os.path.join(work_dir, str_filename_output_wav_ext)
            print(save_filename)

            download_main( str_filename_input, str_filename_output_wav_ext, work_dir )           
            this_data = load_resample(save_filename, sr_org = 22050, sr_target = 16000 )
            show_wave(this_data, sr=16000, dur=10 )
            #chunk_step1_v2(data=this_data, sr=22050, chunk_top_db=60, chunk_frame_length=512, chunk_hop_length=32, csv_filename=str_filename_output_csv_ext, query_value=0.24)            
            chunk_step1_v2(data=this_data, sr=22050, chunk_top_db=10, chunk_frame_length=512, chunk_hop_length=32, csv_filename=str_filename_output_csv_ext,  vo_query_value = 0.24,  si_query_value = 0.4)
            i += 1
