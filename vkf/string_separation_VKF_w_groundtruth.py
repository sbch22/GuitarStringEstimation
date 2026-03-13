from PyVKF.VoldKalmanFilter import vkf
import scipy.signal
import string
import numpy as np
import resource
import os
import csv
import tqdm
import glob
import soundfile

from librosa import resample
from tools.io import load_stacked_pitch_list_jams
from tools.utils import frame_and_win, overlap_add

SAVE_FILE_FLAG = True
TEMP_OUT_PATH = './test_audio/sep'
NHARMS = [20,20,20,20,30,40]

fs = 44100
fs_low = 16000
BLOCK_PROCESS = False
F0_REL_BW = True
FLIP_AUDIO = True

WINLEN_S = 6
WINLEN = WINLEN_S*fs #s
HOPLEN = WINLEN//2
WINLEN_DEC = WINLEN_S*fs_low
HOPLEN_DEC = WINLEN_DEC//2
STRING_LOW_FREQS = [82.41, 110.0, 146.83, 196.0,246.94, 329.63] #TODO: Check if copilot was right!
BW_PERCENT = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05] # [0.2, 0.2*2/3, 0.2*2/3*2/3, 0.1, 0.05, 0.05]
WIN = 'hann'
ONSETLookahead = 20e-3#open up activity window 8ms before onset!


def limit_memory(): 
    soft, hard = resource.getrlimit(resource.RLIMIT_AS) 
    resource.setrlimit(resource.RLIMIT_AS, (soft, hard)) 

def numpy_fixed_seed(ind=None):
    np.random.seed(42+ind)

def get_test_train_split(dataset_path: string, train_list: string, test_list: string):

    test_set_list=[]
    train_set_list=[]
    with open(test_list, mode ='r')as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
                test_set_list.append(lines)

    with open(train_list, mode ='r')as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
                train_set_list.append(lines)
                
    N_train_files = len(train_set_list)
    N_test_files = len(test_set_list)
    
    print("Number of train files: ", N_train_files)
    print("Number of test files: ", N_test_files)

    test_set_path_list = []
    test_set_anno_list = []
    for ii in range(N_test_files):
        test_strs = test_set_list[ii]
        test_file_path = os.path.join(dataset_path, os.path.sep.join(test_strs[2::2]))
        test_file_anno = os.path.join(dataset_path,'annotation', test_strs[-1].replace('_hex_cln.wav','.jams'))
        test_set_path_list.append(test_file_path)
        test_set_anno_list.append(test_file_anno)
    
    train_set_path_list = []
    train_set_anno_list = []    
    for ii in range(N_train_files):
        train_strs = train_set_list[ii]
        train_file_path = os.path.join(dataset_path, os.path.sep.join(train_strs[2::2]))
        train_file_anno = os.path.join(dataset_path,'annotation', train_strs[-1].replace('_hex_cln.wav','.jams'))
        train_set_path_list.append(train_file_path)
        train_set_anno_list.append(train_file_anno)
        
        
    return (train_set_path_list, train_set_anno_list), (test_set_path_list, test_set_anno_list)


def block_based_vkf(sig_blocked, sig_blocked_dec, f0_blocked, f0_blocked_dec, n_harms, fs, fs_dec, vkf_bandwidths, hop_length, hop_length_dec, audio_length, audio_length_dec, win_oa, win_oa_dec, block_processing_flag, B_fit_arr=None):

    n_frames = sig_blocked.shape[-1]
    n_frames_dec = sig_blocked_dec.shape[-1]
    n_strings = f0_blocked.shape[0]
    win_len = sig_blocked.shape[1]
    win_len_dec = sig_blocked_dec.shape[1]
    sig_blocked_vkf = np.zeros((n_strings,win_len,n_frames)) 
    sig_blocked_vkf_dec = np.zeros((int(n_strings/2),win_len_dec,n_frames_dec))
    string_sigs = np.zeros((n_strings, audio_length))
    string_sigs_dec = np.zeros((int(n_strings/2), audio_length_dec))
    
    for ii in tqdm.tqdm(range(n_frames_dec)):

        if FLIP_AUDIO:
            flip_flap=True
        sig_block_dec = sig_blocked_dec[...,ii].squeeze()
        f0_block_dec = f0_blocked_dec[...,ii]
        
        
        for jj in range(int(n_strings/2)):
      
            if block_processing_flag:
                harm_array = []
                for kk in range(1,n_harms[jj]+1):
                    harm_array.append((kk)*np.ones((win_len_dec,1)))
                harm_array = np.hstack(harm_array)#*f0_est_interp[:,None]
                harmonics_matrix = harm_array
            else:
                harm_array_siglen = []
                for kk in range(1,n_harms[jj]+1):
                    harm_array_siglen.append((kk)*np.ones((sig_block_dec.shape[0],1)))
                harm_array_siglen = np.hstack(harm_array_siglen)
                harmonics_matrix = harm_array_siglen

            f0s_dec = f0_block_dec[jj]
            f0_arr_dec = harmonics_matrix*f0s_dec[:,None]
            string_activity = f0s_dec>0
            string_activity_pos = np.where(string_activity)[0]
            activity_change = np.diff(string_activity_pos)
            if len(activity_change)>0:    
                activity_change = np.insert(activity_change, 0, string_activity_pos[0])
            activity_change_pos = np.where(activity_change>1)[0]
            onsets = string_activity_pos[activity_change_pos]
            onsets = np.unique(onsets)
            offsets = string_activity_pos[activity_change_pos-1]
            if len(activity_change)>0:
                offsets = np.append(offsets, string_activity_pos[-1]) #append last activity as offset
            offsets = np.unique(offsets)
            if len(activity_change)>0:
                if len(onsets)>0 and len(offsets)>0:    
                    if offsets[0]<onsets[0]:#if first activity is a offset => prepend onset at 0
                        onsets = np.insert(onsets,0,0)
                    if onsets[-1]>offsets[-1]:#if last activity is a onset => append offset at end
                        offsets = np.append(offsets,win_len_dec-1)
                if len(onsets)==1 and len(offsets)==0:
                    offsets = np.append(offsets,win_len_dec-1)
                if len(offsets)==1 and len(onsets)==0:
                    onsets = np.insert(onsets,0,0)
            f0s_at_onset = f0s_dec[onsets]
            f0s_at_offset = f0s_dec[offsets]
            assert len(f0s_at_onset)==len(f0s_at_offset), 'onset and offset f0s do not match'

            if F0_REL_BW:
                vkf_bw_dec = (f0s_dec[:,None]*BW_PERCENT[jj])
                vkf_bw_dec[vkf_bw_dec==0] = STRING_LOW_FREQS[jj]*BW_PERCENT[jj]#vkf_bandwidths[jj]
            else:
                vkf_bw_dec = vkf_bandwidths[jj]

            if FLIP_AUDIO:
                f0_arr_dec = np.flip(f0_arr_dec,axis=0) # flip f0s for each string!
                vkf_bw_dec = np.flip(vkf_bw_dec,axis=0)
                if len(onsets)>0:
                    old_onsets = onsets
                    old_offsets = offsets
                    onsets = np.flip((win_len_dec-1)-old_offsets,axis=0)
                    offsets = np.flip((win_len_dec-1)-old_onsets, axis=0)
                if flip_flap:
                    #only flip sig block once for each string iteration!
                    sig_block_dec = np.flip(sig_block_dec, axis=-1)
                    flip_flap=False
            
            string_vkf_dec = np.zeros((win_len_dec,))
            
            for start,stop in zip(onsets,offsets):
                old_start = start
                old_stop = stop
                if start>int(fs_dec*ONSETLookahead):
                    start = start-int(fs_dec*ONSETLookahead)
                    f0_arr_dec[start:old_start,:] = f0_arr_dec[old_start,:]
                    vkf_bw_dec[start:old_start,:] = vkf_bw_dec[old_start,:]
                if stop+int(fs_dec*ONSETLookahead)<win_len:
                    stop = stop+int(fs_dec*ONSETLookahead)
                    f0_arr_dec[old_stop:stop,:] = f0_arr_dec[old_stop-1,:]
                    vkf_bw_dec[old_stop:stop,:] = vkf_bw_dec[old_stop-1,:]
                    
                x_dec,c_dec,_ = vkf(sig_block_dec[start:stop], fs_dec, f0_arr_dec[start:stop,:], bw=vkf_bw_dec[start:stop,:], p=2)

                string_vkf_harm_dec = np.real(x_dec*c_dec)
                temp_string_vkf_dec = string_vkf_harm_dec.sum(-1)
                string_vkf_dec[start:stop] = temp_string_vkf_dec#[old_start-start:]
                
            if FLIP_AUDIO:
                sig_blocked_vkf_dec[jj,:,ii] = np.flip(string_vkf_dec, axis=0)
            else:
                sig_blocked_vkf_dec[jj,:,ii] = string_vkf_dec
    
    
    for ii in tqdm.tqdm(range(n_frames)):

        if FLIP_AUDIO:
            flip_flap=True
            flip_toggle = True

        sig_block = sig_blocked[...,ii].squeeze()
        f0_block = f0_blocked[...,ii]
        
        
        for jj in range(int(n_strings/2),n_strings):
            
            if block_processing_flag:
                harm_array = []
                for kk in range(1,n_harms[jj]+1):
                    harm_array.append((kk)*np.ones((win_len,1)))
                harm_array = np.hstack(harm_array)#*f0_est_interp[:,None]
                harmonics_matrix = harm_array
            else:
                harm_array_siglen = []
                for kk in range(1,n_harms[jj]+1):
                    harm_array_siglen.append((kk)*np.ones((sig_block.shape[0],1)))
                harm_array_siglen = np.hstack(harm_array_siglen)
                harmonics_matrix = harm_array_siglen            
            
            if jj == 5 and FLIP_AUDIO:
                flip_toggle=False
                sig_block = np.flip(sig_block, axis=-1) # flip signal back for last string!
                
            f0s = f0_block[jj]
            f0_arr = harmonics_matrix*f0s[:,None]
            string_activity = f0s>0
            string_activity_pos = np.where(string_activity)[0]
            activity_change = np.diff(string_activity_pos)
            if len(activity_change)>0:    
                activity_change = np.insert(activity_change, 0, string_activity_pos[0])
            activity_change_pos = np.where(activity_change>1)[0]
            onsets = string_activity_pos[activity_change_pos]
            onsets = np.unique(onsets)
            offsets = string_activity_pos[activity_change_pos-1]
            if len(activity_change)>0:
                offsets = np.append(offsets, string_activity_pos[-1]) #append last activity as offset
            offsets = np.unique(offsets)
            if len(activity_change)>0:
                if len(onsets)>0 and len(offsets)>0:    
                    if offsets[0]<onsets[0]:#if first activity is a offset => prepend onset at 0
                        onsets = np.insert(onsets,0,0)
                    if onsets[-1]>offsets[-1]:#if last activity is a onset => append offset at end
                        offsets = np.append(offsets,win_len-1)
                if len(onsets)==1 and len(offsets)==0:
                    offsets = np.append(offsets,win_len-1)
                if len(offsets)==1 and len(onsets)==0:
                    onsets = np.insert(onsets,0,0)
            f0s_at_onset = f0s[onsets]
            f0s_at_offset = f0s[offsets]
            assert len(f0s_at_onset)==len(f0s_at_offset), 'onset and offset f0s do not match'
            
            if F0_REL_BW:
                vkf_bw = (f0s[:,None]*BW_PERCENT[jj])
                vkf_bw[vkf_bw==0] = STRING_LOW_FREQS[jj]*BW_PERCENT[jj]#vkf_bandwidths[jj]
            else:
                vkf_bw = vkf_bandwidths[jj]

            if FLIP_AUDIO and flip_toggle:
                f0_arr = np.flip(f0_arr,axis=0) # flip f0s for each string!
                vkf_bw = np.flip(vkf_bw,axis=0)
                if len(onsets)>0:
                    old_onsets = onsets
                    old_offsets = offsets
                    onsets = np.flip((win_len-1)-old_offsets,axis=0)
                    offsets = np.flip((win_len-1)-old_onsets, axis=0)
                if flip_flap:
                    #only flip sig block once for each string iteration!
                    sig_block = np.flip(sig_block, axis=-1)
                    flip_flap=False

            string_vkf = np.zeros((win_len,))
            
            for start,stop in zip(onsets,offsets):
                old_start = start
                old_stop = stop
                if start>int(fs*ONSETLookahead):
                    start = start-int(fs*ONSETLookahead)
                    f0_arr[start:old_start,:] = f0_arr[old_start,:]
                    vkf_bw[start:old_start,:] = vkf_bw[old_start,:]
                if stop+int(fs*ONSETLookahead)<win_len:
                    stop = stop+int(fs*ONSETLookahead)
                    f0_arr[old_stop:stop,:] = f0_arr[old_stop-1,:]
                    vkf_bw[old_stop:stop,:] = vkf_bw[old_stop-1,:]
                    
                x,c,_ = vkf(sig_block[start:stop], fs, f0_arr[start:stop,:], bw=vkf_bw[start:stop,:], p=2)

                string_vkf_harm = np.real(x*c)
                temp_string_vkf = string_vkf_harm.sum(-1)
                string_vkf[start:stop] = temp_string_vkf#[old_start-start:]
            
            if FLIP_AUDIO and flip_toggle:
                sig_blocked_vkf[jj,:,ii] = np.flip(string_vkf, axis=0)
            else:
                sig_blocked_vkf[jj,:,ii] = string_vkf
    if block_processing_flag:
        overlap_add(string_sigs_dec, sig_blocked_vkf_dec*win_oa_dec[None,:,None], hop_length_dec)
        overlap_add(string_sigs, sig_blocked_vkf*win_oa[None,:,None], hop_length)
    else:
        string_sigs_dec = np.squeeze(sig_blocked_vkf_dec)
        string_sigs = np.squeeze(sig_blocked_vkf)
        

    string_sigs_low = resample(string_sigs_dec, orig_sr=fs_dec, target_sr=fs, fix=True)
    string_sigs[:int(n_strings/2),:] = string_sigs_low[:,:audio_length]
    return string_sigs


def main():
    
    limit_memory()#30 GB

    if WIN == 'sqrt_hann':
        win_block = np.sqrt(scipy.signal.windows.hann(WINLEN, sym=False))
        win_block_dec = np.sqrt(scipy.signal.windows.hann(WINLEN_DEC, sym=False))
        win_oa = win_block
        win_oa_dec = win_block_dec
        
    elif WIN == 'tukey_win':
        tukey_alpha = 0.5
        win_block = scipy.signal.windows.tukey(WINLEN, alpha=tukey_alpha, sym=False)
        win_oa = np.ones_like(win_block)
        win_block_dec = scipy.signal.windows.tukey(WINLEN_DEC, alpha=tukey_alpha, sym=False)
        win_oa_dec = np.ones_like(win_block_dec)
        overlap_percent = tukey_alpha/2
        global HOPLEN 
        HOPLEN = int(WINLEN*(1-overlap_percent))
        global HOPLEN_DEC
        HOPLEN_DEC = int(WINLEN_DEC*(1-overlap_percent))
    else:
        win_block = scipy.signal.windows.hann(WINLEN, sym=False)
        win_oa = np.ones_like(win_block)
        win_block_dec = scipy.signal.windows.hann(WINLEN_DEC, sym=False)
        win_oa_dec = np.ones_like(win_block_dec)



    test_file_path = glob.glob('./test_audio/*.wav')[0]
    test_file_anno = glob.glob('./test_audio/*.jams')[0]

    test_audio, fs = soundfile.read(test_file_path)
    gt_audio = test_audio.copy()
    test_audio = np.sum(test_audio, axis=-1)[:,None]
    test_audio_dec = resample(test_audio.T, orig_sr=fs, target_sr=fs_low, fix=True).T

    tax = np.linspace(0,test_audio.shape[0]/fs,test_audio.shape[0])
    tax_dec = np.linspace(0,test_audio_dec.shape[0]/fs_low,test_audio_dec.shape[0])

    pitch_list = load_stacked_pitch_list_jams(test_file_anno,times=tax,uniform=True)
    pitch_time_string_0 = pitch_list['0']
    pitch_time_string_1 = pitch_list['1']
    pitch_time_string_2 = pitch_list['2']
    pitch_time_string_3 = pitch_list['3']
    pitch_time_string_4 = pitch_list['4']
    pitch_time_string_5 = pitch_list['5']   
    string_pitches = np.stack([pitch_time_string_0[1],pitch_time_string_1[1],pitch_time_string_2[1],pitch_time_string_3[1],pitch_time_string_4[1],pitch_time_string_5[1]])
    
    pitch_list_dec = load_stacked_pitch_list_jams(test_file_anno,times=tax_dec,uniform=True)
    pitch_time_string_0_dec = pitch_list_dec['0']
    pitch_time_string_1_dec = pitch_list_dec['1']
    pitch_time_string_2_dec = pitch_list_dec['2']
    pitch_time_string_3_dec = pitch_list_dec['3']
    pitch_time_string_4_dec = pitch_list_dec['4']
    pitch_time_string_5_dec = pitch_list_dec['5']
    string_pitches_dec = np.stack([pitch_time_string_0_dec[1],pitch_time_string_1_dec[1],pitch_time_string_2_dec[1],pitch_time_string_3_dec[1],pitch_time_string_4_dec[1],pitch_time_string_5_dec[1]])
    
    
    n_frames = int(np.ceil(tax.shape[-1]/HOPLEN))
    n_frames_dec = int(np.ceil(tax_dec.shape[-1]/HOPLEN_DEC))
    print("full-band length: "+str(test_audio.shape[0])+" samples")
    print("low-band length: "+str(test_audio_dec.shape[0])+" samples")
    if BLOCK_PROCESS:
        sig_block = frame_and_win(test_audio.T, frame_length=WINLEN, hop_length=HOPLEN, axis=-1, n_frames=n_frames, win=win_block)
        sig_block_dec = frame_and_win(test_audio_dec.T, frame_length=WINLEN_DEC, hop_length=HOPLEN_DEC, axis=-1, n_frames=n_frames_dec, win=win_block_dec)
        sig_block_six_string = frame_and_win(gt_audio.T, frame_length=WINLEN, hop_length=HOPLEN, axis=-1, n_frames=n_frames, win=win_block)
    else:
        sig_block = test_audio.T[...,None]
        sig_block_dec = test_audio_dec.T[...,None]
        sig_block_six_string = gt_audio.T[...,None]
    vkf_bandwidths = BW_PERCENT*np.array(STRING_LOW_FREQS)

    oa_sig = np.zeros(test_audio.T.shape)
    oa_sig_six_string = np.zeros(gt_audio.T.shape) 
    if BLOCK_PROCESS:
        overlap_add(oa_sig, sig_block*win_oa[None,:,None], HOPLEN)
        overlap_add(oa_sig_six_string, sig_block_six_string*win_oa[None,:,None], HOPLEN)
    else:
        oa_sig = sig_block[...,0]
        oa_sig_six_string = sig_block_six_string[...,0]
    filename = test_file_path.split(os.path.sep)[-1]

    os.makedirs(os.path.join(TEMP_OUT_PATH, filename.split('.wav')[0]), exist_ok=True)
    soundfile.write(os.path.join(TEMP_OUT_PATH, filename.split('.wav')[0], filename.replace('.wav','_overlap_add.wav')), oa_sig.T, fs)
    soundfile.write(os.path.join(TEMP_OUT_PATH, filename.split('.wav')[0], filename.replace('.wav','_overlap_add_six_string.wav')), oa_sig_six_string.T, fs)
    soundfile.write(os.path.join(TEMP_OUT_PATH, filename.split('.wav')[0], filename.replace('.wav','_original.wav')), test_audio, fs)

    if BLOCK_PROCESS:
        f0_blocked = frame_and_win(string_pitches, frame_length=WINLEN, hop_length=HOPLEN, axis=-1, n_frames=n_frames, win=np.ones(WINLEN))
        f0_blocked_dec = frame_and_win(string_pitches_dec, frame_length=WINLEN_DEC, hop_length=HOPLEN_DEC, axis=-1, n_frames=n_frames_dec, win=np.ones(WINLEN_DEC))
    else:
        f0_blocked = string_pitches[...,None]
        f0_blocked_dec = string_pitches_dec[...,None]
    

    vkf_filtered_signal = block_based_vkf(sig_block, sig_block_dec, f0_blocked, f0_blocked_dec, NHARMS, fs, fs_low, vkf_bandwidths, hop_length=HOPLEN, hop_length_dec=HOPLEN_DEC, audio_length=test_audio.shape[0], audio_length_dec=test_audio_dec.shape[0], win_oa=win_oa, win_oa_dec=win_oa_dec, block_processing_flag=BLOCK_PROCESS)
        
    if SAVE_FILE_FLAG:
        os.makedirs(TEMP_OUT_PATH, exist_ok=True)
        for id, string_sig in enumerate(vkf_filtered_signal):
            soundfile.write(os.path.join(TEMP_OUT_PATH, filename.split('.wav')[0], filename.replace('.wav','_string_'+str(id)+'.wav')), string_sig, fs)
        soundfile.write(os.path.join(TEMP_OUT_PATH, filename.split('.wav')[0], filename.replace('.wav','_reconst.wav')), np.sum(vkf_filtered_signal,0), fs)
    


if  __name__ == '__main__':
    main()