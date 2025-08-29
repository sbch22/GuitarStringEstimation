import os
import sys

from mpmath import linspace

sys.path.append(os.path.abspath(''))

from collections import Counter
import argparse
import torch

import numpy as np
import scipy
from scipy import signal
import matplotlib.pyplot as plt
import soundfile as sf
import json
from collections import namedtuple
import math
from scipy.fft import rfft
from scipy.fft import rfftfreq
from scipy.signal import find_peaks
from typing import Tuple, Dict, Literal, List, Optional
import torchaudio
import sounddevice as sd
from scipy.stats import norm
import seaborn as sns
import dataclasses
from utils.note_event_dataclasses import matchNote, stringNote

import numpy as np
import math
from scipy.stats import gaussian_kde, norm
from scipy.stats import wasserstein_distance

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from typing import List
from scipy.stats import gaussian_kde, norm, wasserstein_distance






from model.init_train import initialize_trainer, update_config
from utils.task_manager import TaskManager
from config.vocabulary import drum_vocab_presets
from utils.utils import str2bool
from utils.utils import Timer
from utils.audio import slice_padded_array
from utils.note2event import mix_notes
from utils.event2note import merge_zipped_note_events_and_ties_to_notes
from utils.utils import write_model_output_as_midi, write_err_cnt_as_json
from model.ymt3 import YourMT3


def load_model_checkpoint(args=None):
    parser = argparse.ArgumentParser(description="YourMT3")
    # General
    parser.add_argument('exp_id', type=str,
                        help='A unique identifier for the experiment is used to resume training. The "@" symbol can be used to load a specific checkpoint.')
    parser.add_argument('-p', '--project', type=str, default='ymt3', help='project name')
    parser.add_argument('-ac', '--audio-codec', type=str, default=None,
                        help='audio codec (default=None). {"spec", "melspec"}. If None, default value defined in config.py will be used.')
    parser.add_argument('-hop', '--hop-length', type=int, default=None,
                        help='hop length in frames (default=None). {128, 300} 128 for MT3, 300 for PerceiverTFIf None, default value defined in config.py will be used.')
    parser.add_argument('-nmel', '--n-mels', type=int, default=None,
                        help='number of mel bins (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-if', '--input-frames', type=int, default=None,
                        help='number of audio frames for input segment (default=None). If None, default value defined in config.py will be used.')
    # Model configurations
    parser.add_argument('-sqr', '--sca-use-query-residual', type=str2bool, default=None,
                        help='sca use query residual flag. Default follows config.py')
    parser.add_argument('-enc', '--encoder-type', type=str, default=None,
                        help="Encoder type. 't5' or 'perceiver-tf' or 'conformer'. Default is 't5', following config.py.")
    parser.add_argument('-dec', '--decoder-type', type=str, default=None,
                        help="Decoder type. 't5' or 'multi-t5'. Default is 't5', following config.py.")
    parser.add_argument('-preenc', '--pre-encoder-type', type=str, default='default',
                        help="Pre-encoder type. None or 'conv' or 'default'. By default, t5_enc:None, perceiver_tf_enc:conv, conformer:None")
    parser.add_argument('-predec', '--pre-decoder-type', type=str, default='default',
                        help="Pre-decoder type. {None, 'linear', 'conv1', 'mlp', 'group_linear'} or 'default'. Default is {'t5': None, 'perceiver-tf': 'linear', 'conformer': None}.")
    parser.add_argument('-cout', '--conv-out-channels', type=int, default=None,
                        help='Number of filters for pre-encoder conv layer. Default follows "model_cfg" of config.py.')
    parser.add_argument('-tenc', '--task-cond-encoder', type=str2bool, default=True,
                        help='task conditional encoder (default=True). True or False')
    parser.add_argument('-tdec', '--task-cond-decoder', type=str2bool, default=True,
                        help='task conditional decoder (default=True). True or False')
    parser.add_argument('-df', '--d-feat', type=int, default=None,
                        help='Audio feature will be projected to this dimension for Q,K,V of T5 or K,V of Perceiver (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-pt', '--pretrained', type=str2bool, default=False,
                        help='pretrained T5(default=False). True or False')
    parser.add_argument('-b', '--base-name', type=str, default="google/t5-v1_1-small",
                        help='base model name (default="google/t5-v1_1-small")')
    parser.add_argument('-epe', '--encoder-position-encoding-type', type=str, default='default',
                        help="Positional encoding type of encoder. By default, pre-defined PE for T5 or Perceiver-TF encoder in config.py. For T5: {'sinusoidal', 'trainable'}, conformer: {'rotary', 'trainable'}, Perceiver-TF: {'trainable', 'rope', 'alibi', 'alibit', 'None', '0', 'none', 'tkd', 'td', 'tk', 'kdt'}.")
    parser.add_argument('-dpe', '--decoder-position-encoding-type', type=str, default='default',
                        help="Positional encoding type of decoder. By default, pre-defined PE for T5 in config.py. {'sinusoidal', 'trainable'}.")
    parser.add_argument('-twe', '--tie-word-embedding', type=str2bool, default=None,
                        help='tie word embedding (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-el', '--event-length', type=int, default=None,
                        help='event length (default=None). If None, default value defined in model cfg of config.py will be used.')
    # Perceiver-TF configurations
    parser.add_argument('-dl', '--d-latent', type=int, default=None,
                        help='Latent dimension of Perceiver. On T5, this will be ignored (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-nl', '--num-latents', type=int, default=None,
                        help='Number of latents of Perceiver. On T5, this will be ignored (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-dpm', '--perceiver-tf-d-model', type=int, default=None,
                        help='Perceiver-TF d_model (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-npb', '--num-perceiver-tf-blocks', type=int, default=None,
                        help='Number of blocks of Perceiver-TF. On T5, this will be ignored (default=None). If None, default value defined in config.py.')
    parser.add_argument('-npl', '--num-perceiver-tf-local-transformers-per-block', type=int, default=None,
                        help='Number of local layers per block of Perceiver-TF. On T5, this will be ignored (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-npt', '--num-perceiver-tf-temporal-transformers-per-block', type=int, default=None,
                        help='Number of temporal layers per block of Perceiver-TF. On T5, this will be ignored (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-atc', '--attention-to-channel', type=str2bool, default=None,
                        help='Attention to channel flag of Perceiver-TF. On T5, this will be ignored (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-ln', '--layer-norm-type', type=str, default=None,
                        help='Layer normalization type (default=None). {"layer_norm", "rms_norm"}. If None, default value defined in config.py will be used.')
    parser.add_argument('-ff', '--ff-layer-type', type=str, default=None,
                        help='Feed forward layer type (default=None). {"mlp", "moe", "gmlp"}. If None, default value defined in config.py will be used.')
    parser.add_argument('-wf', '--ff-widening-factor', type=int, default=None,
                        help='Feed forward layer widening factor for MLP/MoE/gMLP (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-nmoe', '--moe-num-experts', type=int, default=None,
                        help='Number of experts for MoE (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-kmoe', '--moe-topk', type=int, default=None,
                        help='Top-k for MoE (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-act', '--hidden-act', type=str, default=None,
                        help='Hidden activation function (default=None). {"gelu", "silu", "relu", "tanh"}. If None, default value defined in config.py will be used.')
    parser.add_argument('-rt', '--rotary-type', type=str, default=None,
                        help='Rotary embedding type expressed in three letters. e.g. ppl: "pixel" for SCA and latents, "lang" for temporal transformer. If None, use config.')
    parser.add_argument('-rk', '--rope-apply-to-keys', type=str2bool, default=None,
                        help='Apply rope to keys (default=None). If None, use config.')
    parser.add_argument('-rp', '--rope-partial-pe', type=str2bool, default=None,
                        help='Whether to apply RoPE to partial positions (default=None). If None, use config.')
    # Decoder configurations
    parser.add_argument('-dff', '--decoder-ff-layer-type', type=str, default=None,
                        help='Feed forward layer type of decoder (default=None). {"mlp", "moe", "gmlp"}. If None, default value defined in config.py will be used.')
    parser.add_argument('-dwf', '--decoder-ff-widening-factor', type=int, default=None,
                        help='Feed forward layer widening factor for decoder MLP/MoE/gMLP (default=None). If None, default value defined in config.py will be used.')
    # Task and Evaluation configurations
    parser.add_argument('-tk', '--task', type=str, default='mt3_full_plus',
                        help='tokenizer type (default=mt3_full_plus). See config/task.py for more options.')
    parser.add_argument('-epv', '--eval-program-vocab', type=str, default=None,
                        help='evaluation vocabulary (default=None). If None, default vocabulary of the data preset will be used.')
    parser.add_argument('-edv', '--eval-drum-vocab', type=str, default=None,
                        help='evaluation vocabulary for drum (default=None). If None, default vocabulary of the data preset will be used.')
    parser.add_argument('-etk', '--eval-subtask-key', type=str, default='default',
                        help='evaluation subtask key (default=default). See config/task.py for more options.')
    parser.add_argument('-t', '--onset-tolerance', type=float, default=0.05, help='onset tolerance (default=0.05).')
    parser.add_argument('-os', '--test-octave-shift', type=str2bool, default=False,
                        help='test optimal octave shift (default=False). True or False')
    parser.add_argument('-w', '--write-model-output', type=str2bool, default=True,
                        help='write model test output to file (default=False). True or False')
    # Trainer configurations
    parser.add_argument('-pr', '--precision', type=str, default="bf16-mixed",
                        help='precision (default="bf16-mixed") {32, 16, bf16, bf16-mixed}')
    parser.add_argument('-st', '--strategy', type=str, default='auto',
                        help='strategy (default=auto). auto or deepspeed or ddp')
    parser.add_argument('-n', '--num-nodes', type=int, default=1, help='number of nodes (default=1)')
    parser.add_argument('-g', '--num-gpus', type=str, default='auto', help='number of gpus (default="auto")')
    parser.add_argument('-wb', '--wandb-mode', type=str, default="disabled",
                        help='wandb mode for logging (default=None). "disabled" or "online" or "offline". If None, default value defined in config.py will be used.')
    # Debug
    parser.add_argument('-debug', '--debug-mode', type=str2bool, default=False,
                        help='debug mode (default=False). True or False')
    parser.add_argument('-tps', '--test-pitch-shift', type=int, default=None,
                        help='use pitch shift when testing. debug-purpose only. (default=None). semitone in int.')
    args = parser.parse_args(args)
    # yapf: enable
    if torch.__version__ >= "1.13":
        torch.set_float32_matmul_precision("high")
    args.epochs = None

    # Initialize and update config
    _, _, dir_info, shared_cfg = initialize_trainer(args, stage='test')
    shared_cfg, audio_cfg, model_cfg = update_config(args, shared_cfg, stage='test')

    if args.eval_drum_vocab != None:  # override eval_drum_vocab
        eval_drum_vocab = drum_vocab_presets[args.eval_drum_vocab]

    # Initialize task manager
    tm = TaskManager(task_name=args.task,
                     max_shift_steps=int(shared_cfg["TOKENIZER"]["max_shift_steps"]),
                     debug_mode=args.debug_mode)
    print(f"Task: {tm.task_name}, Max Shift Steps: {tm.max_shift_steps}")

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = YourMT3(
        audio_cfg=audio_cfg,
        model_cfg=model_cfg,
        shared_cfg=shared_cfg,
        optimizer=None,
        task_manager=tm,  # tokenizer is a member of task_manager
        eval_subtask_key=args.eval_subtask_key,
        write_output_dir=dir_info["lightning_dir"] if args.write_model_output or args.test_octave_shift else None
    ).to(device)
    checkpoint = torch.load(dir_info["last_ckpt_path"], map_location=device)
    state_dict = checkpoint['state_dict']
    new_state_dict = {k: v for k, v in state_dict.items() if 'pitchshift' not in k}

    model.load_state_dict(new_state_dict, strict=False)

    return model.eval()


def transcribe(model, audio_info):
    t = Timer()

    # Converting Audio
    t.start()
    audio, sr = torchaudio.load(uri=audio_info['filepath'])
    audio = torch.mean(audio, dim=0).unsqueeze(0)
    audio = torchaudio.functional.resample(audio, sr, model.audio_cfg['sample_rate'])
    audio_segments = slice_padded_array(audio, model.audio_cfg['input_frames'], model.audio_cfg['input_frames'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_segments = torch.from_numpy(audio_segments.astype('float32')).to(device).unsqueeze(1)  # (n_seg, 1, seg_sz)
    t.stop();
    t.print_elapsed_time("converting audio");

    # Inference
    t.start()
    pred_token_arr, _ = model.inference_file(bsz=8, audio_segments=audio_segments)
    t.stop();
    t.print_elapsed_time("model inference");

    # Post-processing
    t.start()
    num_channels = model.task_manager.num_decoding_channels
    n_items = audio_segments.shape[0]
    start_secs_file = [model.audio_cfg['input_frames'] * i / model.audio_cfg['sample_rate'] for i in range(n_items)]
    pred_notes_in_file = []
    n_err_cnt = Counter()
    for ch in range(num_channels):
        pred_token_arr_ch = [arr[:, ch, :] for arr in pred_token_arr]  # (B, L)
        zipped_note_events_and_tie, list_events, ne_err_cnt = model.task_manager.detokenize_list_batches(
            pred_token_arr_ch, start_secs_file, return_events=True)
        pred_notes_ch, n_err_cnt_ch = merge_zipped_note_events_and_ties_to_notes(zipped_note_events_and_tie)
        pred_notes_in_file.append(pred_notes_ch)
        n_err_cnt += n_err_cnt_ch
    pred_notes = mix_notes(pred_notes_in_file)  # This is the mixed notes from all channels

    # Write MIDI
    # output_dir = 'Users/simonbuechner/Documents/Studium/AKT/3.Semester_GRAZ_WS2425/Toningenieur-Projekt/dev/YourMT3_evaluation/amt/content/'
    # print(f"Transcribe working directory: {os.getcwd()}") --> src/
    output_directory = '../content/'

    output_file = write_model_output_as_midi(pred_notes, output_directory,
                                             audio_info['track_name'], model.midi_output_inverse_vocab)
    t.stop();
    t.print_elapsed_time("post processing");
    # output_file =  os.path.join(output_file, audio_info['track_name']  + '.mid')

    # output_directory = os.path.abspath(midifile)
    # print(f"Resolved output directory: {output_directory}")
    # midifile = os.path.join(midifile, audio_info['track_name'] + '.mid')

    output_file = os.path.abspath(output_file)
    # assert os.path.exists(output_directory)
    assert os.path.exists(output_file)

    return output_file


def prepare_media(audio_array: torch.Tensor, sample_rate: int) -> Dict:
    num_channels, num_frames = audio_array.shape if audio_array.ndim == 2 else (1, audio_array.shape[0])

    return {
        "sample_rate": sample_rate,
        "bits_per_sample": 16,  # Torchaudio verwendet standardmäßig 16 Bit für PCM-Audio
        "num_channels": num_channels,
        "num_frames": num_frames,
        "duration": num_frames / sample_rate,
        "encoding": "pcm"  # Typische Codierung, wenn es ein Array ist
    }


def process_audioFile_notes(model, audio_file, sample_rate: int):
    audio_tensor = torch.from_numpy(audio_file).float()  # Sicherstellen, dass es float ist
    audio_info = prepare_media(audio_tensor, sample_rate)
    print(audio_info)
    pred_notes = transcribe_notes(model, audio_info, audio_tensor, sample_rate)

    # return policy
    return pred_notes


def transcribe_notes(model, audio_info, audio_tensor: torch.Tensor, sample_rate: int):
    t = Timer()

    # Converting Audio
    t.start()

    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)  # Fügt eine Kanaldimension hinzu (1, n)

    # Resampling auf die Modellanforderung
    audio_tensor = torchaudio.functional.resample(audio_tensor, sample_rate, model.audio_cfg['sample_rate'])

    # Slice Audio in passende Segmente
    audio_segments = slice_padded_array(audio_tensor, model.audio_cfg['input_frames'], model.audio_cfg['input_frames'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_segments = torch.from_numpy(audio_segments.astype('float32')).to(device).unsqueeze(1)  # (n_seg, 1, seg_sz)
    t.stop();
    t.print_elapsed_time("converting audio")

    # Inference
    t.start()
    pred_token_arr, _ = model.inference_file(bsz=8, audio_segments=audio_segments)
    t.stop();
    t.print_elapsed_time("model inference")

    # Post-processing
    t.start()
    num_channels = model.task_manager.num_decoding_channels
    n_items = audio_segments.shape[0]
    start_secs_file = [model.audio_cfg['input_frames'] * i / model.audio_cfg['sample_rate'] for i in range(n_items)]
    pred_notes_in_file = []
    n_err_cnt = Counter()

    for ch in range(num_channels):
        pred_token_arr_ch = [arr[:, ch, :] for arr in pred_token_arr]
        zipped_note_events_and_tie, list_events, ne_err_cnt = model.task_manager.detokenize_list_batches(
            pred_token_arr_ch, start_secs_file, return_events=True)
        pred_notes_ch, n_err_cnt_ch = merge_zipped_note_events_and_ties_to_notes(zipped_note_events_and_tie)
        pred_notes_in_file.append(pred_notes_ch)
        n_err_cnt += n_err_cnt_ch

    pred_notes = mix_notes(pred_notes_in_file)

    return pred_notes

# also modified for this script
def extract_GT(annotation_filename):
    annotation_directory = '../../data/guitarset_yourmt3_16k/annotation/'
    annotation_filepath = os.path.join(annotation_directory, annotation_filename)

    if os.path.exists(annotation_filepath):
        print(f"Extracting GT from {annotation_filepath}")
        # Hier könnte deine eigentliche Logik stehen
    else:
        print(f"Annotation file not found: {annotation_filepath}")

    assert os.path.exists(annotation_filepath)
    # load annotation
    notes = np.load(annotation_filepath, allow_pickle=True)

    #data = GT_array.array()  # `.item()` gibt das einzelne Objekt im Array zurück
    # Zugriff auf 'notes'
    #notes = data['notes']
    return notes


def estimate_frequency_from_phase(prev_phase, current_phase, k, W, H, sr):
    if k < 0 or k >= W // 2:
        raise ValueError("Bin index k is out of bounds.")

    omega_k = 2 * np.pi * k / W  # Digitale Frequenz bei Bin k
    # Berechne die Phasendifferenz und verhindere Phasenwrapping
    epsilon = 1e-10
    delta_phi = omega_k * H + np.mod(current_phase - prev_phase - omega_k * H + np.pi + epsilon, 2 * np.pi) - np.pi
    # Frequenzberechnung
    est_freq = delta_phi / (2 * np.pi * H) * sr
    return est_freq


def estimate_frequency_and_amplitude(prev_phase, current_phase, k, W, H, sr, frame, window):
    if k < 0 or k >= W // 2:
        raise ValueError("Bin index k is out of bounds.")

    omega_k = 2 * np.pi * k / W  # Digitale Frequenz bei Bin k

    # Berechne die Phasendifferenz und verhindere Phasenwrapping
    epsilon = 1e-10
    delta_phi = omega_k * H + np.mod(current_phase - prev_phase - omega_k * H + np.pi + epsilon, 2 * np.pi) - np.pi

    # Frequenzberechnung
    est_freq = delta_phi / (2 * np.pi * H) * sr

    # Amplitudenschätzung über DFT-Koeffizient
    n = np.arange(W)
    dft_coeff = np.dot(frame * window, np.exp(1j * 2 * np.pi * est_freq / sr * n))
    est_amplitude = 2 * np.abs(dft_coeff) / np.sum(window)

    return est_freq, est_amplitude



def calculate_partials(fundamental, k_fundamental, prev_NOTESIG, NOTESIG, fft_size, H, sr, beta_max,
                       n_partials, dbg, frame, window, peak_indices):
    partials = {0: fundamental}

    for i in range(1, n_partials):
        predicted_partial = (i + 1) * fundamental
        maxBeta_partial = predicted_partial * math.sqrt(1 + beta_max * (i + 1) ** 2)

        k = int((i + 1) * k_fundamental)
        if k >= fft_size // 2:  # Sicherstellen, dass k im gültigen Bereich bleibt
            break

        best_partial = None
        best_amplitude = -float('inf')
        best_offset = None

        # 🔍 **Peak innerhalb des Frequenzbereichs suchen**
        peak_candidates = [p for p in peak_indices if predicted_partial <= p * sr / fft_size <= maxBeta_partial]

        if peak_candidates:
            # Wähle den Peak mit der höchsten Amplitude als Basis
            peak_bin = max(peak_candidates, key=lambda p: np.abs(NOTESIG[p]))

            # 🔍 **Untersuche nur die Umgebung (-2 bis +8 Bins)**
            for offset in range(-2, 9):
                k_test = peak_bin + offset
                if k_test < 0 or k_test >= fft_size // 2:
                    continue  # Überspringe ungültige Werte

                partial, partial_amplitude = estimate_frequency_and_amplitude(
                    np.angle(prev_NOTESIG[k_test]), np.angle(NOTESIG[k_test]), k_test, fft_size, H, sr, frame, window)

                partial_AMP = 20 * np.log10(np.abs(partial_amplitude))

                if partial_AMP > -70 and partial_AMP > best_amplitude:
                    best_partial = partial
                    best_amplitude = partial_AMP
                    best_offset = k_test - peak_bin  # Offset relativ zum Peak-Bin

        # Falls ein valider Partial gefunden wurde, speichere ihn
        if best_partial is not None:
            partials[i] = best_partial
            if dbg:
                print(f"Bin-Offset {best_offset} used for partial {i} with amplitude {best_amplitude} dB")

    if dbg:
        print(f"Length Partials: {len(partials)}")

    return partials


def calculate_fundamental (k_fundamental_guess, prev_NOTESIG, NOTESIG, fft_size, H, sr, frame, window):

    best_partial = None
    best_amplitude = -float('inf')

    for offset in range(-2, 2):  # Durchsuche k-3 bis k+20
        k_test = k_fundamental_guess + offset
        if k_test < 0 or k_test >= fft_size // 2:
            continue  # Überspringe ungültige Werte

        partial, partial_amplitude = estimate_frequency_and_amplitude(
            np.angle(prev_NOTESIG[k_test]), np.angle(NOTESIG[k_test]), k_test, fft_size, H, sr, frame, window)

        partial_AMP = 20 * np.log10(np.abs(partial_amplitude))

        # Filter nach Frequenzbereich und Mindestamplitude
        if partial_AMP > best_amplitude:
            best_partial = partial
            best_amplitude = partial_AMP

    fundamental = best_partial
    return fundamental


# calculates beta distribution for given signal and correctly identified notes in signal
def estBeta(stringSig, sr, notes, dbg):
    beta_max = 10 * 1.4e-4



    for counter, note in enumerate(notes):
        try:
            onsetSample = int(note.onset * sr + 0.05 * sr)
            offsetSample = int(note.offset * sr)
            noteFreq = noteToFreq(note.pitch)
            noteSig = stringSig[onsetSample:offsetSample]
        except IndexError:
            continue

        W = 4096
        (H, fft_size) = W // 16, W
        if len(noteSig) <= W:
            continue

        buffered_signal = np.lib.stride_tricks.sliding_window_view(noteSig, window_shape=W)[::H]
        if buffered_signal.ndim < 2:
            continue

        padded_signal = np.pad(buffered_signal, ((0, 0), (0, fft_size - W)), mode='constant')
        window = scipy.signal.windows.hann(fft_size, sym=False)
        window_sum = np.sum(window)  # Sum of the window coefficients
        buffered_windowed_signal = padded_signal * window
        freqs = rfftfreq(fft_size, d=1 / sr)

        if len(buffered_windowed_signal) < 2:
            print("buffered signal not minimum of 2 frames")
            continue

        W = fft_size

        prev_NOTESIG = rfft(buffered_windowed_signal[0])

        noteBetas = []

        # Normalize
        buffered_windowed_signal /= np.max(np.abs(buffered_windowed_signal))

        for frame_idx, frame in enumerate(buffered_windowed_signal[1:], start=1):
            NOTESIG = rfft(frame)
            magnitude_spectrum = np.abs(NOTESIG)

            # Normalize the magnitude spectrum by 2/window_sum
            magnitude_spectrum_normalized = (2 * magnitude_spectrum) / window_sum

            magnitude_spectrum = 20 * np.log10(magnitude_spectrum_normalized + 1e-10)

            Threshold = -70  # -40 dB
            peak_indices, _ = find_peaks(magnitude_spectrum, height=Threshold)
            peak_freqs = freqs[peak_indices]

            if len(peak_indices) == 0:
                print("no peaks detected")
                continue

            # Estimate the fundamental frequency using phase
            k_noteFreq = int(noteFreq * W / sr)
            fundamental = calculate_fundamental(k_noteFreq, prev_NOTESIG, NOTESIG, fft_size, H, sr, frame,
                                                window)

            if not (noteFreq / math.pow(2, 1 / 24) <= fundamental <= noteFreq * math.pow(2, 1 / 24)):
                print("Fundamental not in quartertone range of noteFreq for Frame")
                continue

            # Calculate partials
            n_partials = min(len(peak_indices), 25)
            partials = calculate_partials(fundamental, int(fundamental * W / sr), prev_NOTESIG, NOTESIG, fft_size, H, sr, beta_max, n_partials, dbg, frame, window, peak_indices)

            # Calculate Beta for each partial
            partialBetas = [
                ((freq / ((order + 1) * partials[0])) ** 2 - 1) / ((order + 1) ** 2)
                for order, freq in partials.items() if order > 0
            ]

            # Filter outliers
            valid_partialBetas = [pb for pb in partialBetas if 0 < pb < beta_max]
            if valid_partialBetas:
                l = np.array(valid_partialBetas)
                Q1 = np.quantile(l, 0.1)
                Q3 = np.quantile(l, 0.9)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                cleaned_partialBetas = l[(l >= lower_bound) & (l <= upper_bound)].tolist()

                if cleaned_partialBetas:
                    noteBetas.append(cleaned_partialBetas)
                else:
                    print("No values left after IQR-filter")
            else:
                print("No valid partial Betas")
                # noteBetas.append(valid_partialBetas)

            # Phase Estimaton Logic
            prev_NOTESIG = NOTESIG



            # plot only if toggle
            if dbg:
                plt.figure(figsize=(12, 6))
                # Frequenzspektrum
                plt.plot(freqs, magnitude_spectrum, label=f"Frame {frame_idx}")
                # Threshold-Linie
                plt.axhline(Threshold, color='black', linestyle='--', label=f'Threshold: {Threshold:.2f} dB')
                # Modellierte Grundfrequenz (noteFreq)
                plt.scatter(noteFreq, 0, color="purple", marker="x", s=100, label="Model NoteFreq")

                # # Gefundene Peaks
                plt.scatter(freqs[peak_indices], magnitude_spectrum[peak_indices], color='orange', marker="o", label="Detected Peaks")

                # Vorhergesagte Partialfrequenzen (basierend auf fundamental)
                predicted_partials = [fundamental * (i + 1) for i in range(1, n_partials)]
                plt.scatter(predicted_partials, [0] * len(predicted_partials),
                            color='red', marker="s", label="Predicted Partials")
                # Tatsächlich gefundene Partialfrequenzen aus dem Dict partials
                real_partials = list(partials.values())
                plt.scatter(real_partials, [-2] * len(real_partials),
                            color='green', marker="d", label="Real Partials")
                # Achsen und Darstellung
                plt.title(f"Frequency Spectrum (Frame {frame_idx})")
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Magnitude (dB)")
                plt.xscale("log")
                plt.grid()
                plt.legend()
                plt.show()


        flat_noteBetas = flatten_recursive(noteBetas)
        print(f"noteBetas: {flat_noteBetas}")


        # create new instance of stringNote with noteBetas
        updated_note = dataclasses.replace(note, noteBetas=flat_noteBetas)
        notes[counter] = updated_note  # Update the notes list in place

    # return notes (stringNotes) with addded note betas
    return notes



def flatten_recursive(nested_list):
    """Recursively flatten a nested list or array."""
    flat_list = []
    if isinstance(nested_list, (list, np.ndarray)):  # Check if it's an iterable
        for item in nested_list:
            if isinstance(item, (list, np.ndarray)):  # If the item is iterable, flatten it recursively
                flat_list.extend(flatten_recursive(item))
            else:
                flat_list.append(item)  # If not iterable, add the item directly
    else:
        flat_list.append(nested_list)  # If the input is not iterable, add it directly

    return flat_list


def process_file(sig, sr, dbg, model, audio_filepath, delta, betas_kde_dict, betas_kde_xvals, GT_matchedNotes, betas):

    # Call model output and match notes, fill String GT
    pred_notes = process_audioFile_notes(model, sig, sr)
    stringNotes = fill_stringNotes(pred_notes, GT_matchedNotes, delta)

    # Schätze Beta-Verteilung für jede Note
    stringNotes = estBeta(sig, sr, stringNotes, dbg)


    # return filled string Notes
    return stringNotes


def plot_frequency_distribution(stringNotes):
    """
    Erzeugt Verteilungstabellen für den Zusammenhang zwischen noteFreq und stringGT und stellt diese als Histogramm dar.

    Args:
        stringNotes (list): Eine Liste von stringNote-Objekten, die die Noteninformationen enthalten.
    """
    # Definiere Frequenzen für jede Saite (E2, A2, D3, G3, B3, E4)
    string_frequencies = {
        1: 82.41,  # Saite 1 (E2)
        2: 110.00,  # Saite 2 (A2)
        3: 146.83,  # Saite 3 (D3)
        4: 196.00,  # Saite 4 (G3)
        5: 246.94,  # Saite 5 (B3)
        6: 329.63,  # Saite 6 (E4)
    }

    # Bereite ein Dictionary vor, um die Frequenzen pro Saite zu speichern
    frequencies_by_string = {i: [] for i in range(6)}

    # Füge Frequenzen in das Dictionary ein
    for note in stringNotes:
        string_gt = note.stringGT  # Erhalte die Ground Truth der Saite
        if string_gt is not None:
            note_freq = noteToFreq(note.pitch)  # Konvertiere die MIDI-Note zu Frequenz
            frequencies_by_string[string_gt].append(note_freq)

    # Erstelle Histogramme für jede Saite
    plt.figure(figsize=(10, 6))

    for string, freqs in frequencies_by_string.items():
        if freqs:  # Nur Saite plotten, wenn es Frequenzen gibt
            plt.hist(freqs, bins=20, alpha=0.5, label=f'Saite {string}', density=True)

    # Plot-Einstellungen
    plt.title("Frequenzverteilung pro Saite")
    plt.xlabel("Frequenz (Hz)")
    plt.ylabel("Relative Häufigkeit")
    plt.legend()
    plt.grid(True)
    plt.show()













""" Function with plotting, just wasserstein"""
# def guess_string_from_betas(stringNotes, betas_roh, use_kde=True, plot=True):
#     """
#     Predicts the most likely string for each note using Wasserstein distance
#     between note KDE and normal distributions.
#
#     Args:
#         stringNotes (list): List of note objects containing noteBetas.
#         betas_roh (dict): Dictionary of raw beta values per string.
#         use_kde (bool): Whether to use KDE for noteBetas.
#         plot (bool): Whether to plot distributions for debugging.
#
#     Returns:
#         list: Updated list of note objects with string predictions.
#     """
#     updated_stringNotes = []
#     string_keys = [f"Saite_{i + 1}" for i in range(6)]
#     string_labels = [f"String_{i}" for i in range(6)]  # Anpassung der Labels für Plots
#
#     # Berechne Mittelwert und Standardabweichung für jede Saite
#     betas_mean_std = {
#         key: (np.mean(vals), max(1e-6, np.std(vals))) if len(vals) > 1 else (vals[0], 1e-6)
#         for key, vals in betas_roh.items()
#     }
#
#     # Farben für die Saiten
#     colors = {
#         "Saite_1": "skyblue",
#         "Saite_2": "red",
#         "Saite_3": "green",
#         "Saite_4": "orange",
#         "Saite_5": "purple",
#         "Saite_6": "brown",
#     }
#
#
#     # Plot setup for string distributions
#     if plot:
#         plt.figure(figsize=(12, 8))
#         x_vals = np.linspace(0, 0.001, 200)
#
#         # Plot distributions for each string
#         for i, string_key in enumerate(string_keys):
#             mean, std = betas_mean_std[string_key]
#             string_pdf = norm.pdf(x_vals, mean, std)
#             plt.plot(x_vals, string_pdf, label=f"{string_labels[i]} (Normal)", color=colors[string_key])
#
#         plt.xlabel("Beta Value")
#         plt.ylabel("Probability Density")
#         plt.title("String PDFs")
#         plt.legend()
#         plt.grid()
#         plt.show()
#
#     for note in stringNotes:
#         best_string = None
#         likelihood_ratio = None
#         wasserstein_scores = []
#         freq_weights = []
#
#         flat_noteBetas = note.noteBetas
#         noteFreq = noteToFreq(note.pitch)
#
#         if flat_noteBetas is not None and len(flat_noteBetas) > 1:
#             # KDE für noteBetas berechnen
#             x_vals = np.linspace(0, 0.001, 200)
#             note_kde = gaussian_kde(flat_noteBetas, bw_method="silverman")
#             note_kde_values = note_kde(x_vals)
#
#             for string_idx, string_key in enumerate(string_keys):
#                 mean, std = betas_mean_std[string_key]
#
#                 # Erstelle Normalverteilungs-PDF für die Saite
#                 string_pdf = norm.pdf(x_vals, mean, std)
#
#                 # Berechne Wasserstein-Distanz zwischen KDE und der Normalverteilung
#                 distance = wasserstein_distance(note_kde_values, string_pdf)
#
#                 wasserstein_scores.append(distance)
#
#
#
#             if wasserstein_scores:
#
#                 best_string_idx = np.argmin(wasserstein_scores)  # Kleinste Distanz ist beste Übereinstimmung
#                 best_string = best_string_idx
#
#                 if len(wasserstein_scores) > 1:
#                     sorted_scores = np.sort(wasserstein_scores)
#                     likelihood_ratio = sorted_scores[1] - sorted_scores[0]  # Differenz der besten zwei
#
#                 if plot:
#                     plt.figure(figsize=(8, 5))
#                     matched_string_key = string_keys[best_string_idx]
#                     mean, std = betas_mean_std[matched_string_key]
#                     matched_string_pdf = norm.pdf(x_vals, mean, std)
#
#                     # Plot noteBeta KDE
#                     plt.plot(x_vals, note_kde_values, label=f"Note {note.pitch} KDE", color="black", linewidth=2)
#
#                     # Plot matched string distribution mit aktualisiertem Label
#                     plt.plot(
#                         x_vals,
#                         matched_string_pdf,
#                         label=f"{string_labels[best_string_idx]} (Normal)",
#                         color=colors[matched_string_key],
#                         linestyle="--"
#                     )
#
#                     # Highlight the maximum of the noteBeta KDE
#                     max_x = x_vals[np.argmax(note_kde_values)]  # x-Wert des Maximums
#                     max_y = np.max(note_kde_values)  # y-Wert des Maximums
#                     plt.scatter(max_x, max_y, color="red", zorder=5, label="KDE Max")
#
#                     plt.xlabel("Beta Value")
#                     plt.ylabel("Probability Density")
#                     plt.title(f"Note {note.pitch} KDE vs Matched String ({string_labels[best_string_idx]})")
#                     plt.legend()
#                     plt.grid()
#                     plt.show()
#
#         updated_note = stringNote(
#             is_drum=note.is_drum,
#             program=note.program,
#             onset=note.onset,
#             offset=note.offset,
#             pitch=note.pitch,
#             velocity=note.velocity,
#             noteBetas=note.noteBetas,
#             string_pred=best_string,  # Beste Saite basierend auf Wasserstein-Distanz
#             likelihood_ratio=likelihood_ratio,  # Sicherheit der Vorhersage
#             stringGT=note.stringGT
#         )
#
#         updated_stringNotes.append(updated_note)
#
#     return updated_stringNotes




"""theoretische Modell"""
# def freq_weight_function(noteFreq, string_freq):
#     if noteFreq <= (string_freq / math.pow(2, 1 / 24)):
#         return 0
#     elif noteFreq <= (4/3) * string_freq:
#         return 1.0
#     elif noteFreq <= (34/9) * string_freq:
#         # Konstanter linearer Abfall ab der Quarte bis 34/9 * string_freq
#         max_drop = 1  # Maximaler Abfall von 1.0 auf 0.0
#         slope = -max_drop / (((34/9) * string_freq) - ((4/3) * string_freq))
#         return 1.0 + slope * (noteFreq - (4/3) * string_freq)
#     else:
#         # Konstanter niedriger Wert über 34/9 * string_freq
#         return 0

"""Custom: semi-theoretisch, bisher bestes Ergebnis"""
def freq_weight_function(noteFreq, string_freq):
    if noteFreq <= (string_freq / math.pow(2, 1 / 24)):  # If note frequency is below or equal to the string frequency
        return 0  # Very likely
    elif noteFreq <= (5/3) * string_freq:  # If frequency is within the fifth (3:2 ratio)
        return 1.0  # Still very likely
    elif noteFreq <= 2 * string_freq:  # If frequency is between the fifth and double the string frequency
        # Exponential decay function for frequencies between the fifth and double the string frequency
        return np.exp(-((noteFreq - (5/3) * string_freq) ** 2) / (2 * (0.2 * string_freq) ** 2))
    else:  # Frequency above double the string frequency
        return 1e-6  # Very unlikely



"""Funktion combined, Wasserstein - best results"""
def guess_string_from_betas(stringNotes, betas_roh, use_kde=True, plot=False):
    updated_stringNotes = []
    string_keys = [f"Saite_{i + 1}" for i in range(6)]
    string_labels = [f"String_{i}" for i in range(6)]  # Anpassung der Labels für Plots

    betas_mean_std = {
        key: (np.mean(vals), max(1e-6, np.std(vals))) if len(vals) > 1 else (vals[0], 1e-6)
        for key, vals in betas_roh.items()
    }

    string_frequencies = {
        "Saite_1": 82.41,  # E2
        "Saite_2": 110.00,  # A2
        "Saite_3": 146.83,  # D3
        "Saite_4": 196.00,  # G3
        "Saite_5": 246.94,  # B3
        "Saite_6": 329.63,  # E4
    }

    #Farben für die Saiten
    colors = {
        "Saite_1": "skyblue",
        "Saite_2": "red",
        "Saite_3": "green",
        "Saite_4": "orange",
        "Saite_5": "purple",
        "Saite_6": "brown",
    }


    # Plot setup for string distributions
    if plot:
        plt.figure(figsize=(12, 8))
        x_vals = np.linspace(0, 0.001, 200)

        # Plot distributions for each string
        for i, string_key in enumerate(string_keys):
            mean, std = betas_mean_std[string_key]
            string_pdf = norm.pdf(x_vals, mean, std)
            plt.plot(x_vals, string_pdf, label=f"{string_labels[i]} (Normal)", color=colors[string_key])

        plt.xlabel("Beta Value")
        plt.ylabel("Probability Density")
        plt.title("String PDFs")
        plt.legend()
        plt.grid()
        plt.show()

    for note in stringNotes:
        best_string = None
        wasserstein_scores = []
        freq_weights = []

        flat_noteBetas = note.noteBetas
        noteFreq = noteToFreq(note.pitch)

        if flat_noteBetas is not None and len(flat_noteBetas) > 1:
            x_vals = np.linspace(0, 0.001, 1000)
            note_kde = gaussian_kde(flat_noteBetas, bw_method="silverman")
            note_kde_values = note_kde(x_vals)
            # note_kde_values /= max(note_kde_values)

            for string_idx, string_key in enumerate(string_keys):
                mean, std = betas_mean_std[string_key]
                string_pdf = norm.pdf(x_vals, mean, std)



                distance = wasserstein_distance(note_kde_values, string_pdf)
                wasserstein_scores.append(distance)

                string_freq = string_frequencies[string_key]
                freq_weight = freq_weight_function(noteFreq, string_freq)
                freq_weights.append(freq_weight)

            wasserstein_scores = np.array(wasserstein_scores)
            freq_weights = np.array(freq_weights)

            if wasserstein_scores.max() > wasserstein_scores.min():
                wasserstein_scores_norm = (wasserstein_scores - wasserstein_scores.min()) / (
                        wasserstein_scores.max() - wasserstein_scores.min())
            else:
                wasserstein_scores_norm = np.zeros_like(wasserstein_scores)

            if freq_weights.max() > freq_weights.min():
                freq_weights_norm = (freq_weights - freq_weights.min()) / (
                        freq_weights.max() - freq_weights.min())
            else:
                freq_weights_norm = np.zeros_like(freq_weights)

            wasserstein_weight = 1
            freq_weight_weight = 1 #* 5 # - wasserstein_weight

            combined_scores = wasserstein_weight * (-wasserstein_scores_norm) + freq_weight_weight * freq_weights_norm
            best_string_idx = np.argmax(combined_scores)
            best_string = best_string_idx

            if plot:
                plt.figure(figsize=(8, 5))
                matched_string_key = string_keys[best_string_idx]
                mean, std = betas_mean_std[matched_string_key]
                matched_string_pdf = norm.pdf(x_vals, mean, std)

                # Plot noteBeta KDE
                plt.plot(x_vals, note_kde_values, label=f"Note {note.pitch} KDE", color="black", linewidth=2)

                # Plot matched string distribution mit aktualisiertem Label
                plt.plot(
                    x_vals,
                    matched_string_pdf,
                    label=f"{string_labels[best_string_idx]} (Normal)",
                    color=colors[matched_string_key],
                    linestyle="--"
                )

                # Highlight the maximum of the noteBeta KDE
                max_x = x_vals[np.argmax(note_kde_values)]  # x-Wert des Maximums
                max_y = np.max(note_kde_values)  # y-Wert des Maximums
                plt.scatter(max_x, max_y, color="red", zorder=5, label="KDE Max")

                plt.xlabel("Beta Value")
                plt.ylabel("Probability Density")
                plt.title(f"Note {note.pitch} KDE vs Matched String ({string_labels[best_string_idx]})")
                plt.legend()
                plt.grid()
                plt.show()

        updated_note = stringNote(
            is_drum=note.is_drum,
            program=note.program,
            onset=note.onset,
            offset=note.offset,
            pitch=note.pitch,
            velocity=note.velocity,
            noteBetas=note.noteBetas,
            string_pred=best_string,
            likelihood_ratio=None,
            stringGT=note.stringGT
        )
        updated_stringNotes.append(updated_note)

    return updated_stringNotes







def fill_stringNotes(pred_notes, GT_matchedNotes, delta):
    """
    Match predicted notes with ground truth notes and initialize `stringNote` objects.

    Args:
        pred_notes (list): List of predicted notes.
        GT_matchedNotes (list): List of ground truth notes.
        delta (float): Time tolerance for matching notes.

    Returns:
        list: List of `stringNote` objects.
    """
    stringNotes = []

    for pred_note in pred_notes:
        for ref_note in GT_matchedNotes:
            if (
                    abs(ref_note.onset - pred_note.onset) <= delta and
                    ref_note.pitch == pred_note.pitch and
                    not pred_note.is_drum and
                    pred_note.program == 24
            ):
                stringNotes.append(stringNote(
                    is_drum=ref_note.is_drum,
                    program=ref_note.program,
                    onset=ref_note.onset,
                    offset=ref_note.offset,
                    pitch=ref_note.pitch,
                    velocity=ref_note.velocity,
                    noteBetas=None,  # Will be filled later
                    string_pred=None,  # Will be filled later
                    likelihood_ratio=None,  # Will be filled later
                    stringGT=ref_note.string_index
                ))

    return stringNotes


def parainterp(idx, data):
    # Überprüfen, ob der Index für die Interpolation gültig ist
    if idx <= 0 or idx >= len(data) - 1:
        # Kein gültiger Bereich für Interpolation
        delta = 0.0
        newMax = data[idx]
        return idx, delta, newMax

    xy = data[idx - 1:idx + 2]

    # mittlerer Idx muss das maximum sein und indices: 0, 1, 2
    p = np.argmax(xy)  # das maximum muss doch laut definition der findpeaks der mittlere Index sein ...

    # Berechnung der parabolischen Verschiebung
    delta = 0.5 * (xy[p - 1] - xy[p + 1]) / (xy[p - 1] - 2 * xy[p] + xy[p + 1])
    newMax = xy[p] - 0.25 * (xy[p - 1] - xy[p + 1]) * delta

    return p, delta, newMax


def noteToFreq(note):
    a = 440  # frequency of A (common value is 440Hz)
    return (a / 32) * (2 ** ((note - 9) / 12))


#IMPORTANT: Different from function in modelBetafinder
def get_annotation_filename(audio_filename):
    # Extrahiere die Base, behalte `solo` oder `comp`, entferne `pshift`
    base_name = audio_filename.split(".")[0]

    if "_pshift" in base_name:
        base_name = base_name.split("_pshift")[0]

    # Entferne die Audio-spezifischen Endungen (wie "_cln" oder "_mic")
    for suffix in ["_hex", "_cln", "_mic", "_debleeded"]:
        base_name = base_name.replace(suffix, "")

    # Füge das notwendige Format für die Annotation an
    return f"{base_name}_matchedNotes.npy"


def load_betas_from_json(filename):
    """Lädt die Beta-Werte aus einer JSON-Datei."""
    with open(filename, 'r') as f:
        return json.load(f)


def analyze_string_predictions(stringNotes: List[stringNote]):
    accuracy = 0
    conf_matrix = None
    string_counts = None
    mean_likelihood = 0
    per_string_accuracy = None

    # Initialize necessary data structures
    total_notes = 0
    correct_predictions = 0
    string_predictions = []
    string_ground_truths = []
    likelihood_ratios = []
    conf_matrix_counts = np.zeros((6, 6), dtype=int)

    for note in stringNotes:
        if note.string_pred is not None and note.stringGT is not None:
            gt = int(note.stringGT)
            pred = int(note.string_pred)

            string_ground_truths.append(gt)
            string_predictions.append(pred)
            conf_matrix_counts[gt, pred] += 1
            total_notes += 1

            if gt == pred:
                correct_predictions += 1

            if note.likelihood_ratio is not None:
                likelihood_ratios.append(note.likelihood_ratio)

    if total_notes == 0:
        print("Keine Noten mit sowohl string_pred als auch stringGT gefunden.")
        return {
            "accuracy": 0,
            "confusion_matrix": None,
            "string_counts": None,
            "mean_likelihood": 0,
            "per_string_accuracy": None
        }

    # Berechne die Gesamtgenauigkeit
    accuracy = (correct_predictions / total_notes) * 100
    print(f"Overall Accuracy (String Prediction): {accuracy:.2f}%")

    # Confusion Matrix
    conf_matrix = conf_matrix_counts
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(6), yticklabels=np.arange(6))
    plt.xlabel("Predicted String")
    plt.ylabel("True String")
    plt.title("Confusion Matrix der Saitenvorhersagen")
    plt.show()

    # Häufigkeit der vorhergesagten Saiten
    unique, counts = np.unique(string_predictions, return_counts=True)
    string_counts = dict(zip(unique, counts))
    print("Häufigkeit der vorhergesagten Saiten:", string_counts)

    # Durchschnittliche Likelihood Ratio berechnen
    mean_likelihood = np.mean(likelihood_ratios) if likelihood_ratios else 0
    print(f"Mittlere Likelihood Ratio: {mean_likelihood:.4f}")

    # Trefferquote pro Saite
    correct_counts = {i: conf_matrix[i, i] for i in range(6)}
    total_counts = {i: np.sum(conf_matrix[i, :]) for i in range(6)}
    per_string_accuracy = {i: (correct_counts[i] / total_counts[i]) * 100 if total_counts[i] > 0 else 0 for i in range(6)}

    print("Trefferquote pro Saite (%):")
    for string, acc in per_string_accuracy.items():
        print(f"Saite {string}: {acc:.2f}%")

    return {
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix,
        "string_counts": string_counts,
        "mean_likelihood": mean_likelihood,
        "per_string_accuracy": per_string_accuracy
    }





# Main
def main():
    model_name = "YPTF+Single (noPS)"
    print(f"Running evaluation for model: {model_name}")

    # model_name = "YMT3+" # @param [YMT3+", "YPTF+Single (noPS)", "YPTF+Multi (PS)", "YPTF.MoE+Multi (noPS)", "YPTF.MoE+Multi (PS)"]

    precision = 'bf16-mixed'  # @param ["32", "bf16-mixed", "16"]
    project = '2024'

    if model_name == "YMT3+":
        checkpoint = "notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72@model.ckpt"
        args = [checkpoint, '-p', project, '-pr', precision]
    elif model_name == "YPTF+Single (noPS)":
        checkpoint = "ptf_all_cross_rebal5_mirst_xk2_edr005_attend_c_full_plus_b100@model.ckpt"
        args = [checkpoint, '-p', project, '-enc', 'perceiver-tf', '-ac', 'spec',
                '-hop', '300', '-atc', '1', '-pr', precision]
    elif model_name == "YPTF+Multi (PS)":
        checkpoint = "mc13_256_all_cross_v6_xk5_amp0811_edr005_attend_c_full_plus_2psn_nl26_sb_b26r_800k@model.ckpt"
        args = [checkpoint, '-p', project, '-tk', 'mc13_full_plus_256',
                '-dec', 'multi-t5', '-nl', '26', '-enc', 'perceiver-tf',
                '-ac', 'spec', '-hop', '300', '-atc', '1', '-pr', precision]
    elif model_name == "YPTF.MoE+Multi (noPS)":
        checkpoint = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt"
        args = [checkpoint, '-p', project, '-tk', 'mc13_full_plus_256', '-dec', 'multi-t5',
                '-nl', '26', '-enc', 'perceiver-tf', '-sqr', '1', '-ff', 'moe',
                '-wf', '4', '-nmoe', '8', '-kmoe', '2', '-act', 'silu', '-epe', 'rope',
                '-rp', '1', '-ac', 'spec', '-hop', '300', '-atc', '1', '-pr', precision]
    elif model_name == "YPTF.MoE+Multi (PS)":
        checkpoint = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2@model.ckpt"
        args = [checkpoint, '-p', project, '-tk', 'mc13_full_plus_256', '-dec', 'multi-t5',
                '-nl', '26', '-enc', 'perceiver-tf', '-sqr', '1', '-ff', 'moe',
                '-wf', '4', '-nmoe', '8', '-kmoe', '2', '-act', 'silu', '-epe', 'rope',
                '-rp', '1', '-ac', 'spec', '-hop', '300', '-atc', '1', '-pr', precision]
    else:
        raise ValueError(model_name)

    # Extension
    model = load_model_checkpoint(args=args)

    filename = '../content/Betas/betas.json'
    betas = load_betas_from_json(filename)

    # calculate Kernel Density Estimation for each beta-curve
    string_keys = [f"Saite_{i + 1}" for i in range(6)]  # ["Saite_1", ..., "Saite_6"]

    betas_kde_dict = {}
    betas_kde_x_vals = np.linspace(0, 0.001, 200)  # Wertebereich für KDE

    # KDE für jede Saite vorbereiten
    for string_key in string_keys:
        beta_values = np.array([w for w in betas.get(string_key, []) if 0 <= w <= 0.001])

        kde = gaussian_kde(beta_values)
        kde_values = kde(betas_kde_x_vals)

        # Fläche unter der KDE berechnen (numerische Integration)
        area = np.trapz(kde_values, betas_kde_x_vals)  # Alternative zu simps
        kde_values /= area  # Auf Fläche = 1 normieren

        betas_kde_dict[string_key] = kde_values

    # Computations
    audio_directory = '../../data/guitarset_yourmt3_16k/audio_mono-mic/'

    i = 0
    delta = 0.05  # 50 ms

    all_stringNotes = []
    filtered_stringNotes = []

    for file in os.listdir(audio_directory):
        # Nur "solo"-Audio-Dateien ohne "pshift"
        # if "solo" in file and "pshift" not in file:
        if "pshift" not in file:
            filename = os.fsdecode(file)

            audio_filepath = os.path.join(audio_directory, filename)
            annotation_filename = get_annotation_filename(filename)

            annotation_directory = '../../data/guitarset_yourmt3_16k/annotation/'
            annotation_filepath = os.path.join(annotation_directory, annotation_filename)

            if os.path.exists(annotation_filepath):

                GT_matchedNotes = extract_GT(annotation_filename)
                # # for debug purposes
                # GT_matchedNotes = extract_GT("00_Funk1-114-Ab_solo_matchedNotes.npy")

                # Load audio
                sig, sr = sf.read(audio_filepath)
                sig = sig / np.max(np.abs(sig))  # Normalize

                # plotting etc.
                dbg = 0


                stringNotes = process_file(sig, sr, dbg, model, audio_filepath, delta, betas_kde_dict, betas_kde_x_vals, GT_matchedNotes, betas)

                annotation_filename = os.path.splitext(annotation_filename)[0]
                annotation_filename = annotation_filename.replace("_notes", "")  # Entfernt ".py", falls vorhanden

                save_dir = "../../data/guitarset_yourmt3_16k/annotation"
                save_path = os.path.join(save_dir, f"{annotation_filename}_stringNotes.npy")
                np.save(save_path, np.array(stringNotes, dtype=object))  # Speichern als NumPy-Array


                all_stringNotes.extend(stringNotes)

                # Progress counter
                i += 1
                print("Analysed File-Number:", i)

            else:
                print(f"Annotation file not found: {annotation_filepath}")

            # if i == 5:
            #     break

    # plot_frequency_distribution(all_stringNotes)

    # stringNotes = guess_string_from_betas(stringNotes, betas_kde_dict, betas_kde_xvals, betas)
    all_stringNotes = guess_string_from_betas(all_stringNotes, betas)

    # filter all string Notes with no noteBetas
    filtered_stringNotes.extend([note for note in all_stringNotes if note.noteBetas is not None])

    # analyze guitar string estimates
    analyze_string_predictions(filtered_stringNotes)


    # save all string Notes
    save_dir = "../../data/guitarset_yourmt3_16k/annotation"
    save_path = os.path.join(save_dir, "000all_stringNotes.npy")
    np.save(save_path, np.array(all_stringNotes, dtype=object))  # Speichern als NumPy-Array

    # save filtered stringNotes
    save_dir = "../../data/guitarset_yourmt3_16k/annotation"
    save_path = os.path.join(save_dir, "000all_stringNotes_filtered.npy")
    np.save(save_path, np.array(filtered_stringNotes, dtype=object))  # Speichern als NumPy-Array




# %%
if __name__ == "__main__":
    main()