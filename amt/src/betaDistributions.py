import os
import sys
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
from typing import Tuple, Dict, Literal, List
import torchaudio
import sounddevice as sd
import jams
from scipy import interpolate
from dataclasses import dataclass
from utils.note_event_dataclasses import matchNote




from model.init_train import initialize_trainer, update_config
from utils.task_manager import TaskManager
from config.vocabulary import drum_vocab_presets
from utils.utils import str2bool
from utils.utils import Timer
from utils.audio import slice_padded_array
from utils.note2event import mix_notes, note2note_event, sort_notes, validate_notes, trim_overlapping_notes
from utils.event2note import merge_zipped_note_events_and_ties_to_notes
from utils.utils import write_model_output_as_midi, write_err_cnt_as_json
from model.ymt3 import YourMT3






def load_model_checkpoint(args=None):
    parser = argparse.ArgumentParser(description="YourMT3")
    # General
    parser.add_argument('exp_id', type=str, help='A unique identifier for the experiment is used to resume training. The "@" symbol can be used to load a specific checkpoint.')
    parser.add_argument('-p', '--project', type=str, default='ymt3', help='project name')
    parser.add_argument('-ac', '--audio-codec', type=str, default=None, help='audio codec (default=None). {"spec", "melspec"}. If None, default value defined in config.py will be used.')
    parser.add_argument('-hop', '--hop-length', type=int, default=None, help='hop length in frames (default=None). {128, 300} 128 for MT3, 300 for PerceiverTFIf None, default value defined in config.py will be used.')
    parser.add_argument('-nmel', '--n-mels', type=int, default=None, help='number of mel bins (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-if', '--input-frames', type=int, default=None, help='number of audio frames for input segment (default=None). If None, default value defined in config.py will be used.')
    # Model configurations
    parser.add_argument('-sqr', '--sca-use-query-residual', type=str2bool, default=None, help='sca use query residual flag. Default follows config.py')
    parser.add_argument('-enc', '--encoder-type', type=str, default=None, help="Encoder type. 't5' or 'perceiver-tf' or 'conformer'. Default is 't5', following config.py.")
    parser.add_argument('-dec', '--decoder-type', type=str, default=None, help="Decoder type. 't5' or 'multi-t5'. Default is 't5', following config.py.")
    parser.add_argument('-preenc', '--pre-encoder-type', type=str, default='default', help="Pre-encoder type. None or 'conv' or 'default'. By default, t5_enc:None, perceiver_tf_enc:conv, conformer:None")
    parser.add_argument('-predec', '--pre-decoder-type', type=str, default='default', help="Pre-decoder type. {None, 'linear', 'conv1', 'mlp', 'group_linear'} or 'default'. Default is {'t5': None, 'perceiver-tf': 'linear', 'conformer': None}.")
    parser.add_argument('-cout', '--conv-out-channels', type=int, default=None, help='Number of filters for pre-encoder conv layer. Default follows "model_cfg" of config.py.')
    parser.add_argument('-tenc', '--task-cond-encoder', type=str2bool, default=True, help='task conditional encoder (default=True). True or False')
    parser.add_argument('-tdec', '--task-cond-decoder', type=str2bool, default=True, help='task conditional decoder (default=True). True or False')
    parser.add_argument('-df', '--d-feat', type=int, default=None, help='Audio feature will be projected to this dimension for Q,K,V of T5 or K,V of Perceiver (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-pt', '--pretrained', type=str2bool, default=False, help='pretrained T5(default=False). True or False')
    parser.add_argument('-b', '--base-name', type=str, default="google/t5-v1_1-small", help='base model name (default="google/t5-v1_1-small")')
    parser.add_argument('-epe', '--encoder-position-encoding-type', type=str, default='default', help="Positional encoding type of encoder. By default, pre-defined PE for T5 or Perceiver-TF encoder in config.py. For T5: {'sinusoidal', 'trainable'}, conformer: {'rotary', 'trainable'}, Perceiver-TF: {'trainable', 'rope', 'alibi', 'alibit', 'None', '0', 'none', 'tkd', 'td', 'tk', 'kdt'}.")
    parser.add_argument('-dpe', '--decoder-position-encoding-type', type=str, default='default', help="Positional encoding type of decoder. By default, pre-defined PE for T5 in config.py. {'sinusoidal', 'trainable'}.")
    parser.add_argument('-twe', '--tie-word-embedding', type=str2bool, default=None, help='tie word embedding (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-el', '--event-length', type=int, default=None, help='event length (default=None). If None, default value defined in model cfg of config.py will be used.')
    # Perceiver-TF configurations
    parser.add_argument('-dl', '--d-latent', type=int, default=None, help='Latent dimension of Perceiver. On T5, this will be ignored (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-nl', '--num-latents', type=int, default=None, help='Number of latents of Perceiver. On T5, this will be ignored (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-dpm', '--perceiver-tf-d-model', type=int, default=None, help='Perceiver-TF d_model (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-npb', '--num-perceiver-tf-blocks', type=int, default=None, help='Number of blocks of Perceiver-TF. On T5, this will be ignored (default=None). If None, default value defined in config.py.')
    parser.add_argument('-npl', '--num-perceiver-tf-local-transformers-per-block', type=int, default=None, help='Number of local layers per block of Perceiver-TF. On T5, this will be ignored (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-npt', '--num-perceiver-tf-temporal-transformers-per-block', type=int, default=None, help='Number of temporal layers per block of Perceiver-TF. On T5, this will be ignored (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-atc', '--attention-to-channel', type=str2bool, default=None, help='Attention to channel flag of Perceiver-TF. On T5, this will be ignored (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-ln', '--layer-norm-type', type=str, default=None, help='Layer normalization type (default=None). {"layer_norm", "rms_norm"}. If None, default value defined in config.py will be used.')
    parser.add_argument('-ff', '--ff-layer-type', type=str, default=None, help='Feed forward layer type (default=None). {"mlp", "moe", "gmlp"}. If None, default value defined in config.py will be used.')
    parser.add_argument('-wf', '--ff-widening-factor', type=int, default=None, help='Feed forward layer widening factor for MLP/MoE/gMLP (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-nmoe', '--moe-num-experts', type=int, default=None, help='Number of experts for MoE (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-kmoe', '--moe-topk', type=int, default=None, help='Top-k for MoE (default=None). If None, default value defined in config.py will be used.')
    parser.add_argument('-act', '--hidden-act', type=str, default=None, help='Hidden activation function (default=None). {"gelu", "silu", "relu", "tanh"}. If None, default value defined in config.py will be used.')
    parser.add_argument('-rt', '--rotary-type', type=str, default=None, help='Rotary embedding type expressed in three letters. e.g. ppl: "pixel" for SCA and latents, "lang" for temporal transformer. If None, use config.')
    parser.add_argument('-rk', '--rope-apply-to-keys', type=str2bool, default=None, help='Apply rope to keys (default=None). If None, use config.')
    parser.add_argument('-rp', '--rope-partial-pe', type=str2bool, default=None, help='Whether to apply RoPE to partial positions (default=None). If None, use config.')
    # Decoder configurations
    parser.add_argument('-dff', '--decoder-ff-layer-type', type=str, default=None, help='Feed forward layer type of decoder (default=None). {"mlp", "moe", "gmlp"}. If None, default value defined in config.py will be used.')
    parser.add_argument('-dwf', '--decoder-ff-widening-factor', type=int, default=None, help='Feed forward layer widening factor for decoder MLP/MoE/gMLP (default=None). If None, default value defined in config.py will be used.')
    # Task and Evaluation configurations
    parser.add_argument('-tk', '--task', type=str, default='mt3_full_plus', help='tokenizer type (default=mt3_full_plus). See config/task.py for more options.')
    parser.add_argument('-epv', '--eval-program-vocab', type=str, default=None, help='evaluation vocabulary (default=None). If None, default vocabulary of the data preset will be used.')
    parser.add_argument('-edv', '--eval-drum-vocab', type=str, default=None, help='evaluation vocabulary for drum (default=None). If None, default vocabulary of the data preset will be used.')
    parser.add_argument('-etk', '--eval-subtask-key', type=str, default='default', help='evaluation subtask key (default=default). See config/task.py for more options.')
    parser.add_argument('-t', '--onset-tolerance', type=float, default=0.05, help='onset tolerance (default=0.05).')
    parser.add_argument('-os', '--test-octave-shift', type=str2bool, default=False, help='test optimal octave shift (default=False). True or False')
    parser.add_argument('-w', '--write-model-output', type=str2bool, default=True, help='write model test output to file (default=False). True or False')
    # Trainer configurations
    parser.add_argument('-pr','--precision', type=str, default="bf16-mixed", help='precision (default="bf16-mixed") {32, 16, bf16, bf16-mixed}')
    parser.add_argument('-st', '--strategy', type=str, default='auto', help='strategy (default=auto). auto or deepspeed or ddp')
    parser.add_argument('-n', '--num-nodes', type=int, default=1, help='number of nodes (default=1)')
    parser.add_argument('-g', '--num-gpus', type=str, default='auto', help='number of gpus (default="auto")')
    parser.add_argument('-wb', '--wandb-mode', type=str, default="disabled", help='wandb mode for logging (default=None). "disabled" or "online" or "offline". If None, default value defined in config.py will be used.')
    # Debug
    parser.add_argument('-debug', '--debug-mode', type=str2bool, default=False, help='debug mode (default=False). True or False')
    parser.add_argument('-tps', '--test-pitch-shift', type=int, default=None, help='use pitch shift when testing. debug-purpose only. (default=None). semitone in int.')
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
    audio_segments = torch.from_numpy(audio_segments.astype('float32')).to(device).unsqueeze(1) # (n_seg, 1, seg_sz)
    t.stop(); t.print_elapsed_time("converting audio");

    # Inference
    t.start()
    pred_token_arr, _ = model.inference_file(bsz=8, audio_segments=audio_segments)
    t.stop(); t.print_elapsed_time("model inference");

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
    #output_dir = 'Users/simonbuechner/Documents/Studium/AKT/3.Semester_GRAZ_WS2425/Toningenieur-Projekt/dev/YourMT3_evaluation/amt/content/'
    #print(f"Transcribe working directory: {os.getcwd()}") --> src/
    output_directory = '../content/'

    output_file = write_model_output_as_midi(pred_notes, output_directory,
                              audio_info['track_name'], model.midi_output_inverse_vocab)
    t.stop(); t.print_elapsed_time("post processing");
    #output_file =  os.path.join(output_file, audio_info['track_name']  + '.mid')

    #output_directory = os.path.abspath(midifile)
    #print(f"Resolved output directory: {output_directory}")
    #midifile = os.path.join(midifile, audio_info['track_name'] + '.mid')

    output_file = os.path.abspath(output_file)
    #assert os.path.exists(output_directory)
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
    GT_array = np.load(annotation_filepath, allow_pickle=True)

    data = GT_array.item()  # `.item()` gibt das einzelne Objekt im Array zurück
    # Zugriff auf 'notes'
    notes = data['notes']
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

                if partial_AMP > -40 and partial_AMP > best_amplitude:
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


    # # Falls Fundamental ungenau ist, erneut für den ersten Partial berechnen
    # if 1 in partials:
    #     first_partial = partials[1]
    #     k_first_partial = 2 * k_fundamental  # Bin für den ersten Partial berechnen
    #
    #     for i in range(1, int(n_partials / 2)):
    #         predicted_partial = (i + 1) * first_partial
    #         maxBeta_partial = predicted_partial * math.sqrt(1 + beta_max * (i + 1) ** 2)
    #
    #         k = (i + 1) * k_first_partial
    #         if k >= fft_size // 2:
    #             break
    #
    #         partial, partial_amplitude = estimate_frequency_and_amplitude(
    #             np.angle(prev_NOTESIG[k]), np.angle(NOTESIG[k]), k, fft_size, H, sr, frame, window)
    #
    #         partial_AMP = 20 * np.log10(np.abs(partial_amplitude))
    #
    #         if predicted_partial < partial < maxBeta_partial and partial_AMP > -40 and i not in partials:
    #             partials[i] = partial
    #             print(f"First partial used for Partial {i} calculation")
    #
    # # Falls Fundamental und erster Partial ungenau sind, erneut für den zweiten Partial berechnen
    # if 2 in partials:
    #     second_partial = partials[2]
    #     k_second_partial = 3 * k_fundamental  # Bin für den zweiten Partial berechnen
    #
    #     for i in range(1, int(n_partials / 3)):
    #         predicted_partial = (i + 1) * second_partial
    #         maxBeta_partial = predicted_partial * math.sqrt(1 + beta_max * (i + 1) ** 2)
    #
    #         k = (i + 1) * k_second_partial
    #         if k >= fft_size // 2:
    #             break
    #
    #         partial, partial_amplitude = estimate_frequency_and_amplitude(
    #             np.angle(prev_NOTESIG[k]), np.angle(NOTESIG[k]), k, fft_size, H, sr, frame, window)
    #
    #         partial_AMP = 20 * np.log10(np.abs(partial_amplitude))
    #
    #         if predicted_partial < partial < maxBeta_partial and partial_AMP > -40 and i not in partials:
    #             partials[i] = partial
    #             print(f"Second partial used for Partial {i} calculation")

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


def estBeta(stringSig, sr, notes, dbg):
    beta_max = 10 * 1.4e-4
    stringBetas = []

    for note in notes:
        try:
            onsetSample = int(note.onset * sr + note.onsetDiff * sr + 0.02 * sr)
            offsetSample = int(note.offset * sr)
            noteFreq = noteToFreq(note.pitch)
            noteSig = stringSig[onsetSample:offsetSample]
        except IndexError:
            continue

        W = 4096 * 2
        (H, fft_size) =  W // 8, W
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

        if len(buffered_windowed_signal) <= 2:
            print("buffered signal not minimum of 3 frames")
            continue

        W = fft_size

        # Extract time and frequency values from the note.contour attribute
        contour_times, contour_freqs = zip(*note.contour)  # Unpack the list of tuples
        # Create an interpolation function for the contour frequencies
        contour_interp = interpolate.interp1d(contour_times, contour_freqs, kind='linear', fill_value="extrapolate")

        prev_NOTESIG = rfft(buffered_windowed_signal[0])
        noteBetas = []

        # Lists to store frequencies for plotting
        frame_times = []
        model_freqs = []
        contour_fundamentals = []
        phase_fundamentals = []

        #Normalize
        buffered_windowed_signal /= np.max(np.abs(buffered_windowed_signal))

        for frame_idx, frame in enumerate(buffered_windowed_signal[1:], start=1):
            NOTESIG = rfft(frame)
            magnitude_spectrum = np.abs(NOTESIG)

            # Normalize the magnitude spectrum by 2/window_sum
            magnitude_spectrum_normalized = (2 * magnitude_spectrum) / window_sum

            # Convert to dB
            magnitude_spectrum = 20 * np.log10(magnitude_spectrum_normalized + 1e-10)

            Threshold = -50  # -40 dB
            peak_indices, _ = find_peaks(magnitude_spectrum, height=Threshold)


            if len(peak_indices) == 0:
                print("no peaks detected")
                continue

            if dbg:
                plt.figure(figsize=(12, 6))
                # Frequenzspektrum
                plt.plot(freqs, magnitude_spectrum, label=f"Frame {frame_idx}")
                # Threshold-Linie
                plt.axhline(Threshold, color='black', linestyle='--', label=f'Threshold: {Threshold:.2f} dB')
                # Modellierte Grundfrequenz (noteFreq)
                plt.scatter(noteFreq, 0, color="purple", marker="x", s=100, label="Model NoteFreq")

                # # Gefundene Peaks
                plt.scatter(freqs[peak_indices], magnitude_spectrum[peak_indices], color='orange', marker="o",
                            label="Detected Peaks")

                # Achsen und Darstellung
                plt.title(f"Magnitude spectrum Note: {note.pitch} (Frame {frame_idx})")
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Magnitude (dB)")
                plt.xscale("log")
                plt.grid()
                plt.legend()
                plt.show()











            # Calculate the timestamp for the current frame
            frame_time = note.onset + (frame_idx * H) / sr  # Time in seconds
            frame_times.append(frame_time)

            # Interpolate the fundamental frequency from the contour
            fundamental_contour = contour_interp(frame_time)
            contour_fundamentals.append(fundamental_contour)

            # use contour frequency as input for phase estimation
            model_freqs.append(noteFreq)

            # Estimate the fundamental frequency using phase
            k_contour_fundamental = int(fundamental_contour * W / sr)

            fundamental = calculate_fundamental(k_contour_fundamental, prev_NOTESIG, NOTESIG, fft_size, H, sr, frame, window)
            phase_fundamentals.append(fundamental)

            if not (noteFreq / math.pow(2, 1 / 24) <= fundamental <= noteFreq * math.pow(2, 1 / 24)):
                print("Fundamental not in quartertone range of noteFreq for Frame")
                continue

            # Calculate partials
            n_partials = min(len(peak_indices), 20)
            partials = calculate_partials(fundamental, int(fundamental * W / sr), prev_NOTESIG, NOTESIG, fft_size, H, sr, beta_max, n_partials, dbg, frame, window, peak_indices)

            # Calculate Beta for each partial
            partialBetas = [
                ((freq / ((order + 1) * partials[0])) ** 2 - 1) / ((order + 1) ** 2)
                for order, freq in partials.items() if order > 0
            ]

            # Filtere Werte zwischen 0 und beta_max
            valid_partialBetas = [pb for pb in partialBetas if 0 < pb < beta_max]

            if valid_partialBetas:
                l = np.array(valid_partialBetas)
                Q1 = np.quantile(l, 0.1)
                Q3 = np.quantile(l, 0.9)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                cleaned_partialBetas = l[(l >= lower_bound) & (l <= upper_bound)].tolist()

                if dbg:
                    # Erstelle Histogramm
                    plt.figure(figsize=(8, 5))
                    plt.hist(valid_partialBetas, bins=100, alpha=0.5, label="Originalwerte", color="blue")
                    plt.hist(cleaned_partialBetas, bins=100, alpha=0.7, label="Gefilterte Werte", color="orange")

                    # Markiere IQR-Bereich
                    plt.axvline(Q1, color="green", linestyle="dashed", label="Q10 (10%)")
                    plt.axvline(Q3, color="red", linestyle="dashed", label="Q90 (90%)")

                    plt.xlabel("Beta-Werte")
                    plt.ylabel("Häufigkeit")
                    plt.title("Verteilung der Partialton-Betas vor und nach IQR80-Filterung")
                    plt.legend()
                    plt.grid()
                    plt.show()

                if cleaned_partialBetas:
                    noteBetas.append(cleaned_partialBetas)
                else:
                    print("No values left after IQR-filter")
            else:
                print("No valid partial Betas")
                noteBetas.append(valid_partialBetas)

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
                plt.title(f"Magnitude Spectrum Note:{note.pitch} (Frame {frame_idx})")
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Magnitude (dB)")
                plt.xscale("log")
                plt.grid()
                plt.legend()
                plt.show()

        if noteBetas:
            stringBetas.append(noteBetas)

        # Plot the frequency traces for the current note
        if dbg and frame_times:
            plt.figure(figsize=(12, 6))
            plt.plot(frame_times, model_freqs, label="Model NoteFreq", color="purple", linestyle="--")
            plt.plot(frame_times, contour_fundamentals, label="Ground Truth Fundamental", color="blue")
            plt.plot(frame_times, phase_fundamentals, label="Phase-Estimated Fundamental", color="orange")
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")
            plt.title(f"Frequency Traces for Note {note.pitch}")
            plt.legend()
            plt.grid()
            plt.show()

    return stringBetas


def flatten_recursive(nested_list):
    """ Rekursive Funktion, um eine verschachtelte Liste oder ein Array zu flachen. """
    flat_list = []
    if isinstance(nested_list, (list, np.ndarray)):  # Überprüfen, ob es ein iterierbares Objekt ist
        for item in nested_list:
            # Wenn das Item wiederum iterierbar ist, dann rekursiv flach machen
            if isinstance(item, (list, np.ndarray)):
                flat_list.extend(flatten_recursive(item))  # Rekursiv flach machen
            else:
                # Wenn es kein iterierbares Objekt ist, einfach den Wert hinzufügen
                flat_list.append(item)
    else:
        # Falls das übergebene Objekt keine Liste oder Array ist, direkt hinzufügen
        flat_list.append(nested_list)

    return flat_list


def process_file(sig, sr, GT_notes, betas, dbg, model, audio_filepath, delta, annotation_filename):
    all_matched_notes = []  # Liste für gesammelte matchedNotes


    for string_index in range(6):  # 6 Kanäle
        stringSig = sig[:, string_index]
        if np.max(np.abs(stringSig)) != 0:
            stringSig = stringSig / np.max(np.abs(stringSig))  # Normalisieren

            # # Audio wiedergeben
            # sd.play(stringSig, sr)
            # sd.wait()  # Warten, bis das Audio fertig abgespielt ist

        print(f"Current analyzed string index: {string_index}")

        # Call model output and match notes
        pred_notes = process_audioFile_notes(model, stringSig, sr)
        matchedNotes = matchNotes(delta, pred_notes, GT_notes, string_index)

        if dbg:
            print(f"Number Matched notes: {len(matchedNotes)}")

        all_matched_notes.extend(matchedNotes)


        # Schätze Beta-Verteilung
        stringBetas = estBeta(stringSig, sr, matchedNotes, dbg)

        # Speichere Betas für die aktuelle Saite
        betas[f"Saite_{string_index + 1}"].append(stringBetas)


    # save matched notes in new Annotation file
    #annotation_filename = annotation_filename.replace(".npy", "")  # Entfernt ".py", falls vorhanden
    annotation_filename = os.path.splitext(annotation_filename)[0]
    annotation_filename = annotation_filename.replace("_notes", "")  # Entfernt ".py", falls vorhanden

    save_dir = "../../data/guitarset_yourmt3_16k/annotation"
    save_path = os.path.join(save_dir, f"{annotation_filename}_matchedNotes.npy")
    np.save(save_path, np.array(all_matched_notes, dtype=object))  # Speichern als NumPy-Array


    # Flatten betas structure
    flattened_betas = {}

    for saite, werte_liste in betas.items():
        flattened_betas[saite] = []
        for unterliste in werte_liste:
            # Verwende die rekursive Flatten-Funktion
            flattened_betas[saite].extend(flatten_recursive(unterliste))  # Rekursiv flatten

    return flattened_betas


def parainterp(idx, data):
    # Überprüfen, ob der Index für die Interpolation gültig ist
    if idx <= 0 or idx >= len(data) - 1:
        # Kein gültiger Bereich für Interpolation
        delta = 0.0
        newMax = data[idx]
        return idx, delta, newMax

    xy = data[idx - 1:idx + 2]

    # mittlerer Idx muss das maximum sein und indices: 0, 1, 2
    p = np.argmax(xy) # das maximum muss doch laut definition der findpeaks der mittlere Index sein ...

    # Berechnung der parabolischen Verschiebung
    delta = 0.5 * (xy[p - 1] - xy[p + 1]) / (xy[p - 1] - 2 * xy[p] + xy[p + 1])
    newMax = xy[p] - 0.25 * (xy[p - 1] - xy[p + 1]) * delta

    return p, delta, newMax


def noteToFreq(note):
    a = 440 #frequency of A (common value is 440Hz)
    return (a / 32) * (2 ** ((note - 9) / 12))


def get_annotation_filename(audio_filename):
    # Extrahiere die Base, behalte `solo` oder `comp`, entferne `pshift`
    base_name = audio_filename.split(".")[0]

    if "_pshift" in base_name:
        base_name = base_name.split("_pshift")[0]

    # Entferne die Audio-spezifischen Endungen (wie "_cln" oder "_mic")
    for suffix in ["_hex", "_cln", "_mic", "_debleeded"]:
        base_name = base_name.replace(suffix, "")

    # Füge das notwendige Format für die Annotation an
    return f"{base_name}_notes.npy"



def matchNotes(delta: float, pred_notes: List[matchNote], GT_notes: List[matchNote], string_index: int) -> List[matchNote]:
    matched_notes = []

    for ref_note in GT_notes:
        for pred_note in pred_notes:
            if (
                abs(ref_note.onset - pred_note.onset) <= delta and
                ref_note.pitch == pred_note.pitch and
                not pred_note.is_drum and
                pred_note.program == 24
            ):
                matched_notes.append(matchNote(
                    is_drum=ref_note.is_drum,
                    program=ref_note.program,
                    onset=ref_note.onset,
                    onsetDiff=abs(ref_note.onset - pred_note.onset),
                    offset=ref_note.offset,
                    pitch=ref_note.pitch,
                    velocity=ref_note.velocity,
                    contour=ref_note.contour,
                    string_index=string_index
                ))
                break  # Sobald eine Übereinstimmung gefunden wurde, zum nächsten GT-Note wechseln

    return matched_notes













# Main
def main():
    model_name = "YPTF+Single (noPS)"
    print(f"Running evaluation for model: {model_name}")

    #model_name = "YMT3+" # @param [YMT3+", "YPTF+Single (noPS)", "YPTF+Multi (PS)", "YPTF.MoE+Multi (noPS)", "YPTF.MoE+Multi (PS)"]

    precision = 'bf16-mixed' # @param ["32", "bf16-mixed", "16"]
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

    # Computations
    audio_directory = '../../data/guitarset_yourmt3_16k/audio_hex-pickup_debleeded/'

    # betas dictionary
    betas = {f"Saite_{i + 1}": [] for i in range(6)}
    fileCounter = 0
    delta = 0.05 #50 ms


    for file in os.listdir(audio_directory):
        # only non-pitchshifted files
        if "pshift" not in file:
            filename = os.fsdecode(file)

            audio_filepath = os.path.join(audio_directory, filename)
            annotation_filename = get_annotation_filename(filename)


            GT_notes = extract_GT(annotation_filename)

            # Load audio
            sig, sr = sf.read(audio_filepath)
            sig = sig / np.max(np.abs(sig))  # Normalize

            # plotting etc.
            dbg = 1

            betas = process_file(sig, sr, GT_notes, betas, dbg, model, audio_filepath, delta, annotation_filename)
            print(betas)




            # Progress counter
            fileCounter += 1
            print("Analysed File-Number:", fileCounter)

            # local debug
            # if(fileCounter == 2):
            #     break


    filename = 'betas2.json'
    with open(filename, 'w') as f:
        json.dump(betas, f, indent=4)
        print(f"Betas wurden erfolgreich in '{filename}' gespeichert.")



# %%
if __name__ == "__main__":
    main()