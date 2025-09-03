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
import math
from scipy.fft import rfft
from scipy.fft import rfftfreq
from scipy.signal import find_peaks
from typing import Dict, List
import torchaudio
from scipy import interpolate
from utils.note_event_dataclasses import matchNote
from typing import Union

from model.init_train import initialize_trainer, update_config
from utils.task_manager import TaskManager
from config.vocabulary import drum_vocab_presets
from utils.utils import str2bool
from utils.utils import Timer
from utils.audio import slice_padded_array
from utils.note2event import mix_notes, note2note_event, sort_notes, validate_notes, trim_overlapping_notes
from utils.event2note import merge_zipped_note_events_and_ties_to_notes
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



def prepare_media(audio_array: torch.Tensor, sample_rate: int) -> Dict:
    """
    Prepare metadata dictionary for audio tensor.

    This function extracts basic information about an audio tensor and returns
    it in a standardized dictionary format for further processing.

    Args:
        audio_array: Input audio tensor (1D for mono, 2D for multi-channel)
        sample_rate: Sampling rate of the audio in Hz

    Returns:
        Dictionary containing audio metadata with keys:
        - sample_rate: Original sampling rate
        - bits_per_sample: Bit depth (assumed 16 for PCM)
        - num_channels: Number of audio channels
        - num_frames: Total number of audio frames
        - duration: Audio duration in seconds
        - encoding: Audio encoding type (PCM)
    """
    # Handle both mono (1D) and multi-channel (2D) audio
    if audio_array.ndim == 2:
        num_channels, num_frames = audio_array.shape
    else:
        num_channels, num_frames = 1, audio_array.shape[0]

    return {
        "sample_rate": sample_rate,
        "bits_per_sample": 16,  # Standard bit depth for PCM audio
        "num_channels": num_channels,
        "num_frames": num_frames,
        "duration": num_frames / sample_rate,  # Calculate duration in seconds
        "encoding": "pcm"  # Pulse-code modulation encoding
    }



def transcribe_file_notes(model, audio_file: np.ndarray, sample_rate: int) -> List:
    """
    Process audio file to extract musical notes using a trained model.

    This function prepares audio data and uses a machine learning model
    to transcribe musical notes from the audio signal.

    Args:
        model: Pre-trained note transcription model
        audio_file: Input audio as numpy array
        sample_rate: Sampling rate of the audio in Hz

    Returns:
        List of predicted musical notes with timing and pitch information
    """
    # Convert numpy array to PyTorch tensor with float32 precision
    audio_tensor = torch.from_numpy(audio_file).float()

    # Prepare audio metadata for processing
    audio_info = prepare_media(audio_tensor, sample_rate)
    print(f"Audio info: {audio_info}")

    # Transcribe notes using the model
    pred_notes = transcribe_notes(model, audio_info, audio_tensor, sample_rate)

    return pred_notes



def transcribe_notes(model, audio_info: Dict, audio_tensor: torch.Tensor, sample_rate: int) -> List:
    """
    Transcribe musical notes from audio using a trained model.

    This function handles the complete note transcription pipeline including:
    1. Audio preprocessing and resampling
    2. Segmenting audio into processing frames
    3. Running model inference
    4. Post-processing model outputs into musical notes

    Args:
        model: Pre-trained note transcription model
        audio_info: Dictionary containing audio metadata
        audio_tensor: Input audio as PyTorch tensor
        sample_rate: Original sampling rate of the audio in Hz

    Returns:
        List of transcribed musical notes
    """
    timer = Timer()

    # --- Audio Preprocessing ---
    timer.start()

    # Ensure audio has channel dimension (convert mono to shape [1, n])
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    # Resample audio to model's expected sampling rate
    target_sample_rate = model.audio_cfg['sample_rate']
    audio_tensor = torchaudio.functional.resample(audio_tensor, sample_rate, target_sample_rate)

    # Segment audio into overlapping frames for processing
    segment_length = model.audio_cfg['input_frames']
    audio_segments = slice_padded_array(audio_tensor.numpy(), segment_length, segment_length)

    # Move data to appropriate device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_segments = torch.from_numpy(audio_segments.astype('float32')).to(device).unsqueeze(1)

    timer.stop()
    timer.print_elapsed_time("Audio preprocessing")

    # --- Model Inference ---
    timer.start()

    # Run model inference on audio segments
    pred_token_arr, _ = model.inference_file(bsz=8, audio_segments=audio_segments)

    timer.stop()
    timer.print_elapsed_time("Model inference")

    # --- Post-processing ---
    timer.start()

    num_channels = model.task_manager.num_decoding_channels
    n_segments = audio_segments.shape[0]

    # Calculate start times for each audio segment
    start_times = [segment_length * i / target_sample_rate for i in range(n_segments)]

    pred_notes_in_file = []  # Store notes from each channel
    error_counter = Counter()  # Track processing errors

    # Process each output channel separately
    for channel in range(num_channels):
        # Extract predictions for current channel
        pred_token_arr_ch = [arr[:, channel, :] for arr in pred_token_arr]

        # Convert token predictions to note events
        zipped_events, list_events, channel_errors = model.task_manager.detokenize_list_batches(
            pred_token_arr_ch, start_times, return_events=True)

        # Merge note events into complete notes
        pred_notes_ch, channel_note_errors = merge_zipped_note_events_and_ties_to_notes(zipped_events)

        pred_notes_in_file.append(pred_notes_ch)
        error_counter += channel_note_errors

    # Combine notes from all channels
    final_notes = mix_notes(pred_notes_in_file)

    timer.stop()
    timer.print_elapsed_time("Post-processing")

    return final_notes



def extract_GT(annotation_filename):
    annotation_directory = '../../data/guitarset_yourmt3_16k/annotation/'
    annotation_filepath = os.path.join(annotation_directory, annotation_filename)

    if os.path.exists(annotation_filepath):
        print(f"Extracting GT from {annotation_filepath}")
        # Hier könnte ihre Logik stehen
    else:
        print(f"Annotation file not found: {annotation_filepath}")

    assert os.path.exists(annotation_filepath)
    # load annotation
    GT_array = np.load(annotation_filepath, allow_pickle=True)

    data = GT_array.item()  # `.item()` gibt das einzelne Objekt im Array zurück
    notes = data['notes']
    return notes



def estimate_frequency_and_amplitude(prev_phase, current_phase, k, W, H, sr, frame, window):
    if k < 0 or k >= W // 2:
        raise ValueError("Bin index k is out of bounds.")

    omega_k = 2 * np.pi * k / W  # Digitale Frequenz bei Bin k

    epsilon = 1e-10
    delta_phi = omega_k * H + np.mod(current_phase - prev_phase - omega_k * H + np.pi + epsilon, 2 * np.pi) - np.pi

    est_freq = delta_phi / (2 * np.pi * H) * sr

    n = np.arange(W)
    dft_coeff = np.dot(frame * window, np.exp(1j * 2 * np.pi * est_freq / sr * n))
    est_amplitude = 2 * np.abs(dft_coeff) / np.sum(window)

    return est_freq, est_amplitude



def calculate_partials(
        fundamental: float,
        k_fundamental: int,
        prev_fft: np.ndarray,
        curr_fft: np.ndarray,
        fft_size: int,
        hop_size: int,
        sample_rate: float,
        beta_max: float,
        n_partials: int,
        debug: bool,
        time_frame: np.ndarray,
        window: np.ndarray,
        peak_indices: np.ndarray
) -> Dict[int, float]:
    """
    Calculate partial frequencies for a given fundamental frequency.

    Args:
        fundamental: Estimated fundamental frequency in Hz
        k_fundamental: Bin index of the fundamental frequency
        prev_fft: FFT of previous frame (for phase analysis)
        curr_fft: FFT of current frame
        fft_size: Size of the FFT
        hop_size: Hop size between frames in samples
        sample_rate: Audio sample rate in Hz
        beta_max: Maximum allowed inharmonicity coefficient
        n_partials: Maximum number of partials to calculate
        debug: Enable debug output and plotting
        time_frame: Current time domain frame
        window: Window function used for FFT
        peak_indices: Indices of spectral peaks found in current frame

    Returns:
        Dictionary mapping partial numbers to their frequencies in Hz
    """
    partials = {0: fundamental}  # Store fundamental as partial 0

    for partial_num in range(1, n_partials):
        # Calculate expected partial frequency and maximum allowed frequency considering inharmonicity
        harmonic_order = partial_num + 1
        expected_frequency = harmonic_order * fundamental
        max_inharmonic_frequency = expected_frequency * math.sqrt(1 + beta_max * harmonic_order ** 2)

        # Convert frequency to bin index
        k_partial = int(harmonic_order * k_fundamental)
        if k_partial >= fft_size // 2:  # Ensure bin is within valid range
            break

        best_partial = None
        best_amplitude = -float('inf')
        best_bin_offset = None

        # Find peak candidates within the expected frequency range
        peak_candidates = [
            p for p in peak_indices
            if expected_frequency <= p * sample_rate / fft_size <= max_inharmonic_frequency
        ]

        if peak_candidates:
            # Select peak with highest amplitude as candidate
            peak_bin = max(peak_candidates, key=lambda p: np.abs(curr_fft[p]))

            # Search in vicinity of peak bin (-2 to +8 bins)
            for bin_offset in range(-2, 9):
                test_bin = peak_bin + bin_offset
                if test_bin < 0 or test_bin >= fft_size // 2:
                    continue  # Skip invalid bins

                # Estimate precise frequency and amplitude using phase information
                partial_freq, partial_amp = estimate_frequency_and_amplitude(
                    np.angle(prev_fft[test_bin]),
                    np.angle(curr_fft[test_bin]),
                    test_bin,
                    fft_size,
                    hop_size,
                    sample_rate,
                    time_frame,
                    window
                )

                partial_amp_db = 20 * np.log10(np.abs(partial_amp))

                # Update best candidate if amplitude is sufficient
                if partial_amp_db > -40 and partial_amp_db > best_amplitude:
                    best_partial = partial_freq
                    best_amplitude = partial_amp_db
                    best_bin_offset = test_bin - peak_bin

        # Store valid partials
        if best_partial is not None:
            partials[partial_num] = best_partial
            if debug:
                print(f"Bin-Offset {best_bin_offset} used for partial {partial_num} "
                      f"with amplitude {best_amplitude:.2f} dB")

    if debug:
        print(f"Number of partials found: {len(partials)}")

    return partials



def calculate_fundamental(
        k_guess: int,
        prev_fft: np.ndarray,
        curr_fft: np.ndarray,
        fft_size: int,
        hop_size: int,
        sample_rate: float,
        time_frame: np.ndarray,
        window: np.ndarray
) -> float:
    """
    Refine fundamental frequency estimate using phase information.

    Args:
        k_guess: Initial guess for fundamental frequency bin index
        prev_fft: FFT of previous frame
        curr_fft: FFT of current frame
        fft_size: Size of the FFT
        hop_size: Hop size between frames in samples
        sample_rate: Audio sample rate in Hz
        time_frame: Current time domain frame
        window: Window function used for FFT

    Returns:
        Refined fundamental frequency estimate in Hz
    """
    best_fundamental = None
    best_amplitude = -float('inf')

    # Search vicinity of initial guess (±2 bins)
    for bin_offset in range(-2, 3):
        test_bin = k_guess + bin_offset
        if test_bin < 0 or test_bin >= fft_size // 2:
            continue  # Skip invalid bins

        # Estimate precise frequency and amplitude
        fundamental_freq, fundamental_amp = estimate_frequency_and_amplitude(
            np.angle(prev_fft[test_bin]),
            np.angle(curr_fft[test_bin]),
            test_bin,
            fft_size,
            hop_size,
            sample_rate,
            time_frame,
            window
        )

        fundamental_amp_db = 20 * np.log10(np.abs(fundamental_amp))

        # Select candidate with highest amplitude
        if fundamental_amp_db > best_amplitude:
            best_fundamental = fundamental_freq
            best_amplitude = fundamental_amp_db

    return best_fundamental



def estimate_inharmonicity_coefficients(
        string_signal: np.ndarray,
        sample_rate: float,
        notes: List,
        debug: bool
) -> List[List[List[float]]]:
    """
    Estimate inharmonicity coefficients (beta) for guitar strings from audio signal.

    Args:
        string_signal: Audio signal containing guitar notes
        sample_rate: Sampling rate of audio signal in Hz
        notes: List of note objects with onset, offset, pitch, and contour information
        debug: Enable debug output and plotting

    Returns:
        Nested list of inharmonicity coefficients organized by note and frame
    """
    beta_max = 10 * 1.4e-4  # Maximum physically reasonable inharmonicity coefficient
    all_notes_betas = []  # Store beta values for all notes

    for note in notes:
        try:
            # Extract note segment from audio signal
            onset_sample = int(note.onset * sample_rate + note.onsetDiff * sample_rate + 0.02 * sample_rate)
            offset_sample = int(note.offset * sample_rate)
            note_freq = noteToFreq(note.pitch)
            note_signal = string_signal[onset_sample:offset_sample]
        except IndexError:
            continue

        # STFT parameters
        fft_size = 4096 * 2
        hop_size = fft_size // 8

        if len(note_signal) <= fft_size:
            continue

        # Create framed view of signal
        framed_signal = np.lib.stride_tricks.sliding_window_view(note_signal, window_shape=fft_size)[::hop_size]
        if framed_signal.ndim < 2:
            continue

        # Apply padding and windowing
        padded_signal = np.pad(framed_signal, ((0, 0), (0, fft_size - framed_signal.shape[1])), mode='constant')
        window = scipy.signal.windows.hann(fft_size, sym=False)
        window_sum = np.sum(window)
        windowed_signal = padded_signal * window
        freqs = rfftfreq(fft_size, d=1 / sample_rate)

        if len(windowed_signal) <= 2:
            print("Not enough frames for analysis")
            continue

        # Prepare frequency contour interpolation
        contour_times, contour_freqs = zip(*note.contour)
        contour_interp = interpolate.interp1d(contour_times, contour_freqs, kind='linear', fill_value="extrapolate")

        # Initialize tracking variables
        prev_fft = rfft(windowed_signal[0])
        note_betas = []
        frame_times = []
        model_freqs = []
        contour_fundamentals = []
        phase_fundamentals = []

        # Normalize signal
        windowed_signal /= np.max(np.abs(windowed_signal))

        # Process each frame
        for frame_idx, frame in enumerate(windowed_signal[1:], start=1):
            curr_fft = rfft(frame)
            magnitude_spectrum = np.abs(curr_fft)

            # Normalize and convert to dB
            magnitude_spectrum_normalized = (2 * magnitude_spectrum) / window_sum
            magnitude_spectrum_db = 20 * np.log10(magnitude_spectrum_normalized + 1e-10)

            # Find peaks above threshold
            peak_threshold = -50  # dB
            peak_indices, _ = find_peaks(magnitude_spectrum_db, height=peak_threshold)

            if len(peak_indices) == 0:
                print("No peaks detected in frame")
                continue

            # Debug visualization
            if debug:
                plot_spectrum(freqs, magnitude_spectrum_db, peak_indices, note_freq, note.pitch, frame_idx,
                              peak_threshold)

            # Calculate frame time and interpolate fundamental frequency
            frame_time = note.onset + (frame_idx * hop_size) / sample_rate
            frame_times.append(frame_time)
            fundamental_contour = contour_interp(frame_time)
            contour_fundamentals.append(fundamental_contour)
            model_freqs.append(note_freq)

            # Refine fundamental frequency estimate
            k_contour = int(fundamental_contour * fft_size / sample_rate)
            fundamental = calculate_fundamental(
                k_contour, prev_fft, curr_fft, fft_size, hop_size, sample_rate, frame, window
            )
            phase_fundamentals.append(fundamental)

            # Validate fundamental frequency
            quarter_tone_low = note_freq / math.pow(2, 1 / 24)
            quarter_tone_high = note_freq * math.pow(2, 1 / 24)
            if not (quarter_tone_low <= fundamental <= quarter_tone_high):
                print("Fundamental frequency outside expected range")
                continue

            # Calculate partials
            n_partials = min(len(peak_indices), 20)
            partials = calculate_partials(
                fundamental, int(fundamental * fft_size / sample_rate),
                prev_fft, curr_fft, fft_size, hop_size, sample_rate,
                beta_max, n_partials, debug, frame, window, peak_indices
            )

            # Calculate beta values for each partial
            partial_betas = []
            for order, freq in partials.items():
                if order == 0:
                    continue
                beta_val = ((freq / ((order + 1) * partials[0])) ** 2 - 1) / ((order + 1) ** 2)
                partial_betas.append(beta_val)

            # Filter valid beta values
            valid_betas = [b for b in partial_betas if 0 < b < beta_max]

            if valid_betas:
                # Apply IQR filtering to remove outliers
                filtered_betas = filter_outliers_iqr(valid_betas, debug)
                note_betas.append(filtered_betas)
            else:
                print("No valid beta values found")
                note_betas.append(valid_betas)

            # Update previous FFT
            prev_fft = curr_fft

            # Debug visualization
            if debug:
                plot_spectrum_with_partials(
                    freqs, magnitude_spectrum_db, peak_indices, note_freq,
                    n_partials, fundamental, partials, note.pitch, frame_idx, peak_threshold
                )

        # Store results for note
        if note_betas:
            all_notes_betas.append(note_betas)

        # Debug visualization for frequency traces
        if debug and frame_times:
            plot_frequency_traces(
                frame_times, model_freqs, contour_fundamentals,
                phase_fundamentals, note.pitch
            )

    return all_notes_betas


# Helper functions for plotting (optional debug visualizations)
def plot_spectrum(freqs, magnitude_db, peak_indices, note_freq, pitch, frame_idx, threshold):
    """Plot magnitude spectrum with detected peaks."""
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, magnitude_db, label=f"Frame {frame_idx}")
    plt.axhline(threshold, color='black', linestyle='--', label=f'Threshold: {threshold:.2f} dB')
    plt.scatter(note_freq, 0, color="purple", marker="x", s=100, label="Model NoteFreq")
    plt.scatter(freqs[peak_indices], magnitude_db[peak_indices], color='orange', marker="o", label="Detected Peaks")
    plt.title(f"Magnitude spectrum Note: {pitch} (Frame {frame_idx})")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.xscale("log")
    plt.grid()
    plt.legend()
    plt.show()


def plot_spectrum_with_partials(freqs, magnitude_db, peak_indices, note_freq, n_partials, fundamental, partials, pitch,
                                frame_idx, threshold):
    """Plot spectrum with predicted and detected partials."""
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, magnitude_db, label=f"Frame {frame_idx}")
    plt.axhline(threshold, color='black', linestyle='--', label=f'Threshold: {threshold:.2f} dB')
    plt.scatter(note_freq, 0, color="purple", marker="x", s=100, label="Model NoteFreq")
    plt.scatter(freqs[peak_indices], magnitude_db[peak_indices], color='orange', marker="o", label="Detected Peaks")

    # Plot predicted and actual partials
    predicted_partials = [fundamental * (i + 1) for i in range(1, n_partials)]
    plt.scatter(predicted_partials, [-2] * len(predicted_partials), color='red', marker="s", label="Predicted Partials")

    real_partials = list(partials.values())[1:]  # Exclude fundamental
    plt.scatter(real_partials, [-4] * len(real_partials), color='green', marker="d", label="Real Partials")

    plt.title(f"Magnitude Spectrum Note:{pitch} (Frame {frame_idx})")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.xscale("log")
    plt.grid()
    plt.legend()
    plt.show()


def plot_frequency_traces(frame_times, model_freqs, contour_fundamentals, phase_fundamentals, pitch):
    """Plot frequency estimates over time."""
    plt.figure(figsize=(12, 6))
    plt.plot(frame_times, model_freqs, label="Model NoteFreq", color="purple", linestyle="--")
    plt.plot(frame_times, contour_fundamentals, label="Ground Truth Fundamental", color="blue")
    plt.plot(frame_times, phase_fundamentals, label="Phase-Estimated Fundamental", color="orange")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"Frequency Traces for Note {pitch}")
    plt.legend()
    plt.grid()
    plt.show()



def filter_outliers_iqr(beta_values, debug=False):
    """Filter outliers from beta values using IQR method."""
    beta_array = np.array(beta_values)
    Q1 = np.quantile(beta_array, 0.1)
    Q3 = np.quantile(beta_array, 0.9)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    filtered_values = beta_array[(beta_array >= lower_bound) & (beta_array <= upper_bound)].tolist()

    if debug:
        plt.figure(figsize=(8, 5))
        plt.hist(beta_values, bins=100, alpha=0.5, label="Original values", color="blue")
        plt.hist(filtered_values, bins=100, alpha=0.7, label="Filtered values", color="orange")
        plt.axvline(Q1, color="green", linestyle="dashed", label="Q10 (10%)")
        plt.axvline(Q3, color="red", linestyle="dashed", label="Q90 (90%)")
        plt.xlabel("Beta values")
        plt.ylabel("Frequency")
        plt.title("Distribution of partial betas before and after IQR filtering")
        plt.legend()
        plt.grid()
        plt.show()

    return filtered_values



def noteToFreq(note):
    a = 440
    return (a / 32) * (2 ** ((note - 9) / 12))



def get_annotation_filename(audio_filename):
    base_name = audio_filename.split(".")[0]
    if "_pshift" in base_name:
        base_name = base_name.split("_pshift")[0]

    for suffix in ["_hex", "_cln", "_mic", "_debleeded"]:
        base_name = base_name.replace(suffix, "")

    return f"{base_name}_notes.npy"



def flatten_recursive(nested_list: Union[list, np.ndarray]) -> list:
    """
    Recursively flatten a nested list or numpy array structure.

    This function converts a nested structure of lists and arrays into a single
    flat list containing all the elements in depth-first order.

    Args:
        nested_list: A potentially nested structure of lists or numpy arrays

    Returns:
        A flat list containing all elements from the nested structure
    """
    flat_list = []

    # Check if the input is a list or array
    if isinstance(nested_list, (list, np.ndarray)):
        for item in nested_list:
            # If the item is itself a list or array, recursively flatten it
            if isinstance(item, (list, np.ndarray)):
                flat_list.extend(flatten_recursive(item))
            else:
                # Base case: add the element to the flat list
                flat_list.append(item)
    else:
        # Handle case where input is a single element
        flat_list.append(nested_list)

    return flat_list



def match_notes(
        delta: float,
        predicted_notes: List['Note'],
        ground_truth_notes: List['Note'],
        string_index: int
) -> List['Note']:
    """
    Match predicted notes with ground truth notes based on timing and pitch criteria.

    This function compares predicted notes against ground truth notes and creates
    matched note objects when they meet the similarity criteria.

    Args:
        delta: Maximum allowed time difference (in seconds) for note onset matching
        predicted_notes: List of notes detected by the model
        ground_truth_notes: List of reference/ground truth notes
        string_index: Index of the current string being processed (0-5)

    Returns:
        List of matched note objects with combined information from predictions and ground truth
    """
    matched_notes = []

    # Compare each ground truth note with all predicted notes
    for ref_note in ground_truth_notes:
        for pred_note in predicted_notes:
            # Check if notes match based on timing, pitch, and instrument criteria
            if (abs(ref_note.onset - pred_note.onset) <= delta and
                    ref_note.pitch == pred_note.pitch and
                    not pred_note.is_drum and
                    pred_note.program == 24):  # 24 is likely the program ID for guitar

                # Create a new note object combining information from both sources
                matched_note = matchNote(
                    is_drum=ref_note.is_drum,
                    program=ref_note.program,
                    onset=ref_note.onset,
                    onsetDiff=abs(ref_note.onset - pred_note.onset),  # Time difference
                    offset=ref_note.offset,
                    pitch=ref_note.pitch,
                    velocity=ref_note.velocity,
                    contour=ref_note.contour,  # Frequency contour over time
                    string_index=string_index  # Which guitar string this note came from
                )
                matched_notes.append(matched_note)
                break  # Move to next ground truth note after finding a match

    return matched_notes



def process_file(
        audio_signal: np.ndarray,
        sample_rate: int,
        ground_truth_notes: List,
        beta_storage: Dict[str, list],
        debug: bool,
        note_detection_model,
        audio_filepath: str,
        matching_threshold: float,
        annotation_filename: str
) -> Dict[str, List[float]]:
    """
    Process a multi-channel guitar audio file to extract and analyze notes.

    This function processes each string channel of a guitar recording to:
    1. Detect notes using a model
    2. Match them with ground truth annotations
    3. Calculate inharmonicity coefficients (beta values)
    4. Save results to files

    Args:
        audio_signal: Multi-channel audio data with shape (samples, 6 channels)
        sample_rate: Sampling rate in Hz
        ground_truth_notes: Reference note annotations for comparison
        beta_storage: Dictionary to store beta values for each string
        debug: Flag to enable debug output and visualization
        note_detection_model: Model for automatic note detection
        audio_filepath: Path to the source audio file
        matching_threshold: Time threshold for note matching
        annotation_filename: Base name for output files

    Returns:
        Dictionary containing flattened beta values for each string
    """
    all_matched_notes = []  # Collection of all matched notes across all strings

    # Initialize beta storage for each string if not already present
    for i in range(6):
        string_key = f"Saite_{i + 1}"  # Using original German key names
        if string_key not in beta_storage:
            beta_storage[string_key] = []

    # Process each string channel (standard guitar has 6 strings)
    for string_index in range(6):
        # Extract and normalize the current string's signal
        string_signal = audio_signal[:, string_index]
        max_amplitude = np.max(np.abs(string_signal))

        if max_amplitude != 0:
            # Normalize to prevent processing issues with very quiet signals
            string_signal = string_signal / max_amplitude

        print(f"Processing string {string_index + 1}")

        # Detect notes using the provided model
        predicted_notes = transcribe_file_notes(note_detection_model, string_signal, sample_rate)

        # Match predicted notes with ground truth annotations
        matched_notes = match_notes(matching_threshold, predicted_notes, ground_truth_notes, string_index)

        if debug:
            print(f"Number of matched notes: {len(matched_notes)}")

        # Collect all matched notes for saving
        all_matched_notes.extend(matched_notes)

        # Calculate inharmonicity coefficients (beta values) for the matched notes
        string_betas = estimate_inharmonicity_coefficients(string_signal, sample_rate, matched_notes, debug)

        # Store beta values for this string using original German key names
        string_key = f"Saite_{string_index + 1}"
        beta_storage[string_key].append(string_betas)

    # Prepare filename for saving matched notes
    base_filename = os.path.splitext(annotation_filename)[0]  # Remove extension
    base_filename = base_filename.replace("_notes", "")  # Clean up suffix

    # Create save directory and path
    save_dir = "../../data/guitarset_yourmt3_16k/annotation"
    os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
    save_path = os.path.join(save_dir, f"{base_filename}_matchedNotes.npy")

    # Save matched notes as a numpy array
    np.save(save_path, np.array(all_matched_notes, dtype=object))

    # Flatten the beta values structure for easier analysis
    flattened_betas = {}
    for string_key, beta_lists in beta_storage.items():
        flattened_betas[string_key] = []
        for beta_list in beta_lists:
            # Recursively flatten nested beta values
            flattened_betas[string_key].extend(flatten_recursive(beta_list))

    return flattened_betas


def main():
    """
    Main function to process guitar audio files and calculate inharmonicity coefficients.

    This function:
    1. Loads a pre-trained model for note transcription
    2. Processes multiple guitar audio files
    3. Extracts and matches notes between model predictions and ground truth
    4. Calculates inharmonicity coefficients (beta values) for each string
    5. Saves the results to a JSON file
    """
    # Configuration
    model_name = "YPTF+Single (noPS)"
    dbg = False  # Set to True for debugging with early stopping
    plot_mode = False
    max_files_to_process = 4  # Process only this many files when debugging
    print(f"Running evaluation for model: {model_name}")

    # Precision setting for model inference
    precision = 'bf16-mixed'  # Options: ["32", "bf16-mixed", "16"]
    project = '2024'

    # Define model configurations
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
        raise ValueError(f"Unknown model name: {model_name}")

    # Load the pre-trained model
    print("Loading model...")
    model = load_model_checkpoint(args=args)

    # Set up file paths and parameters
    audio_directory = '../../data/guitarset_yourmt3_16k/audio_hex-pickup_debleeded/'

    # Initialize storage for beta values (6 strings)
    betas = {f"Saite_{i + 1}": [] for i in range(6)}

    file_counter = 0
    delta = 0.05  # 50 ms time window for note matching

    # Process each audio file in the directory
    for filename in os.listdir(audio_directory):
        # Skip pitch-shifted files
        if "pshift" in filename:
            continue

        # Construct full file paths
        audio_filepath = os.path.join(audio_directory, filename)
        annotation_filename = get_annotation_filename(filename)

        # Extract ground truth notes from annotation file
        ground_truth_notes = extract_GT(annotation_filename)

        # Load and normalize audio
        audio_signal, sample_rate = sf.read(audio_filepath)
        audio_signal = audio_signal / np.max(np.abs(audio_signal))  # Normalize to [-1, 1]

        # Process the file with debug mode setting
        betas = process_file(
            audio_signal, sample_rate, ground_truth_notes,
            betas, plot_mode, model, audio_filepath, delta, annotation_filename
        )

        # Print current beta values for monitoring
        print(f"Beta values after processing {filename}:")
        for string_key, beta_values in betas.items():
            print(f"  {string_key}: {len(beta_values)} values")

        # Update progress counter
        file_counter += 1
        print(f"Processed file {file_counter}: {filename}")

        # Early stopping for debugging
        if dbg and file_counter >= max_files_to_process:
            print(f"Debug mode: Stopping after processing {max_files_to_process} files")
            break

    # Save beta values to JSON file
    output_filename = '../content/Betas/betas.json'
    with open(output_filename, 'w') as output_file:
        json.dump(betas, output_file, indent=4)

    print(f"Beta values successfully saved to '{output_filename}'.")


if __name__ == "__main__":
    main()