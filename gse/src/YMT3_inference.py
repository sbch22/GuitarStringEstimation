import os
import sys
from contextlib import contextmanager

@contextmanager
def temporarily_add_to_syspath(path: str):
    abs_path = os.path.abspath(path)
    sys.path.insert(0, abs_path)
    try:
        yield
    finally:
        sys.path.remove(abs_path)

# --- Imports that depend on amt ---
current_file_dir = os.path.dirname(os.path.abspath(__file__))
amt_src_path = os.path.join(current_file_dir, "../../amt/src")

with temporarily_add_to_syspath(amt_src_path):
    from model.init_train import initialize_trainer, update_config
    from utils.task_manager import TaskManager
    from config.vocabulary import drum_vocab_presets
    from utils.utils import str2bool, Timer
    from utils.audio import slice_padded_array
    from utils.note2event import mix_notes, note2note_event, sort_notes, validate_notes, trim_overlapping_notes
    from utils.event2note import merge_zipped_note_events_and_ties_to_notes
    from model.ymt3 import YourMT3


sys.path.append(os.path.abspath(''))

from collections import Counter
import argparse
import torch
import _pickle as pickle
from typing import Dict, List
import torchaudio
from gse.src.utils.FeatureNote_dataclass import FeatureNote, Attributes, FilterReason
from gse.src.utils.Track_dataclass import filter_analysis
import numpy as np
import pyfar as pf
from collections import defaultdict
from configparser import ConfigParser
import gse.src.utils.FeatureNote_dataclass as FeatureNote_dataclass
sys.modules["utils.FeatureNote_dataclass"] = FeatureNote_dataclass

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
                        help='evaluation vocabulary (default=None). If None, default vocabulary of the noteData preset will be used.')
    parser.add_argument('-edv', '--eval-drum-vocab', type=str, default=None,
                        help='evaluation vocabulary for drum (default=None). If None, default vocabulary of the noteData preset will be used.')
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
    parser.add_argument('--ckpt-path', type=str, default=None)
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
    if args.ckpt_path:
        ckpt_path = Path(args.ckpt_path)
    else:
        rel_path = dir_info["last_ckpt_path"]
        rel_path = rel_path.lstrip("../")

        ckpt_path = PROJECT_ROOT / "amt" / rel_path

    ckpt_path = ckpt_path.resolve()
    print(f"Resolved checkpoint path: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    state_dict = checkpoint['state_dict']
    new_state_dict = {k: v for k, v in state_dict.items() if 'pitchshift' not in k}

    model.load_state_dict(new_state_dict, strict=False)

    return model.eval()

def transcribe_notes(model, audio_info: Dict, audio_tensor: torch.Tensor, sample_rate: int) -> List:
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

    # Move noteData to appropriate device (GPU if available)
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

def process_GuitarSet_track(track, model):
    # load audio
    filepath_hex_debeleed = track.audio_paths["hex_debleeded"]
    strings_signal = pf.io.read_audio(filepath_hex_debeleed)

    # strings_signal = track.audio.hex
    sample_rate = strings_signal.sampling_rate
    strings_audio_data = strings_signal.time

    for i in range(0, 6):
        string_audio_data = strings_audio_data[i, :]
        audio_tensor = torch.from_numpy(string_audio_data).float()
        audio_info = {
            'sample_rate': sample_rate,
            'bits_per_sample': 16, # where do I get this from pyfar?
            'num_channels': 1,
            'num_frames': string_audio_data.shape[-1],
            'duration': string_audio_data.shape[-1] / sample_rate,
            'encoding': 'pcm',
        }
        print(f"Audio info: {audio_info}")

        model_string_notes = transcribe_notes(model, audio_info, audio_tensor, sample_rate)

        for note in model_string_notes:
            # Create an Attributes object for this note
            attr = Attributes(
                midi_note=note.pitch,  # redundant but often useful for comparison
                string_index=i,
                is_drum=note.is_drum,
                program=note.program,
                onset=note.onset,
                offset=note.offset,
                velocity=note.velocity,
                pitch=round(440 * 2 ** ((note.pitch - 69) / 12), 3)

            )

            # Wrap it into a FeatureNote
            fnote = FeatureNote(
                origin='model',
                attributes=attr,
                valid=True
            )

            # Append to this track’s note list
            track.notes.append(fnote)

    track.match_notes(track, delta=0.050) # 50ms
    track.match_notes_between_strings(strings_signal, 0.05, track.notes)

    for note in track.valid_notes:
        note.what_fret()

    filter_analysis(track.notes)
    print("Next Track ...")


def main(track_directory, audio_type):
    # model config
    model_name = "YPTF+Single (noPS)"
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

    print("Loading model...")
    model = load_model_checkpoint(args=args)

    file_counter = 0
    # Process each audio file in the directory
    for filename in os.listdir(track_directory):
        # skip directories
        if os.path.isdir(os.path.join(track_directory, filename)):
            continue

        filepath = os.path.join(track_directory, filename)

        with open(filepath, "rb") as f:
            track = pickle.load(f)

        # Process the file with debug mode setting
        if "GuitarSet" in filepath:
            process_GuitarSet_track(track, model)

        save_path = os.path.join(track_directory, filename)

        track.save(save_path)
        print(f"pickled model-matched note object {filename} into {save_path}.")
        file_counter += 1
        print(f"Processed file {file_counter}: {filename}")

if __name__ == "__main__":
    main('../../data/GuitarSet/noteData/train/', audio_type="hex_debleeded")
    main('../../data/GuitarSet/noteData/test/solo/', audio_type="hex_debleeded")
    main('../../data/GuitarSet/noteData/test/comp/', audio_type="hex_debleeded")
