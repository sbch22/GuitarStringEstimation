import sys
import os

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (amt/src/)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
# Add the parent directory to Python's path
sys.path.insert(0, parent_dir)



from collections import Counter
import argparse
import torch
import sys
import json
from model.init_train import initialize_trainer, update_config
from utils.task_manager import TaskManager
from config.vocabulary import drum_vocab_presets
from utils.utils import str2bool
from utils.utils import Timer
from utils.audio import slice_padded_array
from utils.note2event import mix_notes
from utils.event2note import merge_zipped_note_events_and_ties_to_notes
from model.ymt3 import YourMT3
from typing import Dict, Literal
import torchaudio
import numpy as np

#%% @title model helper
def load_model_checkpoint(args=None):
   # Save current working directory
    original_cwd = os.getcwd()

    # Change to the parent directory (amt/src/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
    os.chdir(parent_dir)

    try:
        # Your existing code for argument parsing and model loading
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

    finally:
        # Always revert to the original working directory
        os.chdir(original_cwd)




def prepare_media(source_path_or_url: os.PathLike,
                  source_type: Literal['audio_filepath', 'youtube_url'],
                  delete_video: bool = True) -> Dict:
    """prepare media from source path or youtube, and return audio info"""
    # Get audio_file
    if source_type == 'audio_filepath':
        audio_file = source_path_or_url
    else:
        raise ValueError(source_type)

    # Create info
    info = torchaudio.info(audio_file)
    return {
        "filepath": audio_file,
        "track_name": os.path.basename(audio_file).split('.')[0],
        "sample_rate": int(info.sample_rate),
        "bits_per_sample": int(info.bits_per_sample),
        "num_channels": int(info.num_channels),
        "num_frames": int(info.num_frames),
        "duration": int(info.num_frames / info.sample_rate),
        "encoding": str.lower(info.encoding),
        }


def process_audio_notes(model, audio_filepath):
    """
    Process an audio file and return predicted notes using the given model.

    Args:
        model: Model object used for inference.
        audio_filepath (str): Path to the audio file.

    Returns:
        list or None: Predicted notes if the audio file exists, otherwise None.
    """
    if audio_filepath is None:
        return None

    audio_info = prepare_media(audio_filepath, source_type="audio_filepath")
    print(audio_info)

    pred_notes = transcribe_notes(model, audio_info)
    return pred_notes


def transcribe_notes(model, audio_info):
    """
    Transcribe audio into note events using a trained model.

    This function loads and processes an audio file, runs inference with the
    given model, and post-processes the output tokens into note sequences.

    Args:
        model: Model object providing audio configuration, inference, and
            token-to-note decoding utilities.
        audio_info (dict): Dictionary containing metadata about the audio file,
            must include the key 'filepath'.

    Returns:
        list: Predicted notes from the audio file.
    """
    t = Timer()

    # Convert audio to model input format
    t.start()
    audio, sr = torchaudio.load(uri=audio_info["filepath"])
    audio = torch.mean(audio, dim=0).unsqueeze(0)
    audio = torchaudio.functional.resample(
        audio, sr, model.audio_cfg["sample_rate"]
    )
    audio_segments = slice_padded_array(
        audio,
        model.audio_cfg["input_frames"],
        model.audio_cfg["input_frames"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_segments = (
        torch.from_numpy(audio_segments.astype("float32"))
        .to(device)
        .unsqueeze(1)  # (n_seg, 1, seg_sz)
    )
    t.stop()
    t.print_elapsed_time("converting audio")

    # Run inference
    t.start()
    pred_token_arr, _ = model.inference_file(
        bsz=8, audio_segments=audio_segments
    )
    t.stop()
    t.print_elapsed_time("model inference")

    # Post-process tokens into note sequences
    t.start()
    num_channels = model.task_manager.num_decoding_channels
    n_items = audio_segments.shape[0]
    start_secs_file = [
        model.audio_cfg["input_frames"] * i / model.audio_cfg["sample_rate"]
        for i in range(n_items)
    ]

    pred_notes_in_file = []
    n_err_cnt = Counter()
    for ch in range(num_channels):
        pred_token_arr_ch = [arr[:, ch, :] for arr in pred_token_arr]
        zipped_events_and_ties, list_events, ne_err_cnt = (
            model.task_manager.detokenize_list_batches(
                pred_token_arr_ch,
                start_secs_file,
                return_events=True,
            )
        )
        pred_notes_ch, n_err_cnt_ch = merge_zipped_note_events_and_ties_to_notes(
            zipped_events_and_ties
        )
        pred_notes_in_file.append(pred_notes_ch)
        n_err_cnt += n_err_cnt_ch

    pred_notes = mix_notes(pred_notes_in_file)
    t.stop()
    t.print_elapsed_time("post-processing")

    return pred_notes

def extract_GT(audio_filepath):
    """
    Extract ground-truth (GT) notes from the corresponding annotation file.

    Given the path to an audio file, this function locates the matching
    annotation `.npy` file in the `annotation/` directory, loads it, and
    returns the stored note data.

    Args:
        audio_filepath (str): Path to an audio file.

    Returns:
        list or None: A list of ground-truth notes if available, otherwise None.

    Notes:
        - The annotation file must follow the naming convention:
          `<audio_filename>_notes.npy`
        - Currently, `_mix` is stripped from the filename when searching
          for the annotation file.
    """
    # Build path to annotation directory
    audio_dir = os.path.dirname(os.path.dirname(audio_filepath))
    annotations_dir = os.path.join(audio_dir, "annotation")

    # Derive expected annotation filename
    base_name, _ = os.path.splitext(os.path.basename(audio_filepath))
    filename = base_name.replace("_mix", "")
    annotation_filename = f"{filename}_notes.npy"
    annotation_filepath = os.path.join(annotations_dir, annotation_filename)

    # Check existence
    if not os.path.exists(annotation_filepath):
        print(f"Warning: Annotation file '{annotation_filepath}' does not exist.")
        return None

    # Load annotation data
    GT_array = np.load(annotation_filepath, allow_pickle=True)
    data = GT_array.item()  # extract dict from array
    notes = data["notes"]

    return notes





def evalOnsetF1(delta, pred_notes, GT_notes):
    """
    Evaluate note onset detection using precision, recall, and F-score.

    This function compares predicted notes against ground-truth (GT) notes.
    A predicted note is counted as correct (true positive) if:
        - Its onset is within the tolerance `delta` of the GT onset.
        - Its pitch matches the GT pitch.
        - It is not marked as a drum.
        - It belongs to instrument program 24 (guitar).

    Args:
        delta (float): Onset tolerance in seconds.
        pred_notes (list): List of predicted notes, where each note has
            attributes (onset, pitch, is_drum, program).
        GT_notes (list): List of ground-truth notes, same format as `pred_notes`.

    Returns:
        dict: Evaluation results containing:
            - "True Positives (TP)" (int)
            - "False Positives (FP)" (int)
            - "False Negatives (FN)" (int)
            - "Precision" (float, percentage)
            - "Recall" (float, percentage)
            - "F-Score" (float, percentage)
    """

    # Convert notes into tuples for easier comparison
    ref_notes = [(note.onset, note.pitch, note.is_drum, note.program) for note in GT_notes]
    pred_notes = [(note.onset, note.pitch, note.is_drum, note.program) for note in pred_notes]

    TP = np.zeros(len(ref_notes))  # True Positives
    FN = np.zeros(len(ref_notes))  # False Negatives

    # Iterate through ground-truth notes
    for i, (ref_onset, ref_pitch, ref_isdrum, ref_program) in enumerate(ref_notes):
        # Check differences with all predicted notes
        temp = [
            (abs(ref_onset - pred_onset), pred_pitch == ref_pitch,
             not pred_isdrum, pred_program == 24)
            for pred_onset, pred_pitch, pred_isdrum, pred_program in pred_notes
        ]

        # Mark as true positive if within tolerance and conditions are met
        if any(t[0] <= delta and t[1] and t[2] and t[3] for t in temp):
            TP[i] = 1
        else:
            FN[i] = 1

    # Counts
    no_TP = int(sum(TP))
    no_FN = int(sum(FN))
    no_FP = len(pred_notes) - no_TP

    # Metrics
    precision = no_TP / (no_TP + no_FP) if (no_TP + no_FP) > 0 else 0
    recall = no_TP / (no_TP + no_FN) if (no_TP + no_FN) > 0 else 0
    f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Collect results
    results = {
        "True Positives (TP)": no_TP,
        "False Positives (FP)": no_FP,
        "False Negatives (FN)": no_FN,
        "Precision": round(precision * 100, 1),
        "Recall": round(recall * 100, 1),
        "F-Score": round(f_score * 100, 1),
    }

    return results


def main(dbg: bool = False):
    """
    Run model evaluation on audio files and compute note onset metrics.

    This script loads a specified model checkpoint, processes audio recordings,
    transcribes notes, compares predictions against ground truth, and evaluates
    the performance using onset-based precision, recall, and F-score.

    Usage:
        python f0-trackingYMT3_eval_controller.py <MODEL_NAME>

    Args:
        dbg (bool, optional): If True, run in debug mode and stop after
            processing 2 audio files. Defaults to False.

    Supported models:
        - "YMT3+"
        - "YPTF+Single (noPS)"
        - "YPTF+Multi (PS)"
        - "YPTF.MoE+Multi (noPS)"
        - "YPTF.MoE+Multi (PS)"

    Raises:
        ValueError: If no model name is provided or an unsupported model name
        is given.
    """

    # Check for required model name argument
    if len(sys.argv) < 2:
        raise ValueError("Model name not provided. Please pass the model name as an argument.")

    model_name = sys.argv[1]
    print(f"Running evaluation for model: {model_name}")

    # Evaluation configuration
    precision = "bf16-mixed"  # Options: ["32", "bf16-mixed", "16"]
    project = "2024"

    # Map model name to checkpoint and arguments
    if model_name == "YMT3+":
        checkpoint = "notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72@model.ckpt"
        args = [checkpoint, "-p", project, "-pr", precision]
    elif model_name == "YPTF+Single (noPS)":
        checkpoint = "ptf_all_cross_rebal5_mirst_xk2_edr005_attend_c_full_plus_b100@model.ckpt"
        args = [
            checkpoint, "-p", project, "-enc", "perceiver-tf", "-ac", "spec",
            "-hop", "300", "-atc", "1", "-pr", precision
        ]
    elif model_name == "YPTF+Multi (PS)":
        checkpoint = "mc13_256_all_cross_v6_xk5_amp0811_edr005_attend_c_full_plus_2psn_nl26_sb_b26r_800k@model.ckpt"
        args = [
            checkpoint, "-p", project, "-tk", "mc13_full_plus_256", "-dec", "multi-t5",
            "-nl", "26", "-enc", "perceiver-tf", "-ac", "spec", "-hop", "300",
            "-atc", "1", "-pr", precision
        ]
    elif model_name == "YPTF.MoE+Multi (noPS)":
        checkpoint = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt"
        args = [
            checkpoint, "-p", project, "-tk", "mc13_full_plus_256", "-dec", "multi-t5",
            "-nl", "26", "-enc", "perceiver-tf", "-sqr", "1", "-ff", "moe", "-wf", "4",
            "-nmoe", "8", "-kmoe", "2", "-act", "silu", "-epe", "rope", "-rp", "1",
            "-ac", "spec", "-hop", "300", "-atc", "1", "-pr", precision
        ]
    elif model_name == "YPTF.MoE+Multi (PS)":
        checkpoint = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2@model.ckpt"
        args = [
            checkpoint, "-p", project, "-tk", "mc13_full_plus_256", "-dec", "multi-t5",
            "-nl", "26", "-enc", "perceiver-tf", "-sqr", "1", "-ff", "moe", "-wf", "4",
            "-nmoe", "8", "-kmoe", "2", "-act", "silu", "-epe", "rope", "-rp", "1",
            "-ac", "spec", "-hop", "300", "-atc", "1", "-pr", precision
        ]
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # Load model from checkpoint
    model = load_model_checkpoint(args=args)

    # Audio input directory
    print(f"Current working directory: {os.getcwd()}")
    audio_directory = "../../../data/guitarset_yourmt3_16k/audio_mono-pickup_mix/"
    # Alternative: "../../data/guitarset_yourmt3_16k/audio_mono-mic/"

    # Onset/offset tolerance in seconds
    delta = 0.05  # 50 ms
    onsetF1_list = []

    # Process audio files
    for i, file in enumerate(os.listdir(audio_directory)):
        if "pshift" in file:
            continue

        filename = os.fsdecode(file)
        audio_filepath = os.path.join(audio_directory, filename)

        # Predict and evaluate notes
        pred_notes = process_audio_notes(model, audio_filepath)
        GT_notes = extract_GT(audio_filepath)
        resultsOnsetF1 = evalOnsetF1(delta, pred_notes, GT_notes)

        print(resultsOnsetF1)

        onsetF1_list.append({
            "Precision": resultsOnsetF1["Precision"],
            "Recall": resultsOnsetF1["Recall"],
            "F-Score": resultsOnsetF1["F-Score"]
        })

        # Stop early in debug mode
        if dbg and len(onsetF1_list) >= 2:
            print("Debug mode active: stopping after 2 files.")
            break

    # Compute mean metrics
    mean_precision = np.mean([res["Precision"] for res in onsetF1_list])
    mean_recall = np.mean([res["Recall"] for res in onsetF1_list])
    mean_f_score = np.mean([res["F-Score"] for res in onsetF1_list])

    print(f"\n### AVERAGE METRICS, Model Name: {model_name} ###")
    print(f"Average Precision: {round(mean_precision, 2)}%")
    print(f"Average Recall: {round(mean_recall, 2)}%")
    print(f"Average F-Score: {round(mean_f_score, 2)}%")

    # Save results as JSON
    result = {
        "Precision": round(mean_precision, 4),
        "Recall": round(mean_recall, 4),
        "F-Score": round(mean_f_score, 4),
    }
    output_file = f"results_{model_name.replace(' ', '_')}.json"

    with open(output_file, "w") as f:
        json.dump(result, f)

    print(f"Results for {model_name} saved to {output_file}.")


# %%
if __name__ == "__main__":
    # Pass dbg=True for a shortened run
    main(dbg=True)