"""
Module: infer.py
Description:
    Complete pipeline for music generation. This script performs the following stages:
      1. Reading and preparing prompts (genre, lyrics, and optionally an audio prompt).
      2. Stage 1: Initial token generation from the prompts.
      3. Stage 2: Refinement of the tokens generated in Stage 1.
      4. Audio reconstruction from the refined tokens using the xcodec model.
      5. Final audio upsampling using a vocoder.
      6. Post-processing to improve consistency (e.g., low-frequency replacement).
      
    In Stage 2, multiprocessing (using the "spawn" method) is used to distribute work among multiple GPUs.
    The strategy is as follows:
      - Calculate the total number of "batches" (segments of 6 s, 300 tokens each).
      - Divide the total batches into segments equal to the number of GPUs (using all available GPUs if enough work exists).
      - Each process (worker) loads its own Stage 2 model on its assigned GPU, processes its batch group, and returns the result.
      - Finally, the results are concatenated in the correct order to continue the pipeline.
"""

import os
import sys

# Allow importing modules from subdirectories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, 'xcodec_mini_infer'))
sys.path.append(os.path.join(SCRIPT_DIR, 'xcodec_mini_infer', 'descriptaudiocodec'))

from codecmanipulator import CodecManipulator
from collections import Counter
from einops import rearrange
from mmtokenizer import _MMSentencePieceTokenizer
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from xcodec_mini_infer.models.soundstream_hubert_new import SoundStream
from xcodec_mini_infer.vocoder import build_codec_model, process_audio
from xcodec_mini_infer.post_process_audio import replace_low_freq_with_energy_matched
import argparse
import copy
import numpy as np
import re
import soundfile as sf
import torch
import torch.multiprocessing as mp
import torchaudio
import uuid


# =============================================================================
# Class to block a specific range of tokens during generation
# =============================================================================
class BlockTokenRangeProcessor(LogitsProcessor):
    """
    LogitsProcessor that blocks a specific range of token IDs by assigning -∞ to their logits.
    This prevents certain tokens (e.g., reserved tokens) from being generated.
    """
    def __init__(self, start_id, end_id):
        self.blocked_token_ids = list(range(start_id, end_id))

    def __call__(self, input_ids, scores):
        scores[:, self.blocked_token_ids] = -float("inf")
        return scores

# =============================================================================
# Worker function for Stage 2 (runs in an independent process)
# =============================================================================
def stage2_worker(segment_tokens, current_batch, gpu_id, stage2_model_name, tokenizer_model_path, result_queue, segment_index):
    """
    Function to process a segment of tokens in Stage 2 on a specific GPU.
    
    Parameters:
      - segment_tokens (np.ndarray): Segment of tokens (from Stage 1) corresponding to this batch.
      - current_batch (int): Number of batches (segments of 6 s, 300 tokens each) to process.
      - gpu_id (int): ID of the GPU on which the model will run.
      - stage2_model_name (str): Checkpoint or identifier for the Stage 2 model.
      - tokenizer_model_path (str): Path to the tokenizer model.
      - result_queue (mp.Queue): Queue to return the result to the main process.
      - segment_index (int): Index of the segment (for ordering results later).
    
    Process:
      1. Load the Stage 2 model, tokenizer, and the codec tool (Stage 1) on the specified GPU.
      2. Process the tokens: "unflatten" the array and apply the corresponding offset.
      3. Form the input prompt:
         - If processing in batch (current_batch > 1), group tokens for each element to obtain a tensor of shape (current_batch, L).
         - Otherwise, work with a batch of 1.
      4. Iterate over each "frame" (token) of the segment, using teacher forcing to generate 7 new tokens at each iteration.
      5. At the end, extract the generated part (after the initial prompt) and return it via the queue.
    """
    # Set the corresponding device (GPU)
    device = torch.device(f"cuda:{gpu_id}")
    print(f"[GPU {gpu_id}] Starting worker for segment {segment_index}", flush=True)

    # Load the Stage 2 model on the assigned GPU
    model = AutoModelForCausalLM.from_pretrained(
        stage2_model_name,
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()
    print(f"[GPU {gpu_id}] Stage 2 model loaded for segment {segment_index}", flush=True)

    # Load the tokenizer and the codec tool (Stage 1)
    tokenizer = _MMSentencePieceTokenizer(tokenizer_model_path)
    codectool_stage1 = CodecManipulator("xcodec", quantizer_begin=0, n_quantizer=1)

    # Transform the tokens:
    #   - Unflatten: convert the array to the expected shape.
    #   - offset_tok_ids: apply the global offset and adjust based on the codebook size.
    codec_ids = codectool_stage1.unflatten(segment_tokens, n_quantizer=1)
    codec_ids = codectool_stage1.offset_tok_ids(
        codec_ids,
        global_offset=codectool_stage1.global_offset,
        codebook_size=codectool_stage1.codebook_size,
        num_codebooks=codectool_stage1.num_codebooks
    ).astype(np.int32)

    # Form the input prompt:
    if current_batch > 1:
        # Case: batch size > 1
        codec_batches = []
        for i in range(current_batch):
            start_idx = i * 300
            end_idx = (i + 1) * 300
            codec_batches.append(codec_ids[:, start_idx:end_idx])
        # Concatenating each batch produces an array of shape (current_batch, 300)
        codec_ids_concat = np.concatenate(codec_batches, axis=0)
        # The prompt is formed by concatenating:
        #   - The start tokens (SOA and Stage 1 token) replicated for each batch element.
        #   - The segment tokens (already concatenated in batch).
        #   - A token indicating the end of the prompt (Stage 2).
        prompt_ids = np.concatenate([
            np.tile([tokenizer.soa, tokenizer.stage_1], (current_batch, 1)),
            codec_ids_concat,
            np.tile([tokenizer.stage_2], (current_batch, 1)),
        ], axis=1)
        codec_ids_for_loop = codec_ids_concat
    else:
        # Case: batch size = 1
        prompt_ids = np.concatenate([
            np.array([tokenizer.soa, tokenizer.stage_1]),
            codec_ids.flatten(),
            np.array([tokenizer.stage_2])
        ]).astype(np.int32)
        prompt_ids = prompt_ids[np.newaxis, ...]
        codec_ids_for_loop = codec_ids  # Shape (1, L)

    prompt_ids_torch = torch.as_tensor(prompt_ids).to(device)
    len_prompt = prompt_ids_torch.shape[-1]

    # Configure the logits processor (to block certain undesired tokens)
    block_processors = LogitsProcessorList([
        BlockTokenRangeProcessor(0, 46358),
        BlockTokenRangeProcessor(53526, tokenizer.vocab_size)
    ])

    # Total number of "frames" to process in this segment
    total_frames = codec_ids_for_loop.shape[1]

    for frame in range(total_frames):
        current_token = torch.as_tensor(codec_ids_for_loop[:, frame:frame+1]).to(device)
        # Append the current token to the prompt (teacher forcing)
        prompt_ids_torch = torch.cat([prompt_ids_torch, current_token], dim=1)

        attn_mask = torch.ones_like(prompt_ids_torch, dtype=torch.bool)
        with torch.no_grad():
            generate_fn = model.module.generate if hasattr(model, "module") else model.generate
            new_tokens = generate_fn(
                input_ids=prompt_ids_torch,
                attention_mask=attn_mask,
                min_new_tokens=7,
                max_new_tokens=7,
                eos_token_id=tokenizer.eoa,
                pad_token_id=tokenizer.eoa,
                logits_processor=block_processors,
            )
        if new_tokens.shape[1] - prompt_ids_torch.shape[1] != 7:
            raise AssertionError("Expected 7 new tokens per iteration.")
        print(f"[GPU {gpu_id}] Segment {segment_index} processed frame {frame+1}/{total_frames}", flush=True)
        prompt_ids_torch = new_tokens.detach()

    # Extract the generated portion
    if current_batch > 1:
        output = prompt_ids_torch.cpu().numpy()[:, len_prompt:]
        output = np.concatenate([output[i] for i in range(current_batch)], axis=0)
    else:
        output = prompt_ids_torch[0].cpu().numpy()[len_prompt:]

    # Return the result via the queue
    result_queue.put((segment_index, output))
    print(f"[GPU {gpu_id}] Segment {segment_index} completed.", flush=True)


# =============================================================================
# Main class that orchestrates the entire music generation pipeline
# =============================================================================
class MusicGenerationPipeline:
    """
    Main class that coordinates the different stages of the pipeline:
      - Prompt preparation.
      - Stage 1: Token generation (with the Stage 1 model).
      - Stage 2: Token refinement (using multiprocessing to distribute load among GPUs).
      - Audio reconstruction.
      - Upsampling with a vocoder.
      - Post-processing.
    """
    def __init__(self, args):
        self.args = args
        # The main device is used for Stage 1, codecs, reconstruction, etc.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_directories()
        self._load_tokenizer_and_stage1_model()
        self._load_codec_tools()
        try:
            # Parse the list of available GPUs (e.g., "0,1,2,3,4,5")
            self.available_gpu_ids = [int(x.strip()) for x in self.args.gpu_ids.split(",")]
        except Exception as e:
            raise ValueError("Error parsing '--gpu_ids'. It must be a comma-separated string of numbers.") from e

    @staticmethod
    def parse_arguments():
        """
        Defines and parses the command line arguments.
        """
        parser = argparse.ArgumentParser(
            description="Music generation pipeline: Stage 1, Stage 2, audio reconstruction, and upsampling."
        )
        # Model parameters
        parser.add_argument("--stage1_model", type=str, default="m-a-p/YuE-s1-7B-anneal-en-cot",
                            help="Checkpoint or model name for Stage 1.")
        parser.add_argument("--stage2_model", type=str, default="m-a-p/YuE-s2-1B-general",
                            help="Checkpoint or model name for Stage 2.")
        parser.add_argument("--max_new_tokens", type=int, default=3000,
                            help="Maximum number of new tokens generated per pass.")
        parser.add_argument("--run_n_segments", type=int, default=6,
                            help="Number of lyric segments to process.")
        # Batch size for Stage 2 (this parameter is no longer used for grouping, as we will distribute by GPU)
        parser.add_argument("--stage2_batch_size", type=int, default=4,
                            help="Batch size for Stage 2 inference (used to define the duration of each batch, 300 tokens per batch).")
        # Prompt parameters
        parser.add_argument("--genre_txt", type=str, required=True,
                            help="Path to the text file containing genre tags.")
        parser.add_argument("--lyrics_txt", type=str, required=True,
                            help="Path to the text file containing song lyrics.")
        parser.add_argument("--use_audio_prompt", action="store_true",
                            help="If activated, an audio file is used as a reference prompt.")
        parser.add_argument("--audio_prompt_path", type=str, default="",
                            help="Path to the audio file for the prompt (if used).")
        parser.add_argument("--prompt_start_time", type=float, default=0.0,
                            help="Start time (in seconds) for trimming the audio prompt.")
        parser.add_argument("--prompt_end_time", type=float, default=30.0,
                            help="End time (in seconds) for trimming the audio prompt.")
        # Output and other parameters
        parser.add_argument("--output_dir", type=str, default="./output",
                            help="Directory where the results will be saved.")
        parser.add_argument("--keep_intermediate", action="store_true",
                            help="If activated, intermediate files will be kept.")
        parser.add_argument("--disable_offload_model", action="store_true",
                            help="If activated, the Stage 1 model will not be offloaded from the GPU at the end.")
        # Multi-GPU parameters for Stage 2
        parser.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,5",
                            help="Available GPU IDs for Stage 2, separated by commas (e.g., '0,1,2,3,4,5').")
        # Parameters for xcodec and vocoder
        parser.add_argument('--basic_model_config', default='./xcodec_mini_infer/final_ckpt/config.yaml',
                            help="Path to the YAML configuration file for xcodec.")
        parser.add_argument('--resume_path', default='./xcodec_mini_infer/final_ckpt/ckpt_00360000.pth',
                            help="Path to the xcodec checkpoint.")
        parser.add_argument('--config_path', type=str, default='./xcodec_mini_infer/decoders/config.yaml',
                            help="Path to the YAML configuration file for the vocoder.")
        parser.add_argument('--vocal_decoder_path', type=str, default='./xcodec_mini_infer/decoders/decoder_131000.pth',
                            help="Path to the vocal decoder weights.")
        parser.add_argument('--inst_decoder_path', type=str, default='./xcodec_mini_infer/decoders/decoder_151000.pth',
                            help="Path to the instrumental decoder weights.")
        parser.add_argument('-r', '--rescale', action='store_true',
                            help="Rescale audio output to avoid clipping.")
        args = parser.parse_args()

        if args.use_audio_prompt and not args.audio_prompt_path:
            raise FileNotFoundError("Audio prompt activated but no 'audio_prompt_path' provided.")
        return args

    def _setup_directories(self):
        """
        Creates output directories for each stage of the pipeline.
        """
        self.stage1_output_dir = os.path.join(self.args.output_dir, "stage1")
        self.stage2_output_dir = os.path.join(self.args.output_dir, "stage2")
        self.recons_output_dir = os.path.join(self.args.output_dir, "recons")
        os.makedirs(self.stage1_output_dir, exist_ok=True)
        os.makedirs(self.stage2_output_dir, exist_ok=True)
        os.makedirs(self.recons_output_dir, exist_ok=True)

    def _load_tokenizer_and_stage1_model(self):
        """
        Loads the tokenizer and the Stage 1 model, assigning them to the main device.
        """
        print("Loading Stage 1 model and tokenizer...", flush=True)
        self.tokenizer = _MMSentencePieceTokenizer("./mm_tokenizer_v0.2_hf/tokenizer.model")
        self.model_stage1 = AutoModelForCausalLM.from_pretrained(
            self.args.stage1_model,
            torch_dtype=torch.bfloat16,
        ).to(torch.device("cpu"))
        self.model_stage1.eval()

    def _load_codec_tools(self):
        """
        Loads the tools for token (codec) manipulation and the xcodec model.
        """
        print("Loading xcodec and codec tools...", flush=True)
        self.codectool_stage1 = CodecManipulator("xcodec", quantizer_begin=0, n_quantizer=1)
        self.codectool_stage2 = CodecManipulator("xcodec", quantizer_begin=0, n_quantizer=8)
        model_config = OmegaConf.load(self.args.basic_model_config)
        self.codec_model = eval(model_config.generator.name)(**model_config.generator.config).to(self.device)
        parameter_dict = torch.load(self.args.resume_path, map_location='cpu', weights_only=False)
        self.codec_model.load_state_dict(parameter_dict['codec_model'])
        self.codec_model.eval()

    @staticmethod
    def load_audio_mono(filepath, sampling_rate=16000):
        """
        Loads an audio file, converts it to mono, and resamples if necessary.
        """
        audio, sr = torchaudio.load(filepath)
        audio = torch.mean(audio, dim=0, keepdim=True)
        if sr != sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampling_rate)
            audio = resampler(audio)
        return audio

    @staticmethod
    def split_lyrics(lyrics_text):
        """
        Splits the lyrics text into segments using regular expressions.
        """
        pattern = r"\[(\w+)\](.*?)(?=\[|\Z)"
        segments = re.findall(pattern, lyrics_text, re.DOTALL)
        return [f"[{seg[0]}]\n{seg[1].strip()}\n\n" for seg in segments]

    @staticmethod
    def save_audio(wav, filepath, sample_rate, rescale=False):
        """
        Saves an audio tensor to a file (WAV/MP3).
        
        Args:
            wav (torch.Tensor): Audio tensor of shape (channels, samples).
            filepath (str): Output file path.
            sample_rate (int): Sampling rate.
            rescale (bool): If True, rescales to avoid clipping.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        limit = 0.99
        max_val = wav.abs().max()
        if rescale:
            wav = wav * min(limit / max_val, 1)
        else:
            wav = wav.clamp(-limit, limit)
        torchaudio.save(filepath, wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)

    def prepare_prompts(self):
        """
        Reads the genre and lyrics files, and prepares the prompts to be used in Stage 1.
        """
        print("Preparing prompts...", flush=True)
        with open(self.args.genre_txt, "r") as f:
            self.genre = f.read().strip()
        with open(self.args.lyrics_txt, "r") as f:
            lyrics_text = f.read()
        self.lyrics_segments = self.split_lyrics(lyrics_text)
        self.full_lyrics = "\n".join(self.lyrics_segments)
        # The first prompt is global; the subsequent ones refer to specific segments.
        self.prompt_texts = [f"Generate music from the given lyrics segment by segment.\n[Genre] {self.genre}\n{self.full_lyrics}"]
        self.prompt_texts.extend(self.lyrics_segments)

    def run_stage1(self):
        """
        Executes Stage 1: generates initial tokens from the prompts and saves them in .npy files.
        """
        print("Running Stage 1 (token generation)...", flush=True)
        top_p = 0.93
        temperature = 1.0
        repetition_penalty = 1.2
        # The first prompt is global (without generation), so we add 1 to the number of segments.
        n_segments = min(self.args.run_n_segments + 1, len(self.lyrics_segments))
        session_id = uuid.uuid4()
        raw_output = None
        self.stage1_output_files = []
        sos_token = self.tokenizer.tokenize('[start_of_segment]')
        eos_token = self.tokenizer.tokenize('[end_of_segment]')
        stage1_device = torch.device("cpu")

        # Process each segment (skipping the global prompt for generation)
        for idx, prompt in enumerate(tqdm(self.prompt_texts[:n_segments], desc="Stage 1 Segments")):
            # The global prompt (index 0) is used for context, without generation.
            if idx == 0:
                continue

            # Prepare the text for the segment
            section_text = prompt.replace('[start_of_segment]', '').replace('[end_of_segment]', '')
            guidance_scale = 1.5 if idx <= 1 else 1.2

            # In the first segment, an audio prompt can be incorporated if activated
            if idx == 1:
                if self.args.use_audio_prompt:
                    audio = self.load_audio_mono(self.args.audio_prompt_path)
                    audio = audio.unsqueeze(0)
                    with torch.no_grad():
                        audio_codes = self.codec_model.encode(audio.to(stage1_device), target_bw=0.5)
                    audio_codes = audio_codes.transpose(0, 1).cpu().numpy().astype(np.int16)
                    # Convert the numpy array to tokens using the codec tool
                    audio_token_ids = self.codectool_stage1.npy2ids(audio_codes[0])
                    # Trim the audio prompt according to the indicated times (assuming 50 fps)
                    start_frame = int(self.args.prompt_start_time * 50)
                    end_frame = int(self.args.prompt_end_time * 50)
                    audio_tokens = audio_token_ids[start_frame:end_frame]
                    # Build the reference prompt with special tokens
                    ref_prompt = (self.tokenizer.tokenize("[start_of_reference]") +
                                  [self.tokenizer.soa] +
                                  self.codectool_stage1.sep_token_ids() +
                                  audio_tokens +
                                  [self.tokenizer.eoa] +
                                  self.tokenizer.tokenize("[end_of_reference]"))
                    base_prompt = self.tokenizer.tokenize(self.prompt_texts[0]) + ref_prompt
                else:
                    base_prompt = self.tokenizer.tokenize(self.prompt_texts[0])
                segment_prompt = base_prompt + sos_token + self.tokenizer.tokenize(section_text) + [self.tokenizer.soa] + self.codectool_stage1.sep_token_ids()
            else:
                segment_prompt = eos_token + sos_token + self.tokenizer.tokenize(section_text) + [self.tokenizer.soa] + self.codectool_stage1.sep_token_ids()

            # Convert the prompt to a tensor and move it to the device
            prompt_tensor = torch.as_tensor(segment_prompt, dtype=torch.long).unsqueeze(0).to(stage1_device)
            if idx > 1:
                input_tensor = torch.cat([raw_output, prompt_tensor], dim=1)
            else:
                input_tensor = prompt_tensor

            # Duplicate tokens for the final part to avoid abrupt cuts
            max_new_tokens = self.args.max_new_tokens*2 if idx == n_segments - 1 else self.args.max_new_tokens
            # Ensure that the input does not exceed the model's context window
            # Use window slicing in case the output sequence exceeds the model's context size
            max_context = 16384 - max_new_tokens - 1
            if input_tensor.shape[-1] > max_context:
                print(f"Segment {idx}: Input exceeds context window; using the last {max_context} tokens.", flush=True)
                input_tensor = input_tensor[:, -max_context:]
            
            # Create the attention mask manually, using torch.bool as masks are boolean by default
            attention_mask = torch.ones_like(input_tensor, dtype=torch.bool)

            # Configure a logits processor to block undesired token ranges
            logits_processor = LogitsProcessorList([
                BlockTokenRangeProcessor(0, 32002),
                BlockTokenRangeProcessor(32016, 32016)
            ])

            with torch.no_grad():
                generate_fn = self.model_stage1.module.generate if hasattr(self.model_stage1, "module") else self.model_stage1.generate
                output_tokens = generate_fn(
                    input_ids=input_tensor,
                    attention_mask=attention_mask,
                    # For the last segment, duplicate tokens so the song ends correctly
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=100,
                    do_sample=True,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    eos_token_id=self.tokenizer.eoa,
                    pad_token_id=self.tokenizer.eoa,
                    logits_processor=logits_processor,
                    guidance_scale=guidance_scale,
                )
                # If the generated sequence does not end with the end token, append it.
                if output_tokens[0, -1].item() != self.tokenizer.eoa:
                    eos_tensor = torch.as_tensor([[self.tokenizer.eoa]], dtype=torch.long).to(stage1_device)
                    output_tokens = torch.cat([output_tokens, eos_tensor], dim=1)

            # Accumulate the generated output
            if idx > 1:
                new_tokens = output_tokens[:, input_tensor.shape[-1]:]
                raw_output = torch.cat([raw_output, prompt_tensor, new_tokens], dim=1)
            else:
                raw_output = output_tokens
                
            print(f"\nSegment {idx} processed, output shape: {output_tokens.shape}")

        # Extract tokens corresponding to vocals and instruments
        print("Extracting vocal and instrumental tokens...")
        output_ids = raw_output[0].cpu().numpy()
        soa_indices = np.where(output_ids == self.tokenizer.soa)[0].tolist()
        eoa_indices = np.where(output_ids == self.tokenizer.eoa)[0].tolist()
        if len(soa_indices) != len(eoa_indices):
            raise ValueError(f"Mismatch between SOA and EOA tokens: {len(soa_indices)} vs {len(eoa_indices)}.")

        vocal_tokens = []
        instrumental_tokens = []
        start_idx = 1 if self.args.use_audio_prompt else 0
        for i in range(start_idx, len(soa_indices)):
            seg_tokens = output_ids[soa_indices[i] + 1 : eoa_indices[i]]
            # Remove an extra separator token if present
            if seg_tokens[0] == 32016:
                seg_tokens = seg_tokens[1:]
            # Ensure the number of tokens is even to allow separation
            seg_tokens = seg_tokens[: 2 * (len(seg_tokens) // 2)]
            # Split tokens into two channels (vocal and instrumental)
            vocal_seg = self.codectool_stage1.ids2npy(rearrange(seg_tokens, "(n b) -> b n", b=2)[0])
            instrumental_seg = self.codectool_stage1.ids2npy(rearrange(seg_tokens, "(n b) -> b n", b=2)[1])
            vocal_tokens.append(vocal_seg)
            instrumental_tokens.append(instrumental_seg)

        vocals_array = np.concatenate(vocal_tokens, axis=1) if vocal_tokens else np.array([], dtype=np.int16)
        instruments_array = np.concatenate(instrumental_tokens, axis=1) if instrumental_tokens else np.array([], dtype=np.int16)

        # Save Stage 1 results as .npy files
        vocal_file = os.path.join(
            self.stage1_output_dir,
            f"cot_{self.genre.replace(' ', '-')}_vocal_{session_id}.npy"
        )
        instrumental_file = os.path.join(
            self.stage1_output_dir,
            f"cot_{self.genre.replace(' ', '-')}_instrumental_{session_id}.npy"
        )
        np.save(vocal_file, vocals_array)
        np.save(instrumental_file, instruments_array)
        self.stage1_output_files.extend([vocal_file, instrumental_file])

        # Optionally offload the Stage 1 model from the GPU to free memory
        if not self.args.disable_offload_model:
            self.model_stage1.cpu()
            del self.model_stage1
            torch.cuda.empty_cache()

    def run_stage2(self):
        """
        Executes Stage 2: refinement of tokens generated in Stage 1.
        
        Procedure (refactored for one batch per GPU in sequential groups):
        1. For each Stage 1 file, calculate the total number of complete batches
           (6 s * 50 fps = 300 tokens per batch).
        2. Each batch is treated as a "segment". These segments are processed in groups,
           where each GPU handles one segment.
        3. The results are collected and concatenated in order before moving to the next group.
        4. If there are extra tokens, they are processed in the usual single-process manner.
        5. The final result is saved to disk.
        """
        print("Running Stage 2 (token refinement)...", flush=True)
        stage2_files = []
        
        for stage1_file in tqdm(self.stage1_output_files, desc="Processing Stage 2"):
            output_file = os.path.join(self.stage2_output_dir, os.path.basename(stage1_file))
            if os.path.exists(output_file):
                print(f"File {output_file} already exists. Skipping.", flush=True)
                stage2_files.append(output_file)
                continue

            tokens = np.load(stage1_file).astype(np.int32)
            # Calculate the duration in seconds (multiple of 6 s) and total complete batches
            output_duration = (tokens.shape[-1] // 50) // 6 * 6
            total_batches = output_duration // 6  # Each batch = 6 s = 300 tokens
            print(f"Total batches: {total_batches}", flush=True)
            
            # If there are no complete batches, process the extra tokens directly.
            if total_batches == 0:
                print("No complete batches, processing extra tokens directly.", flush=True)
                refined = np.array([])  # Will be concatenated later
            else:
                n_gpus = len(self.available_gpu_ids)
                print(f"Using up to {n_gpus} GPUs: {self.available_gpu_ids}", flush=True)
                
                # Dictionary to store results of all segments (ordered by index)
                all_refined_segments = {}
                collected_total = 0
                
                # Process batches sequentially in groups of size n_gpus
                # Each batch is a "segment" of 300 tokens
                batch_index = 0
                group_id = 0
                
                while batch_index < total_batches:
                    # Calculate how many batches to process in this group (may be < n_gpus if not enough remain)
                    group_size = min(n_gpus, total_batches - batch_index)
                    
                    processes = []
                    result_queue = mp.Queue()

                    # Launch one process per segment in this group
                    for i in range(group_size):
                        current_batch = batch_index + i
                        start_idx = current_batch * 300
                        end_idx   = start_idx + 300
                        
                        segment_tokens = tokens[:, start_idx:end_idx]
                        
                        gpu_id = self.available_gpu_ids[i]  # Assign GPU based on local group index
                        
                        print(f"[Group {group_id}] Dispatching batch #{current_batch+1}/{total_batches} to GPU {gpu_id} (tokens [{start_idx}:{end_idx}])", flush=True)
                        
                        p = mp.Process(
                            target=stage2_worker,
                            args=(
                                segment_tokens,
                                1,  # Process only 1 batch
                                gpu_id,
                                self.args.stage2_model,
                                "./mm_tokenizer_v0.2_hf/tokenizer.model",
                                result_queue,
                                current_batch  # Use 'current_batch' as unique index
                            )
                        )
                        processes.append(p)
                        p.start()
                    
                    # Collect results from this group
                    collected = 0
                    while collected < group_size:
                        try:
                            seg_index, out = result_queue.get(timeout=10)
                            all_refined_segments[seg_index] = out
                            collected += 1
                            collected_total += 1
                            print(f"[Group {group_id}] Received result for batch #{seg_index+1}. Collected: {collected}/{group_size}", flush=True)
                        except Exception as e:
                            print(f"Stage 2 results already received: {collected}/{group_size}", e, flush=True)
                    
                    # Wait for all processes in this group to finish
                    for p in processes:
                        p.join()
                    
                    # Move to the next group
                    batch_index += group_size
                    group_id += 1
                
                # Now that all refined segments are in all_refined_segments, concatenate them in order.
                refined_list = [all_refined_segments[i] for i in sorted(all_refined_segments.keys())]
                refined = np.concatenate(refined_list, axis=0) if refined_list else np.array([])
            
            print("Batch processing complete, checking for extra tokens at the end of the prompt.", flush=True)
            # Process extra tokens if they exist
            extra_tokens = tokens[:, output_duration * 50:]
            if extra_tokens.size > 0:
                print(f"Processing final part of Stage 2 on GPU {self.available_gpu_ids[0]}", flush=True)
                result_queue = mp.Queue()
                stage2_worker(
                    extra_tokens,
                    1,
                    self.available_gpu_ids[0],
                    self.args.stage2_model,
                    "./mm_tokenizer_v0.2_hf/tokenizer.model",
                    result_queue,
                    total_batches  # use 'total_batches' as a higher index to append at the end
                )
                
                try:
                    seg_index_extra, extra = result_queue.get(timeout=30)
                    print(f"Received result for final part (index {seg_index_extra})", flush=True)
                except Exception as e:
                    print("Error obtaining result for final part:", e, flush=True)
                    extra = None
                
                if extra is not None:
                    refined = np.concatenate([refined, extra], axis=0) if refined.size else extra

                print("Final part processed.", flush=True)
            else:
                print("No extra tokens to process.", flush=True)

            # Convert the refined tokens to the original representation using the Stage 2 codec tool
            refined_npy = self.codectool_stage2.ids2npy(refined)

            # Fix any token outside the range [0,1023] by replacing it with the most frequent token in that line
            fixed = copy.deepcopy(refined_npy)
            for i, line in enumerate(refined_npy):
                for j, token in enumerate(line):
                    if token < 0 or token > 1023:
                        counts = Counter(line)
                        most_common = sorted(counts.items(), key=lambda x: x[1], reverse=True)[0][0]
                        fixed[i, j] = most_common

            np.save(output_file, fixed)
            stage2_files.append(output_file)

        self.stage2_output_files = stage2_files

    def reconstruct_audio(self):
        """
        Reconstructs audio from the refined Stage 2 tokens using the xcodec model.
        Each reconstructed track is saved, and if both vocal and instrumental tracks exist,
        a mixed version is also generated.
        """
        print("Reconstructing audio from refined tokens...", flush=True)
        self.recons_mix_dir = os.path.join(self.recons_output_dir, "mix")
        os.makedirs(self.recons_mix_dir, exist_ok=True)
        self.reconstructed_tracks = []
        for file in self.stage2_output_files:
            tokens = np.load(file)
            with torch.no_grad():
                waveform = self.codec_model.decode(
                    torch.as_tensor(tokens.astype(np.int16), dtype=torch.long)
                    .unsqueeze(0).permute(1, 0, 2).to(self.device)
                )
            waveform = waveform.cpu().squeeze(0)
            out_path = os.path.join(self.recons_output_dir, os.path.splitext(os.path.basename(file))[0] + ".mp3")
            self.reconstructed_tracks.append(out_path)
            self.save_audio(waveform, out_path, sample_rate=16000, rescale=self.args.rescale)

        # Mix vocal and instrumental tracks (if both are available)
        self.recons_mix = None
        for track in self.reconstructed_tracks:
            try:
                if track.endswith(('.wav', '.mp3')) and 'instrumental' in track:
                    vocal_track = track.replace('instrumental', 'vocal')
                    if not os.path.exists(vocal_track):
                        continue
                    mix_path = os.path.join(self.recons_mix_dir, os.path.basename(track).replace('instrumental', 'mixed'))
                    vocal_wave, sr = sf.read(vocal_track)
                    instrumental_wave, _ = sf.read(track)
                    mix_wave = (vocal_wave + instrumental_wave) / 1.0
                    sf.write(mix_path, mix_wave, sr)
                    self.recons_mix = mix_path
            except Exception as e:
                print(f"Error mixing {track}: {e}", flush=True)

    def upsample_with_vocoder(self):
        """
        Performs audio upsampling using decoders (vocoder).
        Vocal and instrumental tracks are processed to generate a final mix.
        """
        print("Upsampling audio with vocoder...", flush=True)
        vocal_decoder, inst_decoder = build_codec_model(
            self.args.config_path,
            self.args.vocal_decoder_path,
            self.args.inst_decoder_path
        )
        vocoder_out_dir = os.path.join(self.args.output_dir, 'vocoder')
        vocoder_stems_dir = os.path.join(vocoder_out_dir, 'stems')
        vocoder_mix_dir = os.path.join(vocoder_out_dir, 'mix')
        os.makedirs(vocoder_stems_dir, exist_ok=True)
        os.makedirs(vocoder_mix_dir, exist_ok=True)

        vocal_audio = None
        instrumental_audio = None
        for file in self.stage2_output_files:
            if 'instrumental' in file:
                instrumental_audio = process_audio(
                    file,
                    os.path.join(vocoder_stems_dir, 'instrumental.mp3'),
                    self.args.rescale,
                    self.args,
                    inst_decoder,
                    self.codec_model
                )
            else:
                vocal_audio = process_audio(
                    file,
                    os.path.join(vocoder_stems_dir, 'vocal.mp3'),
                    self.args.rescale,
                    self.args,
                    vocal_decoder,
                    self.codec_model
                )
        if vocal_audio is not None and instrumental_audio is not None:
            try:
                final_mix = instrumental_audio + vocal_audio
                vocoder_mix_path = os.path.join(vocoder_mix_dir, os.path.basename(self.recons_mix)) \
                    if self.recons_mix else os.path.join(vocoder_mix_dir, "mixed_final_output.wav")
                self.save_audio(final_mix, vocoder_mix_path, 44100, self.args.rescale)
                print(f"Final mix generated: {vocoder_mix_path}", flush=True)
            except RuntimeError as e:
                print("Error during final mix:", e, flush=True)
                vocoder_mix_path = None
        else:
            vocoder_mix_path = None
        self.vocoder_mix = vocoder_mix_path

    def post_process(self):
        """
        Applies post-processing to improve the consistency between the reconstructed signal
        and the vocoder output, for example by replacing low frequencies.
        """
        if self.recons_mix is not None and self.vocoder_mix is not None:
            print("Applying post-processing to audio...", flush=True)
            try:
                postproc_file = os.path.join(self.args.output_dir, os.path.basename(self.recons_mix))
                replace_low_freq_with_energy_matched(
                    a_file=self.recons_mix,
                    b_file=self.vocoder_mix,
                    c_file=postproc_file,
                    cutoff_freq=5500.0
                )
                print(f"Post-processing completed. Final file: {postproc_file}", flush=True)
            except Exception as e:
                print("Error during post-processing:", e, flush=True)

    def run(self):
        """
        Runs all stages of the pipeline sequentially:
          1. Prompt preparation.
          2. Stage 1: Token generation.
          3. Stage 2: Token refinement.
          4. Audio reconstruction.
          5. Upsampling with vocoder.
          6. Post-processing.
        """
        self.prepare_prompts()
        # For testing, pre-existing Stage 1 files may be used:
        self.run_stage1()
        self.run_stage2()
        self.reconstruct_audio()
        self.upsample_with_vocoder()
        self.post_process()
        print("Music generation pipeline completed successfully.", flush=True)


def main():
    # Set the multiprocessing start method to "spawn"
    mp.set_start_method("spawn", force=True)
    args = MusicGenerationPipeline.parse_arguments()
    pipeline = MusicGenerationPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
