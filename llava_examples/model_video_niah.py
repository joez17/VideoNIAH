import argparse
import torch

from operator import attrgetter
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

import torch
import cv2
import numpy as np
from PIL import Image
import requests
import copy
import warnings
from decord import VideoReader, cpu
from transformers import AutoConfig
import json
import os

import math
from tqdm import tqdm
from decord import VideoReader, cpu

import numpy as np


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--video_dir", help="Directory containing video files.", required=True)
    parser.add_argument('--question_fp', help='Path to the question file.', required=True)
    # parser.add_argument("--gt_file_question", help="Path to the ground truth file containing question.", required=True)
    # parser.add_argument("--gt_file_answers", help="Path to the ground truth file containing answers.", required=True)
    parser.add_argument("--output_dir", help="Directory to save the model results JSON.", required=True)
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", required=True)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    # parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    # parser.add_argument("--model-max-length", type=int, default=None)
    # parser.add_argument("--mm_resampler_type", type=str, default="spatial_pool")
    # parser.add_argument("--mm_spatial_pool_stride", type=int, default=4)
    # parser.add_argument("--mm_spatial_pool_out_channels", type=int, default=1024)
    # parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    # parser.add_argument("--image_aspect_ratio", type=str, default="anyres")
    # parser.add_argument("--image_grid_pinpoints", type=str, default="[(224, 448), (224, 672), (224, 896), (448, 448), (448, 224), (672, 224), (896, 224)]")
    # parser.add_argument("--mm_patch_merge_type", type=str, default="spatial_unpad")
    # parser.add_argument("--overwrite", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--frames_num", type=int, default=4)
    return parser.parse_args()


# Function to extract frames from video
def load_video(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    warnings.filterwarnings("ignore")
    # Load the OneVision model
    pretrained = args.model_path
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa")

    model.eval()

    if '.jsonl' in args.question_fp:
        question_dict = [json.loads(q) for q in open(os.path.expanduser(args.question_fp), "r")]
    else:
        question_dict = json.load(open(args.question_fp))
    question_dict = get_chunk(question_dict, args.num_chunks, args.chunk_idx)



    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    video_formats = [".mp4", ".avi", ".mov", ".mkv"]
    if args.num_chunks > 1:
        output_name = f"{args.num_chunks}_{args.chunk_idx}"
    else:
        output_name = args.output_name
    answers_file = os.path.join(args.output_dir, f"{output_name}.json")
    ans_file = open(answers_file, "w")

    index = 0
    for q_dict in tqdm(question_dict):
        # breakpoint()
        q_uid = q_dict['video'].split('/')[-1].replace('.mp4', '')
        if not os.path.exists(q_dict["video"]):
            video_path = os.path.join(args.video_dir, q_dict["video"])
        else:
            video_path = q_dict["video"]

        # Check if the video exists
        if os.path.exists(video_path):
            video_frames = load_video(video_path, args.frames_num)
            # print(video_frames.shape) # (16, 1024, 576, 3)
            image_tensors = []
            frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
            image_tensors.append(frames)

        question0 = q_dict['question']
        options = q_dict['options']
        question = f"{question0}\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\nAnswer with the option's letter from the given choices directly."
        # Process prompt.
        qs = question

        # Prepare conversation input
        conv_template = "qwen_1_5"
        question = f"{DEFAULT_IMAGE_TOKEN}\n{question}"

        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        image_sizes = [frame.size for frame in video_frames]

        with torch.inference_mode():
            output_ids = model.generate(
                        input_ids,
                        images=image_tensors,
                        image_sizes=image_sizes,
                        do_sample=False,
                        temperature=0,
                        max_new_tokens=5,
                        modalities=["video"],
                    )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()

        outputs = outputs.strip()
        gt = chr(options.index(q_dict['gt']) + ord('A'))
        inf_res = {"video_path": q_dict['video'],
                    "prompt": prompt,
                    "pred": outputs,
                    "gt": gt, 
                    "task_type": q_dict['type'],
                    "try": q_dict['try'],
                    "model_id": model_name}
        # print(inf_res)
        ans_file.write(json.dumps(inf_res) + "\n")

        ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)