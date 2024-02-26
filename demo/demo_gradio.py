import shutil
import zipfile
from datetime import time
from io import BytesIO
from multiprocessing.dummy import Pool

import PIL
import gradio as gr
import os
from os import path
import tempfile

from argparse import ArgumentParser
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm

from deva.inference.eval_args import add_common_eval_args, get_model_and_config
from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.result_utils import ResultSaver
from deva.inference.demo_utils import flush_buffer
from deva.ext.grounding_dino import get_grounding_dino_model
from deva.ext.automatic_sam import get_sam_model
from deva.ext.ext_eval_args import add_ext_eval_args, add_text_default_args, add_auto_default_args
from deva.ext.automatic_processor import process_frame_automatic as process_frame_auto
from deva.ext.with_text_processor import process_frame_with_text as process_frame_text


def round2(num):
    num = round(num)
    if num % 2 != 0:
        num += 1
    return num


def get_frames_from_zip(file_input, resize_ratio_factor=1.0):
    """
    Args:
        video_path:str
        timestamp:float64
    Return
        [[0:nearest_frame], [nearest_frame:], nearest_frame]
    """
    print(file_input)
    with zipfile.ZipFile(file_input.name) as zip_ref:
        img_file_names = [_.filename for _ in zip_ref.infolist() if not _.is_dir()]
        img_file_names = sorted(img_file_names)

        frames = [None] * len(img_file_names)
        exifs = [None] * len(img_file_names)

        def extract_frame(img_file_name_i):
            img_file_name, i = img_file_name_i
            with zip_ref.open(img_file_name) as file:
                image = Image.open(BytesIO(file.read()))
                image = PIL.ImageOps.exif_transpose(image)
                max_length = max(image.size)
                resize_ratio = resize_ratio_factor * 1600. / max_length
                image = image.resize((round2(image.size[0] * resize_ratio), round2(image.size[1] * resize_ratio)), Image.LANCZOS)
                frames[i] = np.array(image)
                exifs[i] = image.info.get('exif', None)
        with Pool() as pool:
            list(tqdm(pool.imap_unordered(extract_frame, zip(img_file_names, range(len(img_file_names)))), total=len(img_file_names)))

    return frames, exifs


def demo_with_text(file: gr.File, text: str, threshold: float, max_num_objects: int,
                   internal_resolution: int, detection_every: int, max_missed_detection: int,
                   chunk_size: int, sam_variant: str, temporal_setting: str):
    np.random.seed(42)
    torch.autograd.set_grad_enabled(False)
    parser = ArgumentParser()
    add_common_eval_args(parser)
    add_ext_eval_args(parser)
    add_text_default_args(parser)
    deva_model, cfg, _ = get_model_and_config(parser)
    cfg['prompt'] = text
    cfg['enable_long_term_count_usage'] = True
    cfg['max_num_objects'] = max_num_objects
    cfg['size'] = internal_resolution
    cfg['DINO_THRESHOLD'] = threshold
    cfg['amp'] = True
    cfg['chunk_size'] = chunk_size
    cfg['detection_every'] = detection_every
    cfg['max_missed_detection_count'] = max_missed_detection
    cfg['sam_variant'] = sam_variant
    cfg['temporal_setting'] = temporal_setting
    gd_model, sam_model = get_grounding_dino_model(cfg, 'cuda')

    deva = DEVAInferenceCore(deva_model, config=cfg)
    deva.next_voting_frame = cfg['num_voting_frames'] - 1
    deva.enabled_long_id()

    print('Configuration:', cfg)

    # obtain temporary directory
    frames, exifs = get_frames_from_zip(file_input=file, resize_ratio_factor=0.5)
    frames = frames

    h, w = frames[0].shape[:2]
    output_folder = path.join(tempfile.gettempdir(), 'gradio-deva')
    print(f'{output_folder=}')
    os.makedirs(output_folder, exist_ok=True)
    vid_name = f'{hash(os.times())}'
    vid_path = path.join(output_folder, f'{vid_name}.mp4')
    print(f'{vid_path=}')

    output_folder_images = path.join(output_folder, vid_name)
    result_saver = ResultSaver(output_folder_images, None, dataset='gradio', object_manager=deva.object_manager)

    # process_frame_text(deva,
    #                    gd_model,
    #                    sam_model,
    #                    'null.png',
    #                    result_saver,
    #                    0,
    #                    image_np=frames[0])
    # flush_buffer(deva, result_saver)

    writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 5, (w, h))
    result_saver.writer = writer

    # only an estimate
    for ti, (frame, exif) in tqdm(enumerate(zip(frames, exifs)), total=len(frames)):
        process_frame_text(deva,
                           gd_model,
                           sam_model,
                           f'{ti}.png',
                           result_saver,
                           ti,
                           image_np=frame)
    flush_buffer(deva, result_saver)
    result_saver.end()
    writer.release()
    deva.clear_buffer()

    zip_out_path = vid_path.replace('.mp4', '')
    shutil.make_archive(zip_out_path, 'zip', output_folder_images)

    return (gr.File(zip_out_path + '.zip'),
            vid_path)


def demo_automatic(video: gr.Video, threshold: float, points_per_side: int, max_num_objects: int,
                   internal_resolution: int, detection_every: int, max_missed_detection: int,
                   sam_num_points: int, chunk_size: int, sam_variant: str, temporal_setting: str,
                   suppress_small_mask: bool):
    np.random.seed(42)
    torch.autograd.set_grad_enabled(False)
    parser = ArgumentParser()
    add_common_eval_args(parser)
    add_ext_eval_args(parser)
    add_auto_default_args(parser)
    deva_model, cfg, _ = get_model_and_config(parser)
    cfg['SAM_NUM_POINTS_PER_SIDE'] = int(points_per_side)
    cfg['SAM_NUM_POINTS_PER_BATCH'] = int(sam_num_points)
    cfg['enable_long_term_count_usage'] = True
    cfg['max_num_objects'] = int(max_num_objects)
    cfg['size'] = int(internal_resolution)
    cfg['SAM_PRED_IOU_THRESHOLD'] = threshold
    cfg['amp'] = True
    cfg['chunk_size'] = chunk_size
    cfg['detection_every'] = detection_every
    cfg['max_missed_detection_count'] = max_missed_detection
    cfg['sam_variant'] = sam_variant
    cfg['suppress_small_objects'] = suppress_small_mask
    cfg['temporal_setting'] = temporal_setting
    sam_model = get_sam_model(cfg, 'cuda')

    deva = DEVAInferenceCore(deva_model, config=cfg)
    deva.next_voting_frame = cfg['num_voting_frames'] - 1
    deva.enabled_long_id()

    print('Configuration:', cfg)

    vid_folder = path.join(tempfile.gettempdir(), 'gradio-deva')
    os.makedirs(vid_folder, exist_ok=True)
    vid_path = path.join(vid_folder, f'{hash(os.times())}.mp4')
    # obtain temporary directory
    result_saver = ResultSaver(path.join(vid_folder, 'output'), None, dataset='gradio', object_manager=deva.object_manager)
    writer_initizied = False

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ti = 0
    # only an estimate
    with torch.cuda.amp.autocast(enabled=cfg['amp']):
        with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret == True:
                    if not writer_initizied:
                        h, w = frame.shape[:2]
                        writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        writer_initizied = True
                        result_saver.writer = writer

                    process_frame_auto(deva,
                                       sam_model,
                                       'null.png',
                                       result_saver,
                                       ti,
                                       image_np=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    ti += 1
                    pbar.update(1)
                else:
                    break
        flush_buffer(deva, result_saver)
    result_saver.end()
    writer.release()
    cap.release()
    deva.clear_buffer()
    return vid_path


text_demo_tab = gr.Interface(
    fn=demo_with_text,
    inputs=[
        gr.File(),
        gr.Text(label='Prompt (class names delimited by full stops)'),
        gr.Slider(minimum=0.01, maximum=0.99, value=0.35, label='Threshold'),
        gr.Slider(
            minimum=1,
            maximum=100,
            value=10,
            label='Max num. objects',
            step=1,
        ),
        gr.Slider(
            minimum=384,
            maximum=1080,
            value=480,
            label='Internal resolution',
            step=1,
        ),
        gr.Slider(
            minimum=3,
            maximum=100,
            value=5,
            label='Incorpate detection every [X] frames',
            step=1,
        ),
        gr.Slider(minimum=1,
                  maximum=1000,
                  value=10,
                  step=1,
                  label='Delete segment if undetected for [X] times'),
        gr.Slider(minimum=1,
                  maximum=256,
                  value=8,
                  step=1,
                  label='DEVA number of objects per batch (reduce to save memory)'),
        gr.Dropdown(choices=['mobile', 'original'],
                    label='SAM variant (mobile is faster but less accurate)',
                    value='original'),
        gr.Dropdown(choices=['semionline', 'online'],
                    label='Temporal setting (semionline is slower but less noisy)',
                    value='semionline'),
    ],
    outputs=["file", "playable_video"],
    examples=[],
    cache_examples=False,
    title='DEVA: Tracking Anything with Decoupled Video Segmentation (text-prompted)')

auto_demo_tab = gr.Interface(
    fn=demo_automatic,
    inputs=[
        gr.Video(),
        gr.Slider(minimum=0.01, maximum=0.99, value=0.88, label='IoU threshold'),
        gr.Slider(minimum=4, maximum=256, value=64, label='Num. points per side for SAM', step=1),
        gr.Slider(minimum=10,
                  maximum=1000,
                  value=200,
                  label='Max num. objects (reduce to save memory)',
                  step=1),
        gr.Slider(minimum=384, maximum=1080, value=480, label='Internal resolution', step=1),
        gr.Slider(
            minimum=3,
            maximum=100,
            value=5,
            label='Incorpate detection every [X] frames',
            step=1,
        ),
        gr.Slider(minimum=1,
                  maximum=1000,
                  value=5,
                  step=1,
                  label='Delete segment if unseen in [X] detections'),
        gr.Slider(minimum=1,
                  maximum=1024,
                  value=64,
                  step=1,
                  label='SAM number of points per batch (reduce to save memory)'),
        gr.Slider(minimum=1,
                  maximum=256,
                  value=8,
                  step=1,
                  label='DEVA number of objects per batch (reduce to save memory)'),
        gr.Dropdown(choices=['mobile', 'original'],
                    label='SAM variant (mobile is faster but less accurate)',
                    value='original'),
        gr.Dropdown(choices=['semionline', 'online'],
                    label='Temporal setting (semionline is slower but less noisy)',
                    value='semionline'),
        gr.Checkbox(label='Suppress small masks', value=False),
    ],
    outputs="playable_video",
    examples=[],
    cache_examples=False,
    title='DEVA: Tracking Anything with Decoupled Video Segmentation (automatic)')

if __name__ == "__main__":
    gr.TabbedInterface([text_demo_tab], ["Text prompt"]).launch()
