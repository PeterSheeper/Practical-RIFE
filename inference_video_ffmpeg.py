import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import skvideo.io
from queue import Queue, Empty
from model.pytorch_msssim import ssim_matlab
from scene_cut_detector import SceneCutDetector
import subprocess

warnings.filterwarnings("ignore")

FRAMES_TO_SKIP = []

NUMBER_OF_WRITE_THREADS = 4
PNG_COMPRESSION = 3 # 0-min 9-max
JPEG_QUALITY = 100 # 0-min 100-max

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--video', dest='video', type=str, default=None)
parser.add_argument('--temp', dest='temp', type=str, default=None)
parser.add_argument('--output', dest='output', type=str, default=None)
parser.add_argument('--img', dest='img', type=str, default=None)
parser.add_argument('--ogv', dest='ogv', type=str, default=None, help="original video for scene cut detection")
parser.add_argument('--start_frame', dest='start_frame', type=int, default=0)
parser.add_argument('--end_frame', dest='end_frame', type=int, default=9999999999)
parser.add_argument('--scene_detect', dest='scene_detect', action='store_true', help='whether to detect scenes')
parser.add_argument('--fancy_blend', dest='fancy_blend', action='store_true', help='whether to blend frames')
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')
parser.add_argument('--fp16', dest='fp16', action='store_true', help='fp16 mode for faster and more lightweight inference on cards with Tensor Cores')
parser.add_argument('--UHD', dest='UHD', action='store_true', help='support 4k video')
parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='Try scale=0.5 for 4k video')
parser.add_argument('--skip', dest='skip', action='store_true', help='whether to remove static frames before processing')
parser.add_argument('--fps', dest='fps', type=int, default=None)
parser.add_argument('--max_fps', dest='max_fps', type=int, default=120)
parser.add_argument('--png', dest='png', action='store_true', help='whether to vid_out png format vid_outs')
parser.add_argument('--jpg', dest='jpg', action='store_true', help='whether to vid_out jpg format vid_outs')
parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='vid_out video extension')
parser.add_argument('--multi', dest='multi', type=int, default=2)

args = parser.parse_args()
assert (args.video is not None or args.img is not None)
assert (args.jpg != args.png or not args.jpg)
assert (args.temp is not None)
if args.UHD and args.scale == 1.0:
    args.scale = 0.5
assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if (args.fp16):
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

try:
    from train_log.RIFE_HDv3 import Model
except:
    print("Please download our model from model list")
model = Model()
if not hasattr(model, 'version'):
    model.version = 0
model.load_model(args.modelDir, -1)
print("Loaded 3.x/4.x HD model.")
model.eval()
model.device()

frame_counter = 0
lastframe = None
if not args.video is None:
    videoCapture = cv2.VideoCapture(args.video)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    videoCapture.release()
    args.fps = fps * args.multi
    videogen = skvideo.io.vreader(args.video)
    while frame_counter <= args.start_frame:
        lastframe = next(videogen)
        frame_counter += 1
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    video_path_wo_ext, ext = os.path.splitext(args.video)
    print('{}.{}, {} frames in total, {}FPS to {}FPS'.format(video_path_wo_ext, args.ext, tot_frame, fps, args.fps))
else:
    videogen = []
    for f in os.listdir(args.img):
        if 'png' in f:
            videogen.append(f)
    tot_frame = len(videogen)
    videogen.sort(key= lambda x:int(x[:-4]))
    lastframe = cv2.imread(os.path.join(args.img, videogen[0]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
    frame_counter += 1
    videogen = videogen[1:]
h, w, _ = lastframe.shape

if args.jpg:
    if not os.path.exists(args.temp):
        os.mkdir(args.temp)
    img_out_ext = 'jpg'
    img_out_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
elif args.png:
    if not os.path.exists(args.temp):
        os.mkdir(args.temp)
    img_out_ext = 'png'
    img_out_params = [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION]


def buffer_to_ffmpeg(user_args, write_buffer):
    output_fps = user_args.fps if user_args.fps < user_args.max_fps else user_args.max_fps

    ffmpeg_cmd = [
        'ffmpeg',
        '-y',                        # Overwrite output file if it exists
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}',
        '-pix_fmt', 'rgb24',
        '-r', f'{user_args.fps}',                  # Replace with your frame rate
        '-i', '-',                  # Read input from stdin
        # '-vf', f'crop={3840}:{2160}:{(w-3840)//2}:{0}',
        '-r', f'{output_fps}',
        '-pix_fmt', 'yuv420p',
        '-c:v', 'libx265',
        '-preset', 'medium',      # Choose a preset according to your need
        '-crf', '14',                # Adjust the CRF value for quality
        user_args.temp
    ]
    ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    while True:
        frame = write_buffer.get()
        if frame is None:
            break

        # yuv_frame = cv2.cvtColor(frame[1], cv2.COLOR_RGB2YUV_I420)

        # Write frame to ffmpeg stdin
        ffmpeg_process.stdin.write(frame[1].tobytes())

    # Close ffmpeg stdin to indicate end of input
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()


def clear_write_buffer(user_args, write_buffer):
    os.chdir(user_args.temp)
    while True:
        item = write_buffer.get()
        if item is None:
            break
        img = item[1]
        cnt = item[0]
        cv2.imwrite('{:0>9d}.{}'.format(cnt, img_out_ext), img[:, :, ::-1], img_out_params)


def build_read_buffer(user_args, read_buffer, videogen):
    try:
        frame_id = user_args.start_frame + 1
        for frame in videogen:
            if user_args.img is not None:
                frame = cv2.imread(os.path.join(user_args.img, frame), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
            elif user_args.end_frame <= frame_id:
                break
            read_buffer.put(frame)
            frame_id += 1
    except:
        pass
    read_buffer.put(None)


def make_inference(I0, I1, n):
    global model
    if model.version >= 3.9:
        res = []
        for i in range(n):
            res.append(model.inference(I0, I1, (i+1) * 1. / (n+1), args.scale))
        return res
    else:
        middle = model.inference(I0, I1, args.scale)
        if n == 1:
            return [middle]
        first_half = make_inference(I0, middle, n=n//2)
        second_half = make_inference(middle, I1, n=n//2)
        if n % 2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]


def pad_image(img):
    if (args.fp16):
        return F.pad(img, padding).half()
    else:
        return F.pad(img, padding)


detector = None
scene_cut_frames = []

detector = SceneCutDetector(args.ogv)
if args.scene_detect:
    detector.detect_scene_cuts()
    detector.user_correction()
    scene_cut_frames = detector.get_scene_cut_frames()

    print(f"Final Scene Cut Frames: {scene_cut_frames}")

tmp = max(128, int(128 / args.scale))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
padding = (0, pw - w, 0, ph - h)
# pbar = tqdm(total=tot_frame)

write_buffer = Queue(maxsize=80)
read_buffer = Queue(maxsize=20)
_thread.start_new_thread(build_read_buffer, (args, read_buffer, videogen))

if args.jpg or args.png:
    for i in range(NUMBER_OF_WRITE_THREADS):
        _thread.start_new_thread(clear_write_buffer, (args, write_buffer))
else:
    _thread.start_new_thread(buffer_to_ffmpeg, (args, write_buffer))

I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
I1 = pad_image(I1)

out_counter = 0
while True:
    frame = read_buffer.get()
    if frame is None:
        break
    I0 = I1
    I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = pad_image(I1)

    if frame_counter in FRAMES_TO_SKIP:
        for i in range(args.multi):
            write_buffer.put([out_counter, lastframe])
            out_counter += 1
        frame_counter += 1
        continue

    if frame_counter in scene_cut_frames:
        output = []
        if args.fancy_blend:
            step = 1 / args.multi
            alpha = 0
            for i in range(args.multi - 1):
                alpha += step
                beta = 1-alpha
                output.append(torch.from_numpy(np.transpose((cv2.addWeighted(frame[:, :, ::-1], alpha, lastframe[:, :, ::-1], beta, 0)[:, :, ::-1].copy()), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
        else:
            for i in range(args.multi - 1):
                output.append(I0)
    else:
        output = make_inference(I0, I1, args.multi-1)

    write_buffer.put([out_counter, lastframe])
    out_counter += 1
    for mid in output:
        mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
        write_buffer.put([out_counter, mid[:h, :w]])
        out_counter += 1
    # pbar.update(1)
    lastframe = frame
    frame_counter += 1

write_buffer.put([out_counter, lastframe])
write_buffer.put(None)

import time
while(not write_buffer.empty()):
    time.sleep(0.1)
# pbar.close()