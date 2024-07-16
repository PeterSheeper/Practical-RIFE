import argparse
import warnings
import os
import cv2
import subprocess
from math import ceil

warnings.filterwarnings("ignore")

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
if not args.video:
    exit()
assert (args.temp is not None and args.output is not None)
if args.ogv is None:
    args.ogv = args.video

videoCapture = cv2.VideoCapture(args.video)
fps = videoCapture.get(cv2.CAP_PROP_FPS)
videoCapture.release()

base_name = os.path.basename(args.video)
file_name, _ = os.path.splitext(base_name)
file_name += ' ' + str(ceil(fps)) + 'fpsUP.' + 'mp4'

output_video = os.path.join(args.output, file_name)

command = [
    'ffmpeg',
    '-i', args.temp,
    '-i', args.ogv,
    '-c:v', 'copy',
    '-c:a', 'copy',
    '-map', '0:v:0',
    '-map', '1:a:0',
    '-shortest',
    output_video
]

try:
    subprocess.run(command, check=True)
    os.remove(args.temp)
    print(f"Successfully merged video from {args.temp} and audio from {args.ogv} into {output_video}")
except subprocess.CalledProcessError as e:
    print(f"Error occurred: {e}")
