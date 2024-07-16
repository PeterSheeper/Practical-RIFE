import subprocess
import json

with open('queue.json', 'r') as file:
    json_data = json.load(file)


def create_interp_command(video, temp_file, max_fps_val):
    video_path = f"--video='{video['video_path']}'"
    multi_factor = f"--multi={video['multi_factor']}"
    max_fps = f"--max_fps={max_fps_val}"

    interp_command = ['python3.11.exe', 'inference_video_ffmpeg.py', video_path, multi_factor, temp_file, max_fps]

    if video['scene_detect']:
        interp_command.append("--scene_detect")

    if 'start_frame' in video:
        interp_command.append(f"--start_frame={video['start_frame']}")
    if 'end_frame' in video:
        interp_command.append(f"--end_frame={video['end_frame']}")
    if 'original_video' in video:
        interp_command.append(f"--ogv='{video['original_video']}'")

    return interp_command


def create_merge_command(video, output_folder, temp_file):
    video_path = f"--video='{video['video_path']}'"
    output_path = f"--output='{output_folder}'"

    merge_command = ['python3.11.exe', 'merge_audio_video.py', video_path, output_path, temp_file]

    if 'original_video' in video:
        merge_command.append(f"--ogv='{video['original_video']}'")

    return merge_command


def run_command(command):
    result = subprocess.run(command, capture_output=True, text=True)

    # Print the output and error (if any) for each command
    print(f"Running command: {' '.join(command)}")
    print("Output:", result.stdout)
    print("Error:", result.stderr, '\n')


max_fps_val = json_data['max_fps']
output_folder = json_data['output_folder']
temp_folder = json_data['temp_folder']

for id, video in enumerate(json_data['videos']):
    temp_filename = "\\interp_temp_" + str(id) + ".mp4"
    temp_file = f"--temp='{temp_folder + temp_filename}'"

    interp_command = create_interp_command(video, temp_file, max_fps_val)
    run_command(interp_command)

    if 'start_frame' in video and video['start_frame'] > 0:
        continue

    merge_command = create_merge_command(video, output_folder, temp_file)
    run_command(merge_command)
