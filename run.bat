@echo off
set inf=99999999999999999

set INPUT_VIDEO=D:\example.mp4
set ORIGINAL_VIDEO=%INPUT_VIDEO%
set OUT_FOLDER=X:\out_folder
set TEMP_FILE=E:\temp_interp_video.mp4
set INTERPOLATION=4
set MAX_FPS=120
set START_FRAME=0
set END_FRAME=%inf%


python3.11 inference_video_ffmpeg.py --multi=%INTERPOLATION% --video="%INPUT_VIDEO%" --output="%OUT_FOLDER%" --temp="%TEMP_FILE%" --ogv="%ORIGINAL_VIDEO%" --max_fps=%MAX_FPS% --start_frame=%START_FRAME% --end_frame=%END_FRAME% --fancy_blend --scene_detect 
pause
python3.11 merge_audio_video.py --video="%INPUT_VIDEO%" --output="%OUT_FOLDER%" --temp="%TEMP_FILE%" --ogv="%ORIGINAL_VIDEO%"