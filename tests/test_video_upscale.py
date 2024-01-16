import pytest
from pathlib import Path
from video_upscaler.main import upscale_video

def test_upscale():
    input_video = 'tests/data/heidi.mp4'
    expected_output = 'tests/data/heidi_HD.mp4'
    upscale_video(input_video, expected_output)
    assert Path(expected_output).exists() 

