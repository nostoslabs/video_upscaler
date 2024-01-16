import sys
import torch
from PIL import Image
import numpy as np
import av
from RealESRGAN import RealESRGAN
from argparse import ArgumentParser
from pathlib import Path

def upscale_frame(frame: av.VideoFrame, upscaler: RealESRGAN) -> av.VideoFrame:
    """Upscale a single frame."""
    img = frame.to_image().convert('RGB')
    sr_img = upscaler.predict(img)
    return av.VideoFrame.from_image(sr_img)

def upscale_video(input_path: Path, output_path: Path, scale: int = 4):
    """Upscale video frames and reassemble the video."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA.")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS.")
    else:
        device = torch.device('cpu')
        print("Using CPU.")
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = mps_device = torch.device("mps")
    #device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    upscaler = RealESRGAN(device, scale)
    upscaler.load_weights('weights/RealESRGAN_x4.pth', download=True)

    input_container = av.open(str(input_path))
    output_container = av.open(str(output_path), 'w')

    stream = input_container.streams.video[0]
    output_stream = output_container.add_stream('mpeg4', rate=stream.average_rate)
    output_stream.width = stream.width * scale
    output_stream.height = stream.height * scale
    output_stream.pix_fmt = 'yuv420p'

    for frame in input_container.decode(stream):
        sr_frame = upscale_frame(frame, upscaler)
        packet = output_stream.encode(sr_frame)
        output_container.mux(packet)

    # Flush and close the containers
    for packet in output_stream.encode():
        output_container.mux(packet)

    input_container.close()
    output_container.close()

def main():
    parser = ArgumentParser(description="Upscale video files.")
    parser.add_argument('input_file', type=Path, help='Path to the input video file.')
    parser.add_argument('output_file', type=Path, help='Path for the output upscaled video file.')
    args = parser.parse_args()

    upscale_video(args.input_file, args.output_file)

if __name__ == "__main__":
    main()

