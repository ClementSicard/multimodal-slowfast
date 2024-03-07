# Add 'asf' folder to the path
import sys

sys.path.append("asf")
sys.path.append("vsf")

from asf.audio_slowfast.tools.run_net import load_config as load_audio_config
from vsf.tools.run_net import load_config as load_video_config


def get_audio_config(path: str):
    return load_audio_config()
