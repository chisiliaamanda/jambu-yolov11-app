from pathlib import Path
import sys

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
ROOT = FILE.parent

# Add the root path to the sys.path list if it is not already there
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
WEBCAM = 'Webcam'
RTSP = 'RTSP'
YOUTUBE = 'YouTube'

SOURCES_LIST = [IMAGE, VIDEO, WEBCAM, RTSP, YOUTUBE]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'jambu1.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'detectedimage1.png'

# Videos config
VIDEOS_DIR = ROOT / 'videos'
VIDEOS_DICT = {
    'video_1': VIDEOS_DIR / 'video_1.mp4',
    'video_2': VIDEOS_DIR / 'video_2.mp4',
    'video_3': VIDEOS_DIR / 'video_3.mp4',
}

# ML Model config
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'best.pt'

# Webcam path
WEBCAM_PATH = 0
