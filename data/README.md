# Data

Video files are not tracked in git.

## Primary Benchmark Video

- **Source:** Mixkit "Dashboard of a car"
- **Resolution:** 1920x1080
- **Duration:** ~18 seconds, 24 FPS
- **Expected path:** `data/clip.mp4`

## Setup in Colab

Mount Google Drive and copy the video:

```python
from google.colab import drive
drive.mount('/content/drive')

import shutil
shutil.copy('/content/drive/MyDrive/inference_project/clip.mp4', 'data/clip.mp4')
```
