# roop-unleashed

[Changelog](#changelog) • [Usage](#usage) • [Wiki](https://github.com/C0untFloyd/roop-unleashed/wiki)


Uncensored Deepfakes for images and videos without training and an easy-to-use GUI.


![Screen](https://github.com/C0untFloyd/roop-unleashed/assets/131583554/6ee6860d-efbe-4337-8c62-a67598863637)

### Features

- Platform-independant Browser GUI
- Selection of multiple input/output faces in one go
- Many different swapping modes, first detected, face selections, by gender
- Batch processing of images/videos
- Masking of face occluders using text prompts or automatically
- Optional Face Upscaler/Restoration using different enhancers
- Preview swapping from different video frames
- Live Fake Cam using your webcam
- Extras Tab for cutting videos etc.
- Settings - storing configuration for next session
- Theme Support

and lots more...


## Disclaimer

This project is for technical and academic use only.
Users of this software are expected to use this software responsibly while abiding the local law. If a face of a real person is being used, users are suggested to get consent from the concerned person and clearly mention that it is a deepfake when posting content online. Developers of this software will not be responsible for actions of end-users.
**Please do not apply it to illegal and unethical scenarios.**

In the event of violation of the legal and ethical requirements of the user's country or region, this code repository is exempt from liability

### Installation

Please refer to the [wiki](https://github.com/C0untFloyd/roop-unleashed/wiki).

#### macOS Installation
Simply run the following command. It will check and install all dependencies if necessary.

`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/PJF16/roop-unleashed/master/installer/macOSinstaller.sh)`



### Usage

- Windows: run the `windows_run.bat` from the Installer.
- Linux: `python run.py`
- macOS: `sh runMacOS.sh`
- Dockerfile - `docker build -t roop-unleashed .`

<a target="_blank" href="https://colab.research.google.com/github/C0untFloyd/roop-unleashed/blob/main/roop-unleashed.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
  

Additional commandline arguments are currently unsupported and settings should be done via the UI.

> Note: When you run this program for the first time, it will download some models roughly ~2Gb in size.




### Changelog

**28.9.2024** v4.3.1

- Bugfix: Several possible memory leaks
- Added different output modes, e.g. to virtual cam stream
- New swapping mode "All input faces"
- Average total fps displayed and setting for autorun


**16.9.2024** v4.2.8

- Bugfix: Starting roop-unleashed without NVIDIA gpu but cuda option enabled
- Bugfix: Target Faces couldn't be moved left/right
- Bugfix: Enhancement and upscaling working again in virtual cam
- Corrupt videos caught when adding to target files, displaying warning msg
- Source Files Component cleared after face detection to release temp files
- Added masking and mouth restore options to virtual cam


**9.9.2024** v4.2.3

- Hotfix for gradio pydantic issue with fastapi
- Upgraded to Gradio 4.43 hoping it will fix remaining issues
- Added new action when no face detected -> use last swapped
- Specified image format for image controls - opening new tabs on preview images possible again!
- Hardcoded image output format for livecam to jpeg - might be faster than previous webp
- Chain events to be only executed if previous was a success


**5.9.2024** v4.2.0

- Added ability to move input & target faces order
- New CLI Arguments override settings
- Small UI changes to faceswapping tab
- Added mask option and code for restoration of original mouth area
- Updated gradio to v4.42.0
- Added CLI Arguments --server_share and --cuda_device_id
- Added webp image support


**15.07.2024** v4.1.1

- Bugfix: Post-processing after swapping


**14.07.2024** v4.1.0

- Added subsample upscaling to increase swap resolution
- Upgraded gradio


**12.05.2024** v4.0.0

- Bugfix: Unnecessary init every frame in live-cam
- Bugfix: Installer downloading insightface package each run
- Added xseg masking to live-cam
- Added realesrganx2 to frame processors
- Upgraded some requirements
- Added subtypes and different model support to frame processors
- Allow frame processors to change resolutions of videos
- Different OpenCV Cap for MacOS Virtual Cam
- Added complete frame processing to extras tab
- Colorize, upscale and misc filters added


**22.04.2024** v3.9.0

- Bugfix: Face detection bounding box corrupt values at weird angles
- Rewrote mask previewing to work with every model
- Switching mask engines toggles text interactivity
- Clearing target files, resets face selection dropdown
- Massive rewrite of swapping architecture, needed for xseg implementation
- Added DFL Xseg Support for partial face occlusion
- Face masking only runs when there is a face detected
- Removed unnecessary toggle checkbox for text masking


**22.03.2024** v3.6.5

- Bugfix: Installer pulling latest update on first installation
- Bugfix: Regression issue, blurring/erosion missing from face swap
- Exposed erosion and blur amounts to UI
- Using same values for manual masking too


**20.03.2024** v3.6.3

- Bugfix: Workaround for Gradio Slider Change Bug
- Bugfix: CSS Styling to fix Gradio Image Height Bug
- Made face swapping mask offsets resolution independant
- Show offset mask as overlay
- Changed layout for masking


**18.03.2024** v3.6.0

- Updated to Gradio 4.21.0 - requiring many changes under the hood
- New manual masking (draw the mask yourself)
- Extras Tab, streamlined cutting/joining videos
- Re-added face selection by gender (on-demand loading, default turned off)
- Removed unnecessary activate live-cam option
- Added time info to preview frame and changed frame slider event to allow faster changes


**10.03.2024** v3.5.5

- Bugfix: Installer Path Env
- Bugfix: file attributes
- Video processing checks for presence of ffmpeg and displays warning if not found
- Removed gender + age detection to speed up processing. Option removed from UI
- Replaced restoreformer with restoreformer++
- Live Cam recoded to run separate from virtual cam and without blocking controls
- Swapping with only 1 target face allows selecting from several input faces



**08.01.2024** v3.5.0

- Bugfix: wrong access options when creating folders
- New auto rotation of horizontal faces, fixing bad landmark positions (expanded on ![PR 364](https://github.com/C0untFloyd/roop-unleashed/pull/364))
- Simple VR Option for stereo Images/Movies, best used in selected face mode
- Added RestoreFormer Enhancer - https://github.com/wzhouxiff/RestoreFormer
- Bumped up package versions for onnx/Torch etc.   


**16.10.2023** v3.3.4

**11.8.2023** v2.7.0

Initial Gradio Version - old TkInter Version now deprecated

- Re-added unified padding to face enhancers
- Fixed DMDNet for all resolutions
- Selecting target face now automatically switches swapping mode to selected
- GPU providers are correctly set using the GUI (needs restart currently)
- Local output folder can be opened from page
- Unfinished extras functions disabled for now
- Installer checks out specific commit, allowing to go back to first install
- Updated readme for new gradio version
- Updated Colab


# Acknowledgements

Lots of ideas, code or pre-trained models borrowed from the following projects:

https://github.com/deepinsight/insightface<br />
https://github.com/s0md3v/roop<br />
https://github.com/AUTOMATIC1111/stable-diffusion-webui<br /> 
https://github.com/Hillobar/Rope<br />
https://github.com/TencentARC/GFPGAN<br />   
https://github.com/kadirnar/codeformer-pip<br />
https://github.com/csxmli2016/DMDNet<br />
https://github.com/glucauze/sd-webui-faceswaplab<br />
https://github.com/ykk648/face_power<br />

<br />
<br />
Thanks to all developers!

