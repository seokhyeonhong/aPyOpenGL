# Python Framework for Motion Data
```aPyOpenGL``` is a Python version framework of [aOpenGL](https://github.com/ltepenguin/aOpenGL) for motion data processing and visualization.
Our framework is based on right-hand coordinate with y-axis as the up vector.
<img src="teaser.gif">

# Installation
### Linux
For Linux users, we provide a shell script that creates a conda environment in ```install.sh```. You can modify the environment name by changing ```ENV_NAME``` in the script, which is set to ```env-apyopengl``` by default.
```
bash install.sh
```

### Windows
For the visualization modules, install necessary modules first.
```
pip install -r requirements.txt
imageio_download_bin freeimage
```
Also, visit this perfect guide to install FBX SDK on your computer: [Link for Windows](https://www.ralphminderhoud.com/blog/build-fbx-python-sdk-for-windows/)

# How to use
```aPyOpenGL``` has four main modules ```agl```, ```kin```, ```transforms```, and ```ops```, and one additional auxiliary module ```utils```. Example codes are in [examples](examples/) and you can run the code you want through:
```
python examples/{script_to_run}.py
```

## Set Up the Environment Variable
If you add the path of this framework to the global environment variable, you can use this framework anywhere in your local computer.

### Linux
Add this line to ```~/.bashrc```:
```
export PYTHONPATH=$PYTHONPATH:{path/to/aPyOpenGL}
```
and then execute this:
```
source ~/.bashrc
```

### Windows
Add the path of the cloned repository to the environment variable PYTHONPATH. If you don't know how, please refer to [this](https://stackoverflow.com/questions/3701646/how-to-add-to-the-pythonpath-in-windows-so-it-finds-my-modules-packages).

### Commands
* F1: Render the scene in GL_FILL mode.
* F2: Render the scene in GL_LINE mode.
* F5: Capture the screen in image and save in ```captures/yyyy-mm-dd/images```.
* F6: Capture the screen in video if entered once, and save in ```captures/yyyy-mm-dd/videos``` if entered again.
* Alt + Left Mouse: Tumble tool for the camera.
* Alt + Middle Mouse: Track tool for the camera.
* Alt + Mouse Scroll: Dolly tool for the camera.
* Mouse Scroll: Zoom tool for the camera.
* A: Switch the visualization of the axis.
* G: Switch the visualization of the grid.
* F: Switch the visualization of the render fps text.
* Left / Right arrow: Move 1 second to the past / future.

Additionally, you can add your own custom commands.
You can find the examples in the codes in [examples](examples/).

## Motion
We provide BVH parser for motion data and FBX parser for both motion and mesh data. Motion data in this framework is basically structred by hierarchy of Joint, Skeleton, Pose, and Motion, and you can see the structure [here](aPyOpenGL/agl/motion).


<!-- ## Learning
```learning``` module provides several neural network models in PyTorch. New models will be updated continuously. -->

## Transforms
```transforms``` provides several operations for transformation in both numpy and pytorch.
Modules that start with ```n_``` indicates that it's for numpy ndarray, and ```t_``` indicates pytorch tensor.
<!-- ```ops``` provides several operations for dealing with motion data. Both NumPy ndarray and PyTorch Tensor are supported.

* ```mathops.py``` provides general mathematical operations.
* ```motionops.py``` provides manipulation functions for motion data (e.g. forward kinematics).
* ```rotation.py``` provides rotation operations and conversions. -->

## Utils
```utils``` provides several utility functions like multiprocessing.

# More to Come
We are planning to support motion manipulation functions, like ```kin``` namespace in [aOpenGL](https://github.com/ltepenguin/aOpenGL). This will be updated soon!

<!-- FBX C++ SDK & FBX Python Bindings & SIP
### How to install SIP and Python Bindings

NOTE 1: sip version 4.19.25 is not supported. Try sip version [4.19.3](https://sourceforge.net/projects/pyqt/) or earlier.

NOTE 2: No white spaces in all paths are allowed. Move and rename the path where those files without spaces.


### Setup the environment variable
We need to setup 3 environment variables, and here's the example:
* SIP_ROOT `C:\dev\sip-4.19.3`
* FBXSDK_ROOT `C:\dev\FBX\FBX_SDK\2020.2.1`
* FBXSDK_LIBS_64_FOLDER `C:\dev\FBX\FBX_SDK\2020.2.1\lib\vs2019\x64\release`

Then compile the scripts as follows:
```
cd SIP_ROOT
python configure.py
"C:\Qt\~~~~"
"C:\...\vcvarsall.bat" (If it doesn't work, try "~~~\vcvars64.bat")
nmake
nmake install
cd PythonBindings
python PythonBindings.py Python3_x64 buildsip
```

Then path_to_binding/version/build/Distrib/site-packages/fbx will be generated.
It would contain 3 files (fbx.pyd, FbxCommon.py, fbxsip.pyd), and you should move them to path_to_python/site_packages.

If interpreting PythonBindings.py fails, you can try changing the variable vcCompiler and vsCompiler to what you are using. -->

# Publications
Published papers developed on top of this framework are as follows:

Geometry-Aware Retargeting for Two-Skinned Characters Interaction [Jang et al. SIGGRAPH Asia 2024]
```
@article{jang2024geometry,
  title={Geometry-Aware Retargeting for Two-Skinned Characters Interaction},
  author={Jang, Inseo and Choi, Soojin and Hong, Seokhyeon and Kim, Chaelin and Noh, Junyong},
  journal={ACM Transactions on Graphics (TOG)},
  volume={43},
  number={6},
  pages={1--17},
  year={2024},
  publisher={ACM New York, NY, USA}
}
```
[Long-term Motion In-Betweening via Keyframe Prediction](https://github.com/seokhyeonhong/long-mib) [Hong et al. SCA2024]
```
@inproceedings{hong2024long,
  title={Long-Term Motion In-Betweening via Keyframe Prediction},
  author={Hong, Seokhyeon and Kim, Haemin and Cho, Kyungmin and Noh, Junyong},
  booktitle={Proceedings of the ACM SIGGRAPH/Eurographics Symposium on Computer Animation},
  pages={1--12},
  year={2024}
}
```

# Acknowledgements
The overall structure of the rendering modules is inspired by
[aOpenGL](https://github.com/ltepenguin/aOpenGL)
and [LearnOpenGL](https://learnopengl.com/)

Data processing, operation functions, and utility functions are inspired by
[fairmotion](https://github.com/facebookresearch/fairmotion),
[pytorch3d](https://github.com/facebookresearch/pytorch3d),
[PFNN](https://github.com/sreyafrancis/PFNN),
and [LaFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) repositories