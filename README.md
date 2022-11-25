// TODO: Update and organize the contents

## Python FBX SDK Installation on Windows
First, it is highly recommended to use Python 3.3 since it is much easier to install Python FBX SDK than in others, but if you are using another version, you can try as follows.

Reference:
[Link1](https://www.ralphminderhoud.com/blog/build-fbx-python-sdk-for-windows/)
[Link2](https://forums.autodesk.com/t5/fbx-forum/cannot-manage-to-compile-python-fbx-2020-0-1-vs2017-with-python/td-p/9270853)
[Link3](https://stackoverflow.com/questions/32054021/how-to-install-sip-pyqt-on-windows-7)

FBX C++ SDK & FBX Python Bindings & SIP
### How to install SIP and Python Bindings

NOTE: sip version 4.19.25 is not supported. Try sip version [4.19.3](https://sourceforge.net/projects/pyqt/) or earlier.

Environment variable
* SIP_ROOT={}
* FBXSDK_ROOT={}
* FBXSDK_LIBS_64_FOLDER={}

cd {SIP}
python configure.py
"C:\Qt\~~~~"
"C:\...\vcvarsall.bat" (If not working, try "~~~\vcvars64.bat")
nmake
nmake install

cd PythonBindings
python PythonBindings.py Python3_x64 buildsip

Then path_to_binding/version/build/Distrib/site-packages/fbx will be generated.
It would contain 3 files (fbx.pyd, FbxCommon.py, fbxsip.pyd), and you should move them to path_to_python/site_packages.

NOTE: No white spaces in all paths are allowed. Move and rename the path where those files without spaces.

If interpreting PythonBindings.py fails, you can try changing the variable vcCompiler and vsCompiler to what you are using.
