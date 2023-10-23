# install aPyOpenGL package

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="env-apyopengl"


pushd "${ROOT_DIR}"
    # initial setup
    apt-get update
    apt-get install libxml2
    apt-get install libxml2-dev

    # !!! this removes existing version of the env
    conda remove -y -n "${ENV_NAME}" --all

    # create and activate conda env
    conda create -y -n "${ENV_NAME}" python=3.9
    CONDA_DIR="$(conda info --base)"
    source "${CONDA_DIR}/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
    if [ $? -ne 0 ]; then
        echo "*** Failed to activate env"
        exit 1
    fi

    # make fbxsdk directory
    mkdir -p python-fbxsdk

    pushd python-fbxsdk
        # download fbx sdk
        SIP_TARGZ=sip-4.19.3.tar.gz
        FBXSDK_TARGZ=fbx202032_fbxsdk_linux.tar.gz
        BINDINGS_TARGZ=fbx202032_fbxpythonbindings_linux.tar.gz
        
        wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1qPvq_23_7jmxMM1gWEQPSCgSyvB6Q7CE' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1qPvq_23_7jmxMM1gWEQPSCgSyvB6Q7CE" -O $SIP_TARGZ && rm -rf ~/cookies.txt
        wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1FkBduIsqSYbV0mmgk-CYXuyvh3nRsQNg' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1FkBduIsqSYbV0mmgk-CYXuyvh3nRsQNg" -O $FBXSDK_TARGZ && rm -rf ~/cookies.txt
        wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1XrZeV8Zz2AYz9jvbPU81vQ9agofVKcRy' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1XrZeV8Zz2AYz9jvbPU81vQ9agofVKcRy" -O $BINDINGS_TARGZ && rm -rf ~/cookies.txt

        # extract tar.gz files
        mkdir -p sdk
        mkdir -p bindings

        tar -xzf $SIP_TARGZ
        tar -xzf $FBXSDK_TARGZ -C sdk
        tar -xzf $BINDINGS_TARGZ -C bindings

        # install sdk
        pushd sdk
            chmod ugo+x fbx202032_fbxsdk_linux
            yes "yes" | ./fbx202032_fbxsdk_linux
            export FBXSDK_ROOT=$(pwd)
        popd

        # install sip
        pushd sip-4.19.3
            python configure.py
            sed -i "s/libs.extend(self.optional_list(\"LIBS\"))/libs = self.optional_list(\"LIBS\").extend(libs)" "sipconfig.py"
            python configure.py
            make
            make install
            export SIP_ROOT=$(pwd)
        popd

        # install bindings
        pushd bindings
            chmod ugo+x fbx202032_fbxpythonbindings_linux
            yes "yes" | ./fbx202032_fbxpythonbindings_linux

            python PythonBindings.py Python3_x64
            pushd build/Python39_x64
                pattern="-lz -lxml2"
                sed -i "s/\(LIBS = .*\) $pattern/\1/g" "Makefile"
                sed -i "s/\(LIBS = .*\)/\1 $pattern/" "Makefile"
                make clean
                make
                make install
            popd
        popd

        # move files
        ENV_PATH=$(conda info --envs | grep -E "^$ENV_NAME\s+" | awk '{print $NF}')
        cp -r ./bindings/build/Python39_x64/fbx.so $ENV_PATH/lib/python3.9/site-packages/
        cp -r ./bindings/build/Distrib/site-packages/fbx/FbxCommon.py $ENV_PATH/lib/python3.9/site-packages/
    popd

    # install additional dependencies
    pip install -r requirements.txt
    imageio_download_bin freeimage
    pip install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

    # remove fbxsdk directory
    rm -rf python-fbxsdk
popd