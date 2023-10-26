# install a conda environment with aPyOpenGL framework

# initialize environment variables
ENV_NAME=""

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --name)
        ENV_NAME="$2"
        echo "ENV_NAME: $ENV_NAME"
        shift # remove --name
        shift # remove its value
        ;;
        *)
        echo "Unknown option: $key"
        exit 1
        ;;
    esac
done

# check if --name is provided
if [[ -z "$ENV_NAME" ]]; then
    echo "The --name argument is mandatory!"
    exit 1
fi

# check if environment with the same name exists
conda info --envs | grep -E "^$ENV_NAME\s+" > /dev/null
if [ $? -eq 0 ]; then
    echo "Environment with the name $ENV_NAME already exists!"

    # ask if the user wants to remove it
    read -p "Do you want to remove it? [y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda remove -y -n "${ENV_NAME}" --all
    else
        echo "Aborting..."
        exit 1
    fi
fi

# ----- main ----- #
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pushd "${ROOT_DIR}"
    # initial setup
    apt-get install libxml2
    apt-get install libxml2-dev

    # # create and activate conda env
    conda create -y -n "${ENV_NAME}" python=3.8.10
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
        FBXSDK_TARGZ=fbx202001_fbxsdk_linux.tar.gz
        BINDINGS_TARGZ=fbx202001_fbxpythonbindings_linux.tar.gz

        wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1qPvq_23_7jmxMM1gWEQPSCgSyvB6Q7CE' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1qPvq_23_7jmxMM1gWEQPSCgSyvB6Q7CE" -O $SIP_TARGZ && rm -rf ~/cookies.txt
        wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Kn8vH2QMkfaCUM6j5gNDUVwawNiMvq4c' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Kn8vH2QMkfaCUM6j5gNDUVwawNiMvq4c" -O $FBXSDK_TARGZ && rm -rf ~/cookies.txt
        wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1r4IcLf6nj10GjEgcDrIvrZ8_0I9XU-hG' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1r4IcLf6nj10GjEgcDrIvrZ8_0I9XU-hG" -O $BINDINGS_TARGZ && rm -rf ~/cookies.txt

        # extract tar.gz files
        mkdir -p sdk
        mkdir -p bindings

        tar -xzf $SIP_TARGZ
        tar -xzf $FBXSDK_TARGZ -C sdk
        tar -xzf $BINDINGS_TARGZ -C bindings

        # install sdk
        pushd sdk
            chmod ugo+x fbx202001_fbxsdk_linux
            yes "yes" | ./fbx202001_fbxsdk_linux $(pwd)
            export FBXSDK_ROOT=$(pwd)
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$FBXSDK_ROOT/lib/gcc/x64/release
        popd

        # install sip
        pushd sip-4.19.3
            python configure.py
            sed -i "s/libs.extend(self.optional_list(\"LIBS\"))/libs = self.optional_list(\"LIBS\").extend(libs)" "sipconfig.py"
            python configure.py
            make -j8
            make install
            export SIP_ROOT=$(pwd)
        popd

        # install bindings
        pushd bindings
            chmod ugo+x fbx202001_fbxpythonbindings_linux
            yes "yes" | ./fbx202001_fbxpythonbindings_linux $(pwd)

            python PythonBindings.py Python3_x64
            pushd build/Python38_x64
                pattern="-lz -lxml2"
                sed -i "s/\(LIBS = .*\) $pattern/\1/g" "Makefile"
                sed -i "s/\(LIBS = .*\)/\1 $pattern/" "Makefile"
                make clean
                make -j8
                make install
            popd
        popd

        # move files
        ENV_PATH=$(conda info --envs | grep -E "^$ENV_NAME\s+" | awk '{print $NF}')
        rm $ENV_PATH/lib/python3.8/site-packages/fbx.so
        cp -r ./bindings/build/Python38_x64/fbx.so $ENV_PATH/lib/python3.8/site-packages
        cp -r ./bindings/build/Distrib/site-packages/fbx/FbxCommon.py $ENV_PATH/lib/python3.8/site-packages
    popd

    # install additional dependencies
    pip install -r requirements.txt
    imageio_download_bin freeimage
    pip install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

    # remove fbxsdk directory
    rm -rf python-fbxsdk

    # develop package
    conda develop $(pwd)
popd