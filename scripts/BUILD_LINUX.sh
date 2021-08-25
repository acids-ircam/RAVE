LIBTORCH="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.9.0%2Bcpu.zip"

echo "building for linux"

# GET LIBTORCH
if [ ! -d "libtorch/" ]
then
echo "downloading libtorch"
curl $LIBTORCH -o libtorch.zip &> /dev/null
unzip libtorch.zip &> /dev/null
rm libtorch.zip
else
echo "libtorch found"
fi

# CONFIGURE PROJECT
echo "configuring project"
cd libtorch
LIBTORCH=$(pwd)
cd ../../realtime
mkdir -p build
cd build
cmake ../ -DCMAKE_PREFIX_PATH=$LIBTORCH &> /dev/null
echo "building project"
make &> /dev/null