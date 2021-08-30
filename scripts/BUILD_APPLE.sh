LIBTORCH="https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.9.0.zip"

echo "building for apple"

# GET LIBTORCH
if [ ! -d "libtorch/" ]
then
echo "downloading libtorch"
curl $LIBTORCH -o libtorch.zip &> /dev/null
echo "unzipping libtorch"
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
cmake ../ -DCMAKE_PREFIX_PATH="$LIBTORCH" -DCMAKE_BUILD_TYPE=Release > /dev/null
echo "building project"
make > /dev/null
echo "moving max external"
rm -fr ../../externals
mv ../externals/ ../../