executions=1
iterations=100
warmups=10
alpha=0.1
beta=1.0

# Using make
# make clean
# make
program=parallel/my_thing

# Using cmake
# mkdir build && cd build
# cmake -DCMAKE_BUILD_TYPE=Release ..
# make
# cd ..
# program=build/main

type=gemm
# type=gemv

split=0 # 0 = tensor cores, 1 = cuda cores, 2 = 50/50 split

size=1024
# size=2048
# size=4096
# size=8192

echo $size $warmups $iterations $split
for i in `seq 1 $executions`; do ./$program $type $size $warmups $iterations $split $alpha $beta; done;
done;


# in another terminal run: sh smi-stuff.sh
# sh power_usage_test.sh >> power_data.txt
