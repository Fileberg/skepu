executions=11
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

for size in 512 1024 2048 4096 8192;
do
for split in 0 1 -128 -32 2 32 128;
do
echo $size $warmups $iterations $split
for i in `seq 1 $executions`; do ./$program $type $size $warmups $iterations $split $alpha $beta; done;
done;
done;


# sh performace_test.sh >> performance_data.txt
