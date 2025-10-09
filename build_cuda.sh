nvcc -x cu -O3 $2 --std=c++20 -arch sm_80 -Iutils $1 -o a.out
