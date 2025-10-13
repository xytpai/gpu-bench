#include <iostream>
#include <random>
#include <vector>
#include <tuple>
#include <chrono>

#include "utils.h"
#include "comm.h"
using namespace std;

void runbench(size_t buffer_size, int nranks, size_t nblocks_per_rank) {
}

int main() {
    std::cout << "1GB communication barrier test ... \n";
    int nranks = enable_p2p();
    size_t buffer_size = (size_t)1024 * 1024 * 1024;
    runbench(buffer_size, nranks, 256);
}
