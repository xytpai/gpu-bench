#include <iostream>
#include <random>

#include "utils.h"
using namespace std;

void worker(int rank, HostBarrier &barrier) {
    for (int i = 0; i < 4; ++i) {
        {
            std::unique_lock<std::mutex> lock(barrier.mtx);
            std::cout << "Thread " << rank << " reached barrier " << barrier.gen() << "!\n";
        }
        barrier.wait();
    }
}

int main() {
    std::cout << "host barrier test ... \n";
    int nranks = 4;
    HostBarrier barrier(nranks);
    std::vector<std::thread> threads;
    for (int i = 0; i < nranks; ++i) {
        threads.emplace_back(worker, i, std::ref(barrier));
    }
    for (auto &t : threads) t.join();
}
