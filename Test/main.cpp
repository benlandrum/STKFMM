#include "Test.hpp"

#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    Config config;
    config.parse(argc, argv);
    config.print();

    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    Point point;
    Input input;
    Result pResult, verifyResult, convResult, transResult;
    Timing timing;
    std::vector<Record> history;

    genPoint(config, point, 3);
    genSrcValue(config, point, input);

    printf("src value generated\n");

    if (config.verify) {
        runSimpleKernel(point, input, verifyResult);
        dumpValue("verify", point, input, verifyResult);
    }

    if (config.convergence) {
        runFMM(config, maxP, point, input, convResult, timing);
        dumpValue("P" + std::to_string(maxP), point, input, verifyResult);
        for (auto &t : timing) {
            Record record;
            record.kernel = t.first;
            record.multOrder = maxP;
            record.treeTime = t.second.first;
            record.runTime = t.second.second;
        }
    }

    for (int p = 6; p < maxP; p += 2) {
        pResult.clear();
        transResult.clear();
        timing.clear();

        runFMM(config, p, point, input, pResult, timing);
        dumpValue("p" + std::to_string(p), point, input, pResult);
        printf_rank0("*********Testing order p = %d*********\n", p);

        if (config.pbc) {
            Point trans_point = point;
            Timing transTiming;
            translatePoint(config, trans_point);
            runFMM(config, p, trans_point, input, transResult, transTiming);
            dumpValue("trans_p" + std::to_string(p), trans_point, input, transResult);
        }

        printf_rank0("------------------------------------\n");

        appendHistory(history, p, timing, pResult, verifyResult, convResult, transResult);
    }

    MPI_Finalize();
    return 0;
}
