#!/bin/bash --norc

cmake --build build --target install \
    && cp \
	   /opt/STKFMM/build/Test/TestFMM.X \
	   /opt/STKFMM/build/M2L/M2LLaplace \
	   /opt/STKFMM/build/M2L/M2LStokesPVel \
	   /opt/STKFMM/build/M2L/M2LStokeslet \
	   /usr/local/bin \
    && cp /opt/STKFMM/Python/timer.py /usr/local/lib64/python
