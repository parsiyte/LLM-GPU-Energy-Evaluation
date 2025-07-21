#!/bin/bash
apt -y install  gcc-11 g++-11
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11
cd /data/lm-evaluation-harness/split/profiling_injection && ./build.sh
