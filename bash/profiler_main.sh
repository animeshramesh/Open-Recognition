#!/bin/bash
python_script=main.py
tmp_dir=../tmp
prof_results=${tmp_dir}/main.cprof

python -m cProfile -s cumtime -o ${prof_results} ${python_script}

# visualize profiler, requires x11
#pyprof2calltree -k -i ${prof_results}
