| Version          |   LOC (Core) |   Median T_NP1 (s) |   Median T_NP=4 (s) |   Speedup (Medians) @NP=4 |
|:-----------------|-------------:|-------------------:|--------------------:|--------------------------:|
| V1 Serial        |          525 |              0.784 |             nan     |                   nan     |
| V2.2 ScatterHalo |          483 |              0.714 |               0.287 |                     2.489 |
| V3 CUDA          |          354 |              0.488 |             nan     |                   nan     |
| V4 MPI+CUDA      |          576 |              0.429 |               0.401 |                     1.07  |