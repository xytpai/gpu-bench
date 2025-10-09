### 1. Memory Bandwidth Test

CMD:

```bash
bash build_cuda.sh standard/global_memory_bandwidth.cpp ; ./a.out
```

Output:

```txt
1GB threads copy test ...
float1: timems:0.579721 throughput:3704.34 GBPS val:ok
float2: timems:0.381722 throughput:5625.78 GBPS val:ok
float4: timems:0.362282 throughput:5927.66 GBPS val:ok
float8: timems:0.456922 throughput:4699.89 GBPS val:ok
half1: timems:0.996206 throughput:2155.66 GBPS val:ok
half2: timems:0.583803 throughput:3678.44 GBPS val:ok
half4: timems:0.379642 throughput:5656.6 GBPS val:ok
half8: timems:0.359082 throughput:5980.48 GBPS val:ok
```

### 2. Peer Bandwidth Test

CMD:

```bash
bash build_cuda.sh standard/peer_memory_bandwidth.cpp ; ./a.out
```

Output:

```txt
1GB peer copy test ... (GBps)
    dst/src        [0]        [1]        [2]        [3]        [4]        [5]        [6]        [7]
        [0]  1483.725     36.285     37.911     37.010     38.260     38.362     38.205     38.445
        [1]    36.736   1492.502     36.623     35.575     38.173     38.403     38.335     38.273
        [2]    37.491     36.596   1492.037     34.792     38.568     38.526     38.345     38.354
        [3]    36.652     34.549     36.252   1487.672     38.726     38.200     38.444     38.461
        [4]    38.287     38.177     38.170     38.254   1487.803     38.673     38.411     38.666
        [5]    38.610     38.422     38.327     38.109     35.317   1488.992     36.720     33.830
        [6]    38.472     38.292     38.328     38.439     36.502     36.124   1489.785     33.482
        [7]    38.211     38.196     38.360     38.352     36.742     34.985     35.346   1491.839
```
