### 1. Memory Bandwidth Test

CMD:

```bash
bash build_rocm.sh standard/global_memory_bandwidth.cpp ; ./a.out
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
bash build_rocm.sh standard/peer_memory_bandwidth.cpp ; ./a.out
```

Output:

```txt
1GB peer copy test ... (GBps)
    dst/src        [0]        [1]        [2]        [3]        [4]        [5]        [6]        [7]
        [0]  2440.089     61.253     61.234     61.252     61.285     61.221     61.263     61.242 
        [1]    61.238   2401.661     61.266     61.251     61.285     61.230     61.266     61.252 
        [2]    61.248     61.283   2498.451     61.241     61.282     61.261     61.276     61.254 
        [3]    61.259     61.273     61.258   2469.036     61.288     61.230     61.272     61.242 
        [4]    61.231     61.254     61.227     61.249   2485.036     61.239     61.271     61.261 
        [5]    61.232     61.260     61.247     61.245     61.265   2438.759     61.269     61.248 
        [6]    61.243     61.249     61.245     61.253     61.268     61.234   2457.958     61.253 
        [7]    61.234     61.257     61.255     61.234     61.290     61.224     61.270   2495.432
```

### 2. UVA Peer Access Test

CMD:

```bash
bash build_rocm.sh standard/uva_peer_access.cpp ; ./a.out
```

Output:

```txt
1GB threads uva peer copy test <src_r, dst_w> ... (GBps)
    dst/src        [0]        [1]        [2]        [3]        [4]        [5]        [6]        [7]
        [0]  5934.866    127.251    127.280    127.286    127.298    127.300    127.295    127.263 
        [1]   127.291   5914.597    127.291    127.302    127.297    127.301    127.296    127.264 
        [2]   127.289    127.274   5897.704    127.294    127.303    127.298    127.285    127.297 
        [3]   127.258    127.286    127.295   5963.874    127.297    127.254    127.289    127.299 
        [4]   127.294    127.312    127.310    127.295   5988.488    127.258    127.309    127.301 
        [5]   127.301    127.313    127.238    127.318    127.314   5936.834    121.056    127.301 
        [6]   127.314    127.311    127.308    127.315    127.308    127.275   5972.499    127.308 
        [7]   127.292    127.314    127.313    127.318    127.279    127.306    127.291   5997.184
```
