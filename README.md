### 1. Memory Bandwidth Test

CMD:

```bash
bash build.sh standard/global_memory_bandwidth.cpp ; ./a.out
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
