
## References

- Single-pass Parallel Prefix Scan with Decoupled Look-back, [PDF](https://research.nvidia.com/publication/single-pass-parallel-prefix-scan-decoupled-look-back)
- StreamScan: Fast Scan Algorithms for GPUs without Global Barrier Synchronization, [PDF](https://docs.google.com/viewer?a=v&pid=sites&srcid=ZGVmYXVsdGRvbWFpbnxzaGVuZ2VueWFufGd4OjQ3MjhiOTU3NGRhY2ZlYzA)
- A FASTER RADIX SORT IMPLEMENTATION, [PDF](https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21572-a-faster-radix-sort-implementation.pdf)
  - better to check the detail
- High Performance and Scalable Radix Sorting: a Case Study of Implementing Dynamic Parallelism for GPU Computing, [Archive](https://code.google.com/archive/p/back40computing/)
- Introduction to GPU Radix Sort, [PDF](http://www.heterogeneouscompute.org/wordpress/wp-content/uploads/2011/06/RadixSort.pdf)
  - intro


## Other Resources
- FidelityFX sort (Radix sort in DX), [web](https://gpuopen.com/fidelityfx-parallel-sort/)
  - Takahiro worked with Jason for this while ago. based on my old sort in OCL. 4 bit per pass, using wave intrinsics. 
- A radix sort based on old CUB used in HIPRT
  - https://github.com/Radeon-Pro/HIPRT/tree/master/hiprt/impl/radix_sort
  