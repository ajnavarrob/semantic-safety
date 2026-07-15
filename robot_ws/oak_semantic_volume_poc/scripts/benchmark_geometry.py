#!/usr/bin/env python3
"""CPU-only reference benchmark for the same occupancy/ray workload."""
import argparse, math, random, time

def cast(origin, direction, occupied, voxel, min_r, max_r):
    n=math.sqrt(sum(x*x for x in direction)); d=[x/n for x in direction]
    steps=int((max_r-min_r)/voxel)
    for i in range(steps):
        t=min_r+i*voxel
        p=[origin[j]+t*d[j] for j in range(3)]
        k=tuple(math.floor(x/voxel) for x in p)
        if k in occupied:return k
    return None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--rays',type=int,default=2500);ap.add_argument('--runs',type=int,default=100);a=ap.parse_args()
    voxel=.1; occupied=set()
    for y in range(-20,21):
        for z in range(-10,16): occupied.add((30,y,z))
    dirs=[((random.random()-.5)*.8,(random.random()-.5)*.6,1.0) for _ in range(a.rays)]
    samples=[]
    for _ in range(a.runs):
        t=time.perf_counter(); hits=sum(cast((0,0,0),d,occupied,voxel,.35,8.0) is not None for d in dirs);samples.append((time.perf_counter()-t)*1000)
    print(f'rays={a.rays} hits={hits} mean_ms={sum(samples)/len(samples):.3f} max_ms={max(samples):.3f}')
if __name__=='__main__':main()
