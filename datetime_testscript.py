#!/usr/bin/python3
import numpy as np
import pandas as pd
import time

print("numpy astype test")
try:
    print(np.array([12.7,-435,np.nan]).astype('i8'))
except Exception as e:
    print(e)
print("arithmetic overflow test")
try:
    print(np.array([2**36,857894946,-2**35],dtype='i8')*(2**36))
except Exception as e:
    print(e)
try:
    print(np.array([2**36,857894946,-2**35],dtype='f8')*(2**36))
except Exception as e:
    print(e)

print("near-limits test")
timestamp_overflow_years=2**63/(1e9*60*60*24*365.2425)
for a in [timestamp_overflow_years+2,timestamp_overflow_years+0.1,timestamp_overflow_years+0.05,timestamp_overflow_years-0.1,-timestamp_overflow_years+0.1,-timestamp_overflow_years-0.05,-timestamp_overflow_years-0.1,-timestamp_overflow_years-2]:
    try:
        b=pd.Series([0,1,a],dtype=float)
        print(b,pd.to_datetime(b,unit="Y",errors='raise'))
    except Exception as e:
        print(a,e)

print("speed test, one long")
try:
    a=pd.Series(1e8*np.random.randn(1000000))
    t=time.time()
    b=pd.to_datetime(a,unit="s",errors='raise')
    print(time.time()-t)
except Exception as e:
    print(e)

print("speed test, many short")
try:
    a=pd.DataFrame(1e8*np.random.randn(10,1000))
    t=time.time()
    for n in range(1000):
        b=pd.to_datetime(a[n],unit="s",errors='raise')
    print(time.time()-t)
except Exception as e:
    print(e)

print("rolling instability test")
try:
    a=pd.Series([10**17,1,1,1,1,1,1]).rolling(2)
    print(a.std(),a.var())
    a=pd.Series([10**17,1,2,3,5,4]).rolling(2)
    print(a.std(),a.var())
except Exception as e:
    print(e)
print("arange overflow test, based on test_date_range_int64_overflow_stride_endpoint_different_signs")
with np.errstate(over="raise"):
    for n in range(10):
        start=9219225600000000000
        end=    -4150800000000000
        step=      -3600000000000
        if n!=0:
            start=start+int(np.random.rand(1)*2.**32.)
        r1=np.arange(start,end,step,dtype=np.int64)
        print("int",hex(start),r1.shape,r1[0],r1[-1])
        r1=np.arange(np.int64(start),end,step,dtype=np.int64)
        print("int64",r1.shape,r1[0],r1[-1])
