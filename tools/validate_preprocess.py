#!/usr/bin/env python3
"""Compare the saved PC and ESP32 artifacts from one identical PCM capture."""
import argparse, math, struct
from pathlib import Path
N = 40 * 49
def floats(p):
    b = Path(p).read_bytes()
    if len(b) != N * 4: raise ValueError(f'{p}: expected {N*4} bytes, got {len(b)}')
    return struct.unpack('<%df' % N, b)
def ints(p):
    b = Path(p).read_bytes()
    if len(b) != N: raise ValueError(f'{p}: expected {N} bytes, got {len(b)}')
    return struct.unpack('<%db' % N, b)
def metrics(a,b):
    d=[abs(x-y) for x,y in zip(a,b)]; denom=sum(abs(x) for x in a)/len(a)
    return max(d),sum(d)/len(d),(sum(d)/len(d))/max(denom,1e-12)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--pc-mfcc',required=True); p.add_argument('--esp-mfcc',required=True); p.add_argument('--pc-input',required=True); p.add_argument('--esp-input',required=True); p.add_argument('--pc-output',nargs=3,type=int); p.add_argument('--esp-output',nargs=3,type=int); a=p.parse_args()
    maximum, mean, relative=metrics(floats(a.pc_mfcc),floats(a.esp_mfcc)); print(f'MFCC max_abs_diff={maximum:.9g} mean_abs_diff={mean:.9g} relative_error={relative:.9g}')
    maximum, mean, relative=metrics(ints(a.pc_input),ints(a.esp_input)); print(f'INT8 max_abs_diff={maximum:.9g} mean_abs_diff={mean:.9g} relative_error={relative:.9g}')
    if a.pc_output is None or a.esp_output is None: print('prediction_same=UNKNOWN — needs PC and ESP32 raw output vectors')
    else: print('prediction_same=' + str(max(range(3),key=lambda i:a.pc_output[i]) == max(range(3),key=lambda i:a.esp_output[i])))
if __name__ == '__main__': main()
