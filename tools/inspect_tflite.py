#!/usr/bin/env python3
"""Minimal dependency-free inspector for the checked-in TFLite flatbuffer."""
import argparse, struct
TYPES = {9:'INT8'}
class FB:
 def __init__(self,b): self.b=b
 def u16(self,p): return struct.unpack_from('<H',self.b,p)[0]
 def i32(self,p): return struct.unpack_from('<i',self.b,p)[0]
 def u32(self,p): return struct.unpack_from('<I',self.b,p)[0]
 def field(self,t,n):
  v=t-self.i32(t); off=4+2*n
  return t+self.u16(v+off) if off < self.u16(v) else None
 def indirect(self,p): return p+self.u32(p)
 def vector(self,p):
  p += self.u32(p)
  return p+4, self.u32(p)
 def vector_field(self,t,n):
  p=self.field(t,n); return self.vector(p) if p else (None,0)
 def table_vector(self,t,n):
  p,count=self.vector_field(t,n); return [self.indirect(p+4*i) for i in range(count)]
 def i32_vector(self,t,n):
  p,count=self.vector_field(t,n); return [self.i32(p+4*i) for i in range(count)]
 def scalar(self,t,n,fmt,default=0):
  p=self.field(t,n); return struct.unpack_from(fmt,self.b,p)[0] if p else default
 def quant(self,t):
  q=self.field(t,4)
  if not q:return None
  q=self.indirect(q)
  scale=self.vector_field(q,2); zp=self.vector_field(q,3)
  s=struct.unpack_from('<f',self.b,scale[0])[0] if scale[1] else None
  z=struct.unpack_from('<q',self.b,zp[0])[0] if zp[1] else None
  return s,z
def main():
 a=argparse.ArgumentParser(); a.add_argument('model',nargs='?',default='ds_cnn_int8.tflite'); x=a.parse_args(); f=FB(open(x.model,'rb').read()); root=f.u32(0)
 subs=f.table_vector(root,2); codes=f.table_vector(root,1); print('model_bytes=',len(f.b),'operator_codes=',len(codes),'subgraphs=',len(subs))
 for si,s in enumerate(subs):
  tensors=f.table_vector(s,0); inputs=f.i32_vector(s,1); outputs=f.i32_vector(s,2); ops=f.table_vector(s,3)
  print(f'subgraph[{si}] tensors={len(tensors)} operators={len(ops)} inputs={inputs} outputs={outputs}')
  for label,ids in [('input',inputs),('output',outputs)]:
   for i in ids:
    t=tensors[i]; shape=f.i32_vector(t,0); typ=f.scalar(t,1,'<b'); print(f'  {label}[{i}] shape={shape} type={TYPES.get(typ,typ)} quant={f.quant(t)}')
  # This intentionally reports the exact operator count but does not decode
  # builtin options; use the generated TFLite schema/runtime for that deeper
  # inspection rather than making an incomplete host-side schema parser.
if __name__ == '__main__': main()
