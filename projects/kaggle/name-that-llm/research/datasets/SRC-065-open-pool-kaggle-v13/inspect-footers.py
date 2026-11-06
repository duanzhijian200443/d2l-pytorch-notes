import pathlib,struct,json
from thrift.protocol import TCompactProtocol
from thrift.transport import TTransport
from thrift.Thrift import TType
p=pathlib.Path(__file__).resolve().parent
def value(pr,t):
    if t==TType.STRUCT:
        d={};pr.readStructBegin()
        while True:
            _,ft,fid=pr.readFieldBegin()
            if ft==TType.STOP:break
            d[fid]=value(pr,ft);pr.readFieldEnd()
        pr.readStructEnd();return d
    if t in [TType.LIST,TType.SET]:
        et,n=pr.readListBegin();a=[value(pr,et) for _ in range(n)];pr.readListEnd();return a
    if t==TType.MAP:
        kt,vt,n=pr.readMapBegin();a=[[value(pr,kt),value(pr,vt)] for _ in range(n)];pr.readMapEnd();return a
    if t==TType.STRING:return pr.readBinary().decode('utf8',errors='replace')
    return {TType.BOOL:pr.readBool,TType.BYTE:pr.readByte,TType.I16:pr.readI16,TType.I32:pr.readI32,TType.I64:pr.readI64,TType.DOUBLE:pr.readDouble}[t]()
results=[]
for f in p.glob('v*-tail.bin'):
    b=f.read_bytes();n=struct.unpack('<I',b[-8:-4])[0]
    assert n<=len(b)-8
    d=value(TCompactProtocol.TCompactProtocol(TTransport.TMemoryBuffer(b[-8-n:-8])),TType.STRUCT)
    r={'tail_file':f.name,'footer_bytes':n,'num_rows':d[3],'schema':[{'name':x.get(4),'physical_type_enum':x.get(1),'repetition_enum':x.get(3),'children':x.get(5),'converted_type_enum':x.get(6)} for x in d[2]],'row_groups':len(d[4]),'created_by':d.get(6)}
    results.append(r)
(p/'parquet-metadata-check.json').write_text(json.dumps(results,indent=2),encoding='utf8')
print(json.dumps(results,indent=2))
