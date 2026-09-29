"""AiiStream prefetching expert reader (default mode).

Reads the experts the model selects, and reads likely next experts ahead of time.
The model's own router always decides which experts are used, so output is unchanged.
Call reset_request() at the end of every request."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from threading import Condition, Thread
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import aiistream_model as p1
import aiistream_parallel as p3
import aiistream_direct as z0

EXPERTS=8
CAPACITY=16
LAYER_COUNT=40

def tensor_keys(layer):
    prefix=f"language_model.model.layers.{int(layer)}.mlp.switch_mlp."
    return [prefix+proj+"."+kind for proj in ("gate_proj","up_proj","down_proj")
            for kind in ("weight","scales","biases")]

class BatchPriorityPool:
    """Eight read threads; requested reads run before speculative ones."""
    def __init__(self,workers=8):
        if workers!=8:raise ValueError("eight workers required")
        self.cv=Condition()
        self.demand,self.prefetch=deque(),deque()
        self.closed=False
        self.threads=[]
        for i in range(workers):
            t=Thread(target=self._worker,name=f"aiistream-read-{i}",daemon=False)
            t.start();self.threads.append(t)
    def _worker(self):
        while True:
            with self.cv:
                while not self.demand and not self.prefetch and not self.closed:self.cv.wait()
                if not self.demand and not self.prefetch and self.closed:return
                future,fn,args=(self.demand.popleft() if self.demand else self.prefetch.popleft())
            if not future.set_running_or_notify_cancel():continue
            try:result=fn(*args)
            except BaseException as e:future.set_exception(e)
            else:future.set_result(result)
    def submit_many(self,tasks,priority="demand"):
        if priority not in ("demand","prefetch"):raise ValueError(priority)
        done=[]
        with self.cv:
            if self.closed:raise RuntimeError("pool closed")
            q=self.demand if priority=="demand" else self.prefetch
            for fn,args in tasks:
                f=Future();q.append((f,fn,args));done.append(f)
            self.cv.notify_all()
        return done
    def submit(self,fn,*args):
        return self.submit_many([(fn,args)],"demand")[0]
    def promote_many(self,futures):
        target=set(futures)
        with self.cv:
            for task in tuple(self.prefetch):
                if task[0] in target:
                    self.prefetch.remove(task)
                    self.demand.appendleft(task)
            self.cv.notify_all()
    def shutdown(self):
        with self.cv:
            if self.closed:return
            self.closed=True;self.cv.notify_all()
        for t in self.threads:t.join()
        if self.demand or self.prefetch:raise RuntimeError("pending queue at shutdown")

MX_DTYPES={"U8":mx.uint8,"I8":mx.int8,"U16":mx.uint16,"U32":mx.uint32,
           "F16":mx.float16,"F32":mx.float32,"BF16":mx.bfloat16}

@dataclass
class ReusableBank:
    raw:np.ndarray
    per:int
    meta:dict
    shard_name:str
    fd:int
    base:int
    key:str
    parity:int
    layer:int=-1
    call_key:str=""
    used:int=0
    slots:dict=field(default_factory=dict)
    futures:dict=field(default_factory=dict)
    kinds:dict=field(default_factory=dict)
    def reset(self,layer,call_key):
        if any(not f.done() for f in self.futures.values()):
            raise RuntimeError("attempt to recycle in-flight buffer")
        for f in self.futures.values():f.result()
        self.layer=int(layer);self.call_key=str(call_key)
        self.used=0;self.slots.clear();self.futures.clear();self.kinds.clear()
    def can_recycle(self,e):
        return e in self.slots and self.futures[e].done()

class PrefetchShard(z0.DirectShardIndex):
    """Expert reader with bounded request-scoped staging buffers."""
    LAST_INSTANCE=None
    def __init__(self):
        p1.ShardIndex.__init__(self)
        self.parallel_pread=True;self.workers=8;self.profile_timing=False
        self.executor_creation_count=0
        self.read_path_s_total=0.;self.read_path_invocations=0
        self.read_path_s_by_call=defaultdict(float)
        self.checkpoint_read_started=False;self.short_reads_total=0
        self.fetch_exceptions=0
        self.pool=BatchPriorityPool();self.executor=self.pool
        self.reusable={};self.banks={};self._layer_groups={}
        self.predictions={};self.orphans=[];self._slotmap={}
        self.current_call_key=None;self.current_layer=None
        self.decode_step=0;self._closed=False
        self.max_live_slots=0
        type(self).LAST_INSTANCE=self
    def resolve_call_key(self,layer,x):
        if int(x.shape[-2])!=1:return None,False
        if layer==0:self.decode_step+=1
        return f"decode:{self.decode_step}",True
    def _locate(self,key):
        shard_name=self.weight_map[key]
        _,fd,header,data_start=self.shards[shard_name]
        meta=header[key];begin,end=meta["data_offsets"]
        per=(end-begin)//meta["shape"][0]
        return shard_name,fd,meta,per,data_start+begin
    def _bank(self,key,layer,call_key):
        ident=(layer%2,key.rsplit("switch_mlp.",1)[1])
        bank=self.reusable.get(ident)
        if bank is None:
            shard_name,fd,meta,per,base=self._locate(key)
            mbuf=mx.zeros((CAPACITY*per,),dtype=mx.uint8);mx.eval(mbuf)
            raw=np.asarray(mbuf)
            if not raw.flags.writeable or raw.ctypes.data!=np.asarray(mbuf).ctypes.data:
                raise RuntimeError("MLX buffer is not a writable shared view")
            bank=ReusableBank(raw,per,meta,shard_name,fd,base,key,layer%2)
            bank.mbuf=mbuf
            self.reusable[ident]=bank
        if bank.layer!=layer or bank.call_key!=str(call_key):
            for f in bank.futures.values():f.result()
            bank.reset(layer,call_key)
            if bank.key!=key:
                shard_name,fd,meta,per,base=self._locate(key)
                if per!=bank.per or meta["dtype"]!=bank.meta["dtype"] or list(meta["shape"][1:])!=list(bank.meta["shape"][1:]):
                    raise RuntimeError(f"cannot share bank across layers: {key} vs {bank.key}")
                bank.shard_name,bank.fd,bank.meta,bank.base,bank.key=shard_name,fd,meta,base,key
        self.banks[(str(call_key),layer,key)]=bank
        return bank
    def _group(self,key,layer):
        ident=(str(key),int(layer))
        if ident not in self._layer_groups:
            self._layer_groups[ident]=[self._bank(name,layer,key) for name in tensor_keys(layer)]
        return self._layer_groups[ident]
    def _slot_map(self,group):
        mapping=group[0].slots
        if any(bank.slots!=mapping for bank in group[1:]):
            raise RuntimeError("nine tensor slot maps disagree")
        return mapping
    def _enqueue(self,group,experts,priority,assigned):
        tasks=[];targets=[]
        for bank in group:
            for expert in experts:
                slot=assigned[expert]
                if expert in bank.futures and bank.slots.get(expert)==slot:continue
                bank.slots[expert]=slot;bank.kinds[expert]=priority
                view=memoryview(bank.raw)[slot*bank.per:(slot+1)*bank.per]
                tasks.append((z0.preadv_exact_into,(bank.fd,view,bank.base+expert*bank.per)))
                targets.append((bank,expert))
        fs=self.pool.submit_many(tasks,priority)
        for (bank,e),future in zip(targets,fs):bank.futures[e]=future
    def _sweep_orphans(self,block=False):
        waiting=[]
        for f in self.orphans:
            if block or f.done():f.result()
            else:waiting.append(f)
        self.orphans=waiting
    def _assert_bound(self):
        layers=defaultdict(int)
        for bank in self.banks.values():
            if bank.used>CAPACITY or len(bank.slots)>CAPACITY:raise RuntimeError("bank overflow")
            layers[bank.layer]+=len(bank.slots)
        if len(layers)>2:raise RuntimeError("more than two live layers")
        self.max_live_slots=max(self.max_live_slots,sum(layers.values()))
    def prepare_layer(self,key,layer,actual_ids,prediction):
        self.current_call_key=key;self.current_layer=layer
        self._sweep_orphans()
        k=str(key);actual=set(map(int,actual_ids))
        group=self._group(k,layer);mapping=self._slot_map(group)
        existing=actual&set(mapping)
        self.pool.promote_many(bank.futures[e] for bank in group for e in existing)
        missing=sorted(actual-set(mapping))
        reusable=[e for e in sorted(mapping) if e not in actual
                  and all(bank.can_recycle(e) for bank in group)]
        assigned={}
        for e in missing:
            if reusable:
                discarded=reusable.pop(0);slot=mapping[discarded]
                for bank in group:
                    bank.futures[discarded].result()
                    del bank.slots[discarded];del bank.futures[discarded];del bank.kinds[discarded]
            else:
                slot=group[0].used
                for bank in group:bank.used=max(bank.used,slot+1)
            if slot>=CAPACITY:raise RuntimeError("slot overflow")
            assigned[e]=slot
        self._enqueue(group,missing,"demand",assigned)
        if prediction is not None and layer<LAYER_COUNT-1:
            next_ids=sorted(set(map(int,prediction)))
            if len(next_ids)!=EXPERTS:raise RuntimeError("prediction cardinality")
            next_group=self._group(k,layer+1)
            next_map=self._slot_map(next_group)
            new=[e for e in next_ids if e not in next_map]
            assigned={}
            for e in new:
                slot=next_group[0].used
                for bank in next_group:bank.used=slot+1
                if slot>=CAPACITY:raise RuntimeError("prefetch overflow")
                assigned[e]=slot
            self._enqueue(next_group,new,"prefetch",assigned)
            self.predictions[(k,layer+1)]=set(next_ids)
        self.predictions.pop((k,layer),None)
        self._slotmap[(k,layer)]=dict(self._slot_map(group))
        self._assert_bound()
    def slotmap(self,key,layer):
        return self._slotmap[(str(key),int(layer))]
    def pread_experts(self,key,expert_ids,telemetry=None):
        if self.current_call_key is None:
            return z0.DirectShardIndex.pread_experts(self,key,expert_ids,telemetry)
        bank=self.banks.pop((str(self.current_call_key),int(self.current_layer),key))
        requested=list(map(int,expert_ids))
        if any(e not in bank.slots for e in requested):raise RuntimeError("missing staged expert")
        try:
            for e in requested:
                calls,sizes,short=bank.futures[e].result()
                self.short_reads_total+=int(short)
                if telemetry is not None:
                    if hasattr(telemetry, "pread_calls"):
                        telemetry.pread_calls+=int(calls)
                    if hasattr(telemetry, "pread_sizes"):
                        telemetry.pread_sizes.update(sizes)
                    telemetry.short_reads+=int(short)
            for e,future in bank.futures.items():
                if e not in requested:
                    _,_,short=future.result();self.short_reads_total+=int(short)
            count=bank.used
            result=bank.mbuf[:count*bank.per].view(MX_DTYPES[bank.meta["dtype"]]).reshape(
                (count,)+tuple(bank.meta["shape"])[1:])
            for e,future in bank.futures.items():
                if e not in requested:self.orphans.append(future)
            group_key=(str(self.current_call_key),int(self.current_layer))
            if not any(x[0]==group_key[0] and x[1]==group_key[1] for x in self.banks):
                self._layer_groups.pop(group_key,None)
            return result,len(requested)*bank.per,bank.shard_name
        except BaseException:
            self.fetch_exceptions+=1
            raise
    def pending_futures(self):
        return sum(not f.done() for b in self.reusable.values() for f in b.futures.values()) + sum(not f.done() for f in self.orphans)
    def live_slots(self):
        return sum(b.used for b in self.reusable.values())
    def reset_request(self):
        """Run in the server's unconditional finally, including abort/errors."""
        error=None
        try:
            for bank in self.reusable.values():
                for f in bank.futures.values():
                    try:f.result()
                    except BaseException as e:
                        if error is None:error=e
            for f in self.orphans:
                try:f.result()
                except BaseException as e:
                    if error is None:error=e
        finally:
            for bank in self.reusable.values():
                bank.used=0;bank.slots.clear();bank.futures.clear();bank.kinds.clear()
                bank.layer=-1;bank.call_key=""
            self.banks.clear();self._layer_groups.clear()
            self.predictions.clear();self.orphans.clear();self._slotmap.clear()
            self.current_call_key=None;self.current_layer=None;self.decode_step=0
            if self.live_slots() or self.pending_futures():raise RuntimeError("request reset retained live slots/futures")
        if error is not None:raise error
    def close(self):
        if self._closed:return
        try:self.reset_request()
        finally:
            try:self.pool.shutdown()
            finally:
                p1.ShardIndex.close(self);self._closed=True

def _switch_projection(inp,out,k,tensors):
    return p1.SwitchProjection(tensors)

class _SwitchBase(nn.Module):
    def __init__(self,layer_idx,args,shard,telemetry):
        super().__init__()
        self.layer_idx=int(layer_idx);self.args=args;self.shard=shard
        self.telemetry=telemetry;self.activation=p1.SwiGLU()
    def _finish_demand(self,x,src,ids,indices):
        remap={e:i for i,e in enumerate(ids)}
        idx=mx.array(np.vectorize(remap.__getitem__)(src).astype(np.int32))
        prefix=f"language_model.model.layers.{self.layer_idx}.mlp.switch_mlp."
        mods=[];total=0
        for proj,inp,out in (("gate_proj",self.args.hidden_size,self.args.moe_intermediate_size),
                             ("up_proj",self.args.hidden_size,self.args.moe_intermediate_size),
                             ("down_proj",self.args.moe_intermediate_size,self.args.hidden_size)):
            tensors={}
            for kind in ("weight","scales","biases"):
                ar,nb,_=self.shard.pread_experts(prefix+proj+"."+kind,ids,self.telemetry)
                tensors[kind]=ar;total+=nb
            mods.append(_switch_projection(inp,out,len(ids),tensors))
        self.telemetry.record(self.layer_idx,total,0.,ids)
        return self._compute(x,src,indices,idx,mods)
    def _compute(self,x,src,indices,idx,mods):
        gate,up,down=mods
        xe=mx.expand_dims(x,(-2,-3))
        do_sort,inv=idx.size>=64,None
        if do_sort:xe,idx,inv=p1._gather_sort(xe,idx)
        xu=up(xe,idx,sorted_indices=do_sort)
        xg=gate(xe,idx,sorted_indices=do_sort)
        y=down(self.activation(xu,xg),idx,sorted_indices=do_sort)
        if do_sort:y=p1._scatter_unsort(y,inv,indices.shape)
        return y.squeeze(-2)
    def _finish_staged(self,x,src,ids,indices):
        slots=self.shard.slotmap(self.shard.current_call_key,self.layer_idx)
        idx=mx.array(np.vectorize(slots.__getitem__)(src).astype(np.int32))
        prefix=f"language_model.model.layers.{self.layer_idx}.mlp.switch_mlp."
        mods=[];total=0;used=max(slots.values())+1
        for proj,inp,out in (("gate_proj",self.args.hidden_size,self.args.moe_intermediate_size),
                             ("up_proj",self.args.hidden_size,self.args.moe_intermediate_size),
                             ("down_proj",self.args.moe_intermediate_size,self.args.hidden_size)):
            tensors={}
            for kind in ("weight","scales","biases"):
                ar,nb,_=self.shard.pread_experts(prefix+proj+"."+kind,ids,self.telemetry)
                tensors[kind]=ar;total+=nb
            mods.append(_switch_projection(inp,out,used,tensors))
        self.telemetry.record(self.layer_idx,total,0.,ids)
        self.shard._slotmap.pop((str(self.shard.current_call_key),self.layer_idx),None)
        return self._compute(x,src,indices,idx,mods)

class PrefetchSwitch(_SwitchBase):
    def __init__(self,layer_idx,args,shard,telemetry,next_gate=None):
        super().__init__(layer_idx,args,shard,telemetry)
        self.next_gate=next_gate
    def __call__(self,x,indices):
        key,eligible=self.shard.resolve_call_key(self.layer_idx,x)
        if not eligible:
            self.shard.current_call_key=None;self.shard.current_layer=self.layer_idx
            mx.eval(indices)
            src=np.array(indices);ids=sorted(set(map(int,src.reshape(-1))))
            return self._finish_demand(x,src,ids,indices)
        if self.layer_idx<LAYER_COUNT-1:
            probs=mx.softmax(self.next_gate(x),axis=-1,precise=True)
            predicted_mx=mx.argpartition(probs,kth=-EXPERTS,axis=-1)[...,-EXPERTS:]
            mx.eval(indices,predicted_mx)
            prediction=[int(e) for e in np.array(predicted_mx).reshape(-1)]
        else:
            mx.eval(indices);prediction=None
        src=np.array(indices);ids=sorted(set(map(int,src.reshape(-1))))
        self.shard.prepare_layer(key,self.layer_idx,ids,prediction)
        return self._finish_staged(x,src,ids,indices)

def install_blocks(model,shard,telemetry):
    args=model.language_model.args
    gates=[layer.mlp.gate for layer in model.layers]
    for i,layer in enumerate(model.layers):
        layer.mlp.switch_mlp=PrefetchSwitch(i,args,shard,telemetry,
                    next_gate=gates[i+1] if i<LAYER_COUNT-1 else None)
    return model
