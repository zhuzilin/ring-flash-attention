from typing import List, Optional, Tuple
import time
from statistics import mean

import torch
import torch.distributed as dist

__all__ = ["send_recv_kv", "update_out_and_lse", "RingComm"]


req_wait_times = {}


def wait_reqs(reqs: Optional[List], key: str):
    if reqs is None:
        return

    start_time = time.perf_counter_ns() 
    for req in reqs:
        req.wait()
    elapsed = time.perf_counter_ns() - start_time

    times = req_wait_times.get(key)
    if times is None:
        req_wait_times[key] = [elapsed]
    else:
        times.append(elapsed)


def reset_wait_times():
    req_wait_times.clear()


def print_wait_times(rank):
    for key, times in req_wait_times.items():
        mean(times)
        print(f"{key}: rank={rank}, total={sum(times)//1000}us, mean={mean(times)//1000:.2f}us, min={min(times)/1000:.2f}us, max={max(times)//1000:.2f}us, num_calls={len(times)}")


@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(1, 2).unsqueeze(dim=-1)

    new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))

    out = torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out

    lse = new_lse
    return out, lse


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(1, 2).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(
            slice_out, slice_lse, block_out, block_lse
        )
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


class RingComm:
    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

    def send_recv(
        self, to_send: torch.Tensor, send_direction="incr", inplace=False
    ) -> torch.Tensor:
        if inplace:
            res = to_send
        else:
            res = torch.empty_like(to_send)
        if send_direction == "incr":
            send_rank = (self.rank + 1) % self.world_size
            recv_rank = (self.rank - 1) % self.world_size
        else:
            send_rank = (self.rank - 1) % self.world_size
            recv_rank = (self.rank + 1) % self.world_size

        send_op = dist.P2POp(dist.isend, to_send, send_rank, group=self._process_group)
        recv_op = dist.P2POp(dist.irecv, res, recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        wait_reqs(self._reqs, "RingComm.wait()")
        self._reqs = None
        self._ops = []


def send_recv_kv(process_group, local_k, local_v, step, causal, is_grad=False):
    rank = dist.get_rank(process_group)
    world_size = dist.get_world_size(process_group)
    if step == 0:
        assert is_grad
        return None, None, rank, None
    send_rank = (rank + step) % world_size
    recv_rank = (rank - step) % world_size

    need_to_recv = not (causal and step > rank)
    need_to_send = not (causal and step + rank >= world_size)

    if is_grad:
        # when sending grad, we need to reverse send and recv
        send_rank, recv_rank = recv_rank, send_rank
        need_to_recv, need_to_send = need_to_send, need_to_recv

    ops = []
    if need_to_recv:
        remote_k = torch.empty_like(local_k)
        remote_v = torch.empty_like(local_v)
        recv_k = dist.P2POp(dist.irecv, remote_k, recv_rank, group=process_group)
        recv_v = dist.P2POp(dist.irecv, remote_v, recv_rank, group=process_group)
        ops += [recv_k, recv_v]
    if need_to_send:
        # need to send
        send_k = dist.P2POp(dist.isend, local_k, send_rank, group=process_group)
        send_v = dist.P2POp(dist.isend, local_v, send_rank, group=process_group)
        ops += [send_k, send_v]

    if need_to_recv or need_to_send:
        reqs = dist.batch_isend_irecv(ops)
    else:
        reqs = None

    if not need_to_recv:
        return None, None, None, reqs
    return remote_k, remote_v, recv_rank, reqs
