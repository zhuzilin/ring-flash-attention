from typing import Optional, Tuple

import torch
import torch.distributed as dist

__all__ = ['send_recv_kv', 'update_out_and_lse', 'RingComm']
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
        out = block_out
        lse = block_lse
    else:
        lse_with_slice = lse
        if slice_ is not None:
            lse_with_slice = lse[slice_]

        new_lse = lse_with_slice + torch.log(1 + torch.exp(block_lse - lse_with_slice))

        sliced_out = out
        if slice_ is not None:
            sliced_out = out[slice_]

        out[slice_] = (
            torch.exp(lse_with_slice - new_lse) * sliced_out
            + torch.exp(block_lse - new_lse) * block_out
        )

        if slice_ is not None:
            lse[slice_] = new_lse
        else:
            lse = new_lse
    return out, lse


class RingComm:
    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

    def send_recv(self, to_send: torch.Tensor,
                  send_direction="incr", inplace=False) -> torch.Tensor:
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
        for req in self._reqs:
            req.wait()
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
