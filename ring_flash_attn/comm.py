import torch
import torch.distributed as dist


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
