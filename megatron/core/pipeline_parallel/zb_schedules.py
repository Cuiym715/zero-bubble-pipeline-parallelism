import contextlib
import itertools
from typing import Iterator, List, Union

import torch

from megatron import core, get_args, get_num_microbatches, print_rank_0
from megatron.core import parallel_state
from megatron.core.pipeline_parallel import auto_schedule, v_schedule, v_schedule_greedy
from megatron.core.utils import get_model_config, get_model_type
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank,
)
from megatron.core.pipeline_parallel.schedules import (
    recv_forward,
    send_forward,
    recv_backward,
    send_backward,
    deallocate_output_tensor,
    forward_step,
    backward_step,
    get_tensor_shapes,
)
from megatron.core.weight_grad_store import WeightGradStore
from megatron.timers import Timer
from megatron.utils import is_second_last_pipeline_stage


AUTO_SCHEDULE_COMMUNICATION_TYPES = {'RECV_FORWARD', 'RECV_BACKWARD', 'SEND_FORWARD', 'SEND_BACKWARD'}


class ScheduleTimers:
    iter_counter = 0
    comm_time = 0
    concluded = False
    
    chunks = []

    def __init__(self):
        self.f = Timer('f')
        self.b = Timer('b')
        self.w = Timer('w')
        self.f_cnt = 0
        self.b_cnt = 0
        self.w_cnt = 0
        self.f_mem = 0
        self.b_mem = 0
    
    def conclusion(self):
        assert self.concluded
        assert self.f_cnt > 0
        assert self.b_cnt > 0
        avg_f = int(self.f.elapsed(reset=False) / self.f_cnt * 1000000)
        avg_b = int(self.b.elapsed(reset=False) / self.b_cnt * 1000000)
        avg_f_mem = self.f_mem / self.f_cnt // 1000000
        avg_b_mem = self.b_mem / self.b_cnt // 1000000
        if self.w_cnt > 0:
            avg_w = int(self.w.elapsed(reset=False) / self.w_cnt * 1000000)
        else:
            avg_w = avg_b
        avg_w_mem = 0 - avg_f_mem - avg_b_mem
        return (avg_f, avg_b, avg_w, int(self.comm_time * 1000000), 
            avg_f_mem, avg_b_mem, avg_w_mem)

    @classmethod
    def for_chunk(cls, chunk):
        while len(cls.chunks) <= chunk:
            cls.chunks.append(cls())
        return cls.chunks[chunk]
    
    @classmethod
    def joint_conclusion(cls):
        ret = [x.conclusion() for x in cls.chunks]
        ret = list(zip(*ret))
        # C is shared bwteen chunks
        ret[3] = ret[3][0]
        return ret




def bootstrap_and_profile_p2p_communication(
    config, send_tensor_shapes, recv_tensor_shapes
    ):
    if ScheduleTimers.iter_counter == 1 and parallel_state.get_pipeline_model_parallel_world_size() > 1:
        nccl_init_tensor = [torch.Tensor([0]).cuda()]
        shape = [(1,)]
        if get_args().zero_bubble_v_schedule:
            # Make everyone think they are the first chunk, so we still need additional check to prevent rank -1 to send_forward/recv_backward
            parallel_state.set_virtual_pipeline_model_parallel_rank(0)
        if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            recv_forward(shape, config)
        if not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            send_forward(nccl_init_tensor, shape, config)
            recv_backward(shape, config)
        if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            send_backward(nccl_init_tensor, shape, config)

        # Benchmarking the communication cost
        send_data = [torch.zeros(*shape, dtype=config.pipeline_dtype).cuda() for
                     shape in send_tensor_shapes]
        recv_data = [torch.zeros(*shape, dtype=config.pipeline_dtype).cuda() for
                     shape in recv_tensor_shapes]
        torch.distributed.barrier()
        t = Timer('comm-benchmark')
        t.start()
        print_rank_0(
            f"Start benchmarking communication with size {recv_tensor_shapes}, {send_tensor_shapes}")
        for i in range(10):
            if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                recv_forward(recv_tensor_shapes, config)
            if not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                send_forward(send_data, send_tensor_shapes, config)
                recv_backward(send_tensor_shapes, config)
            if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                send_backward(recv_data, recv_tensor_shapes, config)
        t.stop()
        per_communication = torch.cuda.FloatTensor([t.elapsed() / (
                parallel_state.get_pipeline_model_parallel_world_size() - 1) / 2 / 10])
        torch.distributed.all_reduce(per_communication,
                                     torch.distributed.ReduceOp.MAX)
        ScheduleTimers.comm_time = per_communication.item()
        print_rank_0(f"Communication time: {ScheduleTimers.comm_time}")


def fused_pipeline_ops(
    tensor_send_prev: List[torch.Tensor],
    tensor_recv_prev: List[torch.Tensor],
    tensor_send_next: List[torch.Tensor],
    tensor_recv_next: List[torch.Tensor],
):
    ops = []
    group = get_pipeline_model_parallel_group()
    for t in tensor_send_prev:
        send_prev_op = torch.distributed.P2POp(
            torch.distributed.isend,
            t,
            get_pipeline_model_parallel_prev_rank(),
            group,
        )
        ops.append(send_prev_op)
    for t in tensor_recv_prev:
        recv_prev_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            t,
            get_pipeline_model_parallel_prev_rank(),
            group,
        )
        ops.append(recv_prev_op)
    for t in tensor_send_next:
        send_next_op = torch.distributed.P2POp(
            torch.distributed.isend,
            t,
            get_pipeline_model_parallel_next_rank(),
            group,
        )
        ops.append(send_next_op)
    for t in tensor_recv_next:
        recv_next_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            t,
            get_pipeline_model_parallel_next_rank(),
            group,
        )
        ops.append(recv_next_op)
    if len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
    else:
        reqs = []
    return reqs


class ZeroBubbleVPipeScheduler:

    def __init__(self):
        self._reset()
        self.tensor_shape = None
        self.config = None
        self.run_timer = None
        self.forward_step_func = None
        self.model = None
        self.data_iterator = None
        self.num_microbatches = None
        self.collect_non_loss_data = None
        self.forward_only = None
        self.model_type = None
        self.no_sync_context = None
        self.no_sync_func = None

        self.schedules = None
        self.it = 0
        self.do_post_validation = False
        self.is_first_run = True
        self.optimizer = None
        
        self.flag_debug = 0
    def _free_buffers(self):
        # two dim array, first dim is the model chunk, second dim is the microbatch queue
        self.input_tensors = [[], []]
        self.output_tensors = [[], []]
        self.send_forward_buffer = [[], []]
        self.recv_forward_buffer = [[], []]
        self.send_backward_buffer = [[], []]
        self.recv_backward_buffer = [[], []]
        self.forward_data_store = []
        self.local_send_forward_buffer = []
        self.local_send_backward_buffer = []

        
    def _reset(self):
        self._free_buffers()

        self.send_handles = []
        self.w_clear_run = [False, False]
        # map of {direction -> {node, shape}}
        self.communication_batch = {
            'SEND_NEXT': [],
            'RECV_NEXT': [],
            'SEND_PREV': [],
            'RECV_PREV': [],
        }


    @classmethod
    def direction_map(cls, node):
        if node.chunk == 0:
            return {
                'SEND_FORWARD': 'SEND_NEXT',
                'RECV_FORWARD': 'RECV_PREV',
                'SEND_BACKWARD': 'SEND_PREV',
                'RECV_BACKWARD': 'RECV_NEXT',
            }[node.type]
        else:
            return {
                'SEND_FORWARD': 'SEND_PREV',
                'RECV_FORWARD': 'RECV_NEXT',
                'SEND_BACKWARD': 'SEND_NEXT',
                'RECV_BACKWARD': 'RECV_PREV',
            }[node.type]

    def buffer_map(self, node):
        return {
            'SEND_FORWARD': self.send_forward_buffer[node.chunk],
            'RECV_FORWARD': self.recv_forward_buffer[node.chunk],
            'SEND_BACKWARD': self.send_backward_buffer[node.chunk],
            'RECV_BACKWARD': self.recv_backward_buffer[node.chunk],
        }[node.type]

    def flush(self):
        name = '_'.join(
            [f'{v[0].type}.{v[0].chunk}.{v[0].minibatch}' for v in itertools.chain(*[vs for k, vs in self.communication_batch.items()])])
        sn_tensors = [
            self.buffer_map(x[0]).pop(0)
            for x in self.communication_batch['SEND_NEXT']
        ]
        sp_tensors = [
            self.buffer_map(x[0]).pop(0)
            for x in self.communication_batch['SEND_PREV']
        ]
        rn_tensors = [
            torch.empty(
                self.tensor_shape,
                requires_grad=True,
                device=torch.cuda.current_device(),
                dtype=self.config.pipeline_dtype,
            ) for x in self.communication_batch['RECV_NEXT']
        ]
        rp_tensors = [
            torch.empty(
                self.tensor_shape,
                requires_grad=True,
                device=torch.cuda.current_device(),
                dtype=self.config.pipeline_dtype,
            ) for x in self.communication_batch['RECV_PREV']
        ]
        if get_args().profile:
            torch.cuda.nvtx.range_push(name)
        req = fused_pipeline_ops(
            sp_tensors,
            rp_tensors,
            sn_tensors,
            rn_tensors
        )
        if get_args().profile:
            torch.cuda.nvtx.range_pop()
        # We don't care about the reqs order here, all users need to all reqs to finish
        for x in self.communication_batch['RECV_NEXT']:
            self.buffer_map(x[0]).append((rn_tensors.pop(0), req))
        for x in self.communication_batch['RECV_PREV']:
            self.buffer_map(x[0]).append((rp_tensors.pop(0), req))
        self.send_handles.append(req)
        assert(not rn_tensors)
        assert(not rp_tensors)
        for direction in ['SEND_PREV', 'SEND_NEXT']:
            for id, x in enumerate(self.communication_batch[direction]):
                if x[0].type == 'SEND_FORWARD':
                    deallocate_output_tensor(sp_tensors[id] if direction == 'SEND_PREV' else sn_tensors[id],
                                             self.config.deallocate_pipeline_outputs)
        for k, v in self.communication_batch.items():
            v.clear()


    def add_communication(
        self,
        scheduled_node: auto_schedule.ScheduledNode,
        next_is_comm: bool,
        next_compute: auto_schedule.ScheduledNode
    ):
        if self.forward_only and 'BACKWARD' in scheduled_node.type:
            return
        self.communication_batch[self.direction_map(scheduled_node)].append(
            (scheduled_node, self.tensor_shape))
        def is_consumer(scheduled_node, next_compute):
            if scheduled_node.chunk == next_compute.chunk and scheduled_node.minibatch == next_compute.minibatch:
                if scheduled_node.type == 'RECV_FORWARD' and next_compute.type == 'F':
                    return True
                if scheduled_node.type == 'RECV_BACKWARD' and next_compute.type == 'B':
                    return True
            return False
        if (next_compute is not None and is_consumer(scheduled_node, next_compute)) or not next_is_comm or self.forward_only:
            self.flush()

    def schedule_f(self, scheduled_node):
        # print(f"rank {torch.distributed.get_rank()} schedule_f {scheduled_node.chunk} {scheduled_node.minibatch}")
        if self.config.timers is not None:
            self.config.timers('computeF', log_level=2).start()
        if core.parallel_state.is_pipeline_first_stage():
            input_tensor = None
        elif scheduled_node.chunk == 1 and core.parallel_state.is_pipeline_last_stage(
            ignore_virtual=True):
            input_tensor = self.local_send_forward_buffer.pop(0)
        else:
            input_tensor = self.recv_forward_buffer[scheduled_node.chunk].pop(0)
            for h in input_tensor[1]:
                h.wait()
            input_tensor = input_tensor[0]
        if get_args().profile:
            torch.cuda.nvtx.range_push(
                f'F{scheduled_node.minibatch}.{scheduled_node.chunk}')
        if self.run_timer:
            ScheduleTimers.for_chunk(scheduled_node.chunk).f_cnt += 1
            ScheduleTimers.for_chunk(scheduled_node.chunk).f.start()
            mem_before = torch.cuda.memory_allocated()
        output_tensor = forward_step(
            self.forward_step_func,
            self.data_iterator[scheduled_node.chunk],
            self.model[scheduled_node.chunk],
            self.num_microbatches,
            input_tensor,
            self.forward_data_store,
            self.config,
            self.collect_non_loss_data,
            checkpoint_activations_microbatch=None,
        )
        if self.run_timer:
            ScheduleTimers.for_chunk(scheduled_node.chunk).f.stop()
            ScheduleTimers.for_chunk(scheduled_node.chunk).f_mem += torch.cuda.memory_allocated() - mem_before
        if get_args().profile:
            torch.cuda.nvtx.range_pop()
        if not core.parallel_state.is_pipeline_last_stage():
            if scheduled_node.chunk == 0 and core.parallel_state.is_pipeline_last_stage(
                ignore_virtual=True):
                detached_output_tensor = output_tensor.detach()
                detached_output_tensor.requires_grad_()
                self.local_send_forward_buffer.append(detached_output_tensor)
                deallocate_output_tensor(output_tensor,
                                         self.config.deallocate_pipeline_outputs)
            else:
                self.send_forward_buffer[scheduled_node.chunk].append(output_tensor)
        else:
            if core.parallel_state.is_pipeline_last_stage():
                deallocate_output_tensor(output_tensor, self.config.deallocate_pipeline_outputs)
        if not self.forward_only:
            self.input_tensors[scheduled_node.chunk].append(input_tensor)
            self.output_tensors[scheduled_node.chunk].append(output_tensor)
        if self.config.timers is not None:
            self.config.timers('computeF').stop()

    def schedule_b(self, scheduled_node):
        if not self.forward_only:
            # print(f"rank {torch.distributed.get_rank()} schedule_b {scheduled_node.chunk} {scheduled_node.minibatch}")
            if self.config.timers is not None:
                self.config.timers('computeB', log_level=2).start()
            input_tensor = self.input_tensors[scheduled_node.chunk].pop(0)
            output_tensor = self.output_tensors[scheduled_node.chunk].pop(0)

            if core.parallel_state.is_pipeline_last_stage():
                # Keep the original behavior when we do a dummy communication
                output_tensor_grad = None
            elif scheduled_node.chunk == 0 and core.parallel_state.is_pipeline_last_stage(
                ignore_virtual=True):
                output_tensor_grad = self.local_send_backward_buffer.pop(0)
            else:
                output_tensor_grad = self.recv_backward_buffer[
                    scheduled_node.chunk].pop(0)
                for h in output_tensor_grad[1]:
                    h.wait()
                output_tensor_grad = output_tensor_grad[0]
            if get_args().profile:
                torch.cuda.nvtx.range_push(
                    f'B{scheduled_node.minibatch}.{scheduled_node.chunk}')
            if self.run_timer:
                ScheduleTimers.for_chunk(scheduled_node.chunk).b_cnt += 1
                ScheduleTimers.for_chunk(scheduled_node.chunk).b.start()
                mem_before = torch.cuda.memory_allocated()
            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, self.model_type,
                self.config
            )
            if self.run_timer:
                ScheduleTimers.for_chunk(scheduled_node.chunk).b.stop()
                ScheduleTimers.for_chunk(scheduled_node.chunk).b_mem += torch.cuda.memory_allocated() - mem_before
            if get_args().profile:
                torch.cuda.nvtx.range_pop()
            if not core.parallel_state.is_pipeline_first_stage():
                if scheduled_node.chunk == 1 and core.parallel_state.is_pipeline_last_stage(
                    ignore_virtual=True):
                    self.local_send_backward_buffer.append(input_tensor_grad)
                else:
                    self.send_backward_buffer[scheduled_node.chunk].append(
                        input_tensor_grad)
            WeightGradStore.flush(chunk=scheduled_node.chunk)
            if self.config.timers is not None:
                self.config.timers('computeB').stop()

    def schedule_w(self, scheduled_node, non_w_pending):
        if not self.forward_only:
            # print(f"rank {torch.distributed.get_rank()} schedule_w {scheduled_node.chunk} {scheduled_node.minibatch}")
            if self.config.timers is not None:
                self.config.timers('computeW', log_level=2).start()
            chunk = scheduled_node.chunk

            if non_w_pending and scheduled_node.minibatch != self.num_microbatches - 1:
                if get_args().profile:
                    torch.cuda.nvtx.range_push(f'W{scheduled_node.minibatch}.{scheduled_node.chunk}')
                if self.run_timer:
                    ScheduleTimers.for_chunk(scheduled_node.chunk).w_cnt += 1
                    ScheduleTimers.for_chunk(scheduled_node.chunk).w.start()
                WeightGradStore.pop(chunk=scheduled_node.chunk)
                if self.run_timer:
                    ScheduleTimers.for_chunk(scheduled_node.chunk).w.stop()
                if get_args().profile:
                    torch.cuda.nvtx.range_pop()
            elif not self.w_clear_run[chunk]:
                # Clear if this is the last minibatch or there is no non-W pending
                pending_ws = WeightGradStore.weight_grad_queue[chunk].qsize()
                if get_args().profile:
                    torch.cuda.nvtx.range_push(f'W_clear.{chunk}')
                if self.run_timer:
                    ScheduleTimers.for_chunk(scheduled_node.chunk).w_cnt += pending_ws
                    ScheduleTimers.for_chunk(scheduled_node.chunk).w.start()
                WeightGradStore.clear(self.model[chunk], chunk=chunk)
                if self.run_timer:
                    ScheduleTimers.for_chunk(scheduled_node.chunk).w.stop()
                if get_args().profile:
                    torch.cuda.nvtx.range_pop()  # W
                self.w_clear_run[chunk] = True
            if self.config.timers is not None:
                self.config.timers('computeW').stop()

    def run_until_post_validation(self):
        print_rank_0("run until post validation")
        print_rank_0(f"schedules: {self.schedules}")
        optimizer = self.optimizer
        updated, grad_norm, rollback, succeed = None, None, None, None
        it = 0
        if optimizer.do_this_step:
            assert optimizer.do_prev_step
            for data_iter in self.data_iterator:
                if data_iter is None:
                    continue
                data_iter.clear_buffer()
                data_iter.save_to_buffer()
            while it < len(self.schedules):
                scheduled_node = self.schedules[it]
                parallel_state.set_virtual_pipeline_model_parallel_rank(scheduled_node.chunk)
                if scheduled_node.type in ["SEND_FORWARD", "RECV_FORWARD"]:
                    assert scheduled_node.chunk == 0
                    next_is_comm = it + 1 < len(self.schedules) and self.schedules[it + 1].type in AUTO_SCHEDULE_COMMUNICATION_TYPES
                    next_compute = list(filter(lambda x: x.type in ['F', 'B', 'W'], self.schedules[it + 1:]))
                    next_compute = next_compute[0] if len(next_compute) > 0 else None
                    self.add_communication(scheduled_node, next_is_comm, next_compute)
                elif scheduled_node.type == 'F':
                    assert scheduled_node.chunk == 0
                    self.schedule_f(scheduled_node)
                elif scheduled_node.type == "RECV_POST_VALIDATION":
                    optimizer.recv_post_validation()
                elif scheduled_node.type == "SEND_POST_VALIDATION":
                    optimizer.send_post_validation()
                elif scheduled_node.type == "POST_VALIDATION":
                    self.flush()
                    updated, grad_norm, rollback, succeed = optimizer.post_validation(self._free_buffers)
                    break
                else:
                    raise ValueError(f"Unexpected type {scheduled_node.type}")
                it += 1
            assert succeed is not None
        else:
            while it < len(self.schedules):
                scheduled_node = self.schedules[it]
                parallel_state.set_virtual_pipeline_model_parallel_rank(scheduled_node.chunk)
                if scheduled_node.type in ["SEND_FORWARD", "RECV_FORWARD", "F"]:
                    if optimizer.do_prev_step and scheduled_node.type == "RECV_FORWARD":
                        next_is_comm = it + 1 < len(self.schedules) and self.schedules[it + 1].type in AUTO_SCHEDULE_COMMUNICATION_TYPES
                        next_compute = list(filter(lambda x: x.type in ['F', 'B', 'W'], self.schedules[it + 1:]))
                        next_compute = next_compute[0] if len(next_compute) > 0 else None
                        self.add_communication(scheduled_node, next_is_comm, next_compute)
                elif scheduled_node.type == "RECV_POST_VALIDATION":
                    optimizer.recv_post_validation()
                elif scheduled_node.type == "SEND_POST_VALIDATION":
                    optimizer.send_post_validation()
                elif scheduled_node.type == "POST_VALIDATION":
                    self.flush()
                    updated, grad_norm, rollback, succeed = optimizer.post_validation(self._free_buffers)
                    break
                else:
                    raise ValueError(f"Unexpected type {scheduled_node.type}")
                it += 1
            assert not succeed
        if not succeed:
            if optimizer.do_prev_step:
                # send dummy recv_forward to clear send_forward request of last rank
                while it < len(self.schedules):
                    scheduled_node = self.schedules[it]
                    parallel_state.set_virtual_pipeline_model_parallel_rank(scheduled_node.chunk)
                    if scheduled_node.type == "RECV_FORWARD" and scheduled_node.rollback:
                        self.add_communication(scheduled_node, False, None)
                    it += 1
            self._reset()
            it = 0
        for data_iter in self.data_iterator:
            if data_iter is None:
                continue
            if succeed:
                data_iter.clear_buffer()
            data_iter.pop_from_buffer()
        self.it = it
        return updated, grad_norm, rollback

    def run(self):
        # print_rank_0(f"run {self.flag_debug}")
        # if torch.distributed.get_rank() == 0:
        #     self.flag_debug += 1
        #     if self.flag_debug > 0:
        #         for node in self.schedules:
        #             print(f"{node.type}-{node.minibatch}", end=', ')
        #         print()
        
        self.disable_grad_sync()

        if get_args().profile:
            torch.cuda.nvtx.range_push(f'iter_{torch.distributed.get_rank()}_{ScheduleTimers.iter_counter}')

        it = self.it
        while it < len(self.schedules):
            scheduled_node = self.schedules[it]
            parallel_state.set_virtual_pipeline_model_parallel_rank(scheduled_node.chunk)
            if "POST_VALIDATION" in scheduled_node.type:
                pass
            elif scheduled_node.type in AUTO_SCHEDULE_COMMUNICATION_TYPES:
                next_is_comm = it + 1 < len(self.schedules) and self.schedules[it + 1].type in AUTO_SCHEDULE_COMMUNICATION_TYPES
                next_compute = list(filter(lambda x: x.type in ['F', 'B', 'W'], self.schedules[it + 1:]))
                next_compute = next_compute[0] if len(next_compute) > 0 else None
                self.add_communication(scheduled_node, next_is_comm, next_compute)
            elif scheduled_node.type == 'F':
                self.schedule_f(scheduled_node)
            elif scheduled_node.type == 'B':
                self.schedule_b(scheduled_node)
            elif scheduled_node.type == 'W':
                non_w_pending = any([node.type != 'W' for node in self.schedules[it + 1:]])
                self.schedule_w(scheduled_node, non_w_pending)
            else:
                raise ValueError(f"Unknown node type {scheduled_node.type}")
            it += 1
        self.it = it
        for h in self.send_handles:
            for hh in h:
                hh.wait()

        if get_args().profile:
            torch.cuda.nvtx.range_pop()  # iter

        if not self.forward_only:
            # Launch any remaining grad reductions
            if self.no_sync_context is not None:
                self.enable_grad_sync()

            if self.config.finalize_model_grads_func is not None:
                # Finalize model grads (perform full grad all-reduce / reduce-scatter for
                # data parallelism, layernorm all-reduce for sequence parallelism).
                self.config.finalize_model_grads_func(self.model)

            if get_args().zero_bubble_pipeline_timers_end_iter == ScheduleTimers.iter_counter:
                ScheduleTimers.concluded = True

        return self.forward_data_store

    def disable_grad_sync(self):
        """Disable asynchronous grad reductions"""
        if self.no_sync_context is None:
            self.no_sync_context = self.no_sync_func()
            self.no_sync_context.__enter__()

    def enable_grad_sync(self):
        """Enable asynchronous grad reductions"""
        if self.no_sync_context is not None:
            self.no_sync_context.__exit__(None, None, None)
            self.no_sync_context = None

    def prepare(
        self,
        schedule: List[auto_schedule.ScheduledNode],
        forward_step_func,
        data_iterator: Union[Iterator, List[Iterator]],
        model: Union[torch.nn.Module, List[torch.nn.Module]],
        num_microbatches: int,
        seq_length: int,
        micro_batch_size: int,
        decoder_seq_length: int = None,
        forward_only: bool = False,
        collect_non_loss_data: bool = False,
    ):
        assert isinstance(model, list), "interleaved pipeline parallelism expected model chunking"
        assert all(isinstance(chunk, torch.nn.Module) for chunk in model), "invalid model chunking"
        assert isinstance(
            data_iterator, list
        ), "interleaved pipeline parallelism expected each model chunk to have a data iterator"
        config = get_model_config(model[0])
        self.config = config
        if config.overlap_p2p_comm and config.batch_p2p_comm:
            raise ValueError(
                "Can not use both overlap_p2p_comm and batch_p2p_comm")

        # Disable async grad reductions
        no_sync_func = config.no_sync_func
        if isinstance(no_sync_func, list):

            def multi_no_sync():
                stack = contextlib.ExitStack()
                for model_chunk_no_sync_func in config.no_sync_func:
                    stack.enter_context(model_chunk_no_sync_func())
                return stack

            no_sync_func = multi_no_sync
        assert no_sync_func is None, "Sync func is not supported yet"
        if no_sync_func is None:
            no_sync_func = contextlib.nullcontext
        self.no_sync_func = no_sync_func
        self.no_sync_context = None

        assert config.param_sync_func is None, "Param sync func is not supported yet"

        # Checkpoint the activations of partial Transformer layers in a number of micro-batches
        # within the maximum outstanding micro-batch backpropagations.
        # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
        # checkpoint partial Transformer layers (or skip checkpointing) and
        # the rest of micro-batches within a window of micro-batches checkpoint
        # all Transformer layers. The window of micro-batches is set by the maximum
        # outstanding backpropagations and becomes smaller at later pipeline stages.
        # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
        max_outstanding_backprops = None
        assert config.num_microbatches_with_partial_activation_checkpoints is None

        model_type = get_model_type(model[0])
        self.model_type = model_type

        tensor_shape = [seq_length, micro_batch_size, config.hidden_size]
        if config.sequence_parallel:
            tensor_shape[0] = tensor_shape[0] // parallel_state.get_tensor_model_parallel_world_size()
        tensor_shape = tuple(tensor_shape)

        self.tensor_shape = tensor_shape
        if decoder_seq_length is not None and decoder_seq_length != tensor_shape[0]:
            raise RuntimeError(
                "Interleaving is not supported with a different decoder sequence length."
            )

        rank = parallel_state.get_pipeline_model_parallel_rank()
        assert get_tensor_shapes(
            rank=rank - 1,
            model_type=model_type,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            config=config,
        )[0] == tensor_shape
        assert get_tensor_shapes(
            rank=rank,
            model_type=model_type,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            config=config,
        )[0] == tensor_shape
        if not forward_only:
            ScheduleTimers.iter_counter += 1
        run_timer = (
            get_args().zero_bubble_pipeline_timers_end_iter
            >= ScheduleTimers.iter_counter
            >= get_args().zero_bubble_pipeline_timers_start_iter
        )
        bootstrap_and_profile_p2p_communication(config, [tensor_shape], [tensor_shape])
        # TODO: DEBUG，但是这是用来profile然后获得固定调度的
        # self.run_timer = None
        self.run_timer = run_timer

        self.schedules = schedule
        # if torch.distributed.get_rank() == 0:
        #     print('prepare')
        #     for node in self.schedules:
        #         print(f"{node.type}-{node.minibatch}", end=', ')
        #     print()
        self.forward_step_func = forward_step_func
        self.data_iterator = data_iterator
        self.model = model
        self.num_microbatches = num_microbatches
        self.forward_only = forward_only
        self.collect_non_loss_data = collect_non_loss_data

        self._reset()
        self.it = 0

    def __call__(self, *args, **kwargs):
        if kwargs['forward_only']:
            self.prepare(*args, **kwargs)
            assert self.do_post_validation
            self.do_post_validation = True
            self.is_first_run = True
            return self.run()
        if not get_args().enable_optimizer_post_validation:
            self.prepare(*args, **kwargs)
            self.is_first_run = False
            self.do_post_validation = False
            return self.run()
        # enable_optimizer_post_validation == True
        if self.is_first_run:
            self.prepare(*args, **kwargs)
            self.is_first_run = False
            self.do_post_validation = False
        if self.do_post_validation:
            self.prepare(*args, **kwargs)
            result = self.run_until_post_validation()
            self.do_post_validation = False
        else:
            result = self.run()
            self.do_post_validation = True
        return result


class ZeroBubbleScheduler:

    def __init__(self):
        self._reset()

        self.schedules = None
        self.send_tensor_shapes = None
        self.recv_tensor_shapes = None
        self.config = None
        self.run_timer = None
        self.forward_step_func = None
        self.data_iterator = None
        self.model = None
        self.model_type = None
        self.num_microbatches = None
        self.collect_non_loss_data = None
        self.forward_only = None
        self.no_sync_context = None
        self.no_sync_func = None

        self.it = 0
        self.do_post_validation = False
        self.is_first_run = True
        self.optimizer = None

    def _free_buffers(self):
        self.input_tensors = []
        self.output_tensors = []
        self.send_forward_buffer = []
        self.recv_forward_buffer = []
        self.send_backward_buffer = []
        self.recv_backward_buffer = []
        self.forward_data_store = []

    def _reset(self):
        # Input, output tensors only need to be saved when doing backward passes
        self._free_buffers()
        self.send_handles = []
        self.communication_batch = {
            'SEND_NEXT': [],
            'RECV_NEXT': [],
            'SEND_PREV': [],
            'RECV_PREV': [],
        }

    @classmethod
    def direction_map(cls, node):
        return {
            'SEND_FORWARD': 'SEND_NEXT',
            'RECV_FORWARD': 'RECV_PREV',
            'SEND_BACKWARD': 'SEND_PREV',
            'RECV_BACKWARD': 'RECV_NEXT',
        }[node.type]

    def buffer_map(self, node):
        return {
            'SEND_FORWARD': self.send_forward_buffer,
            'RECV_FORWARD': self.recv_forward_buffer,
            'SEND_BACKWARD': self.send_backward_buffer,
            'RECV_BACKWARD': self.recv_backward_buffer,
        }[node.type]

    def flush(self):
        name = '_'.join(
            [f'{v[0].type}.{v[0].minibatch}' for v in itertools.chain(
                *[vs for k, vs in self.communication_batch.items()])])
        assert self.send_tensor_shapes == self.recv_tensor_shapes
        assert len(self.send_tensor_shapes) == 1
        sn_tensors = [
            self.buffer_map(x[0]).pop(0)[0]
            for x in self.communication_batch['SEND_NEXT']
        ]
        sp_tensors = [
            self.buffer_map(x[0]).pop(0)[0]
            for x in self.communication_batch['SEND_PREV']
        ]

        rn_tensors = [
            torch.empty(
                self.send_tensor_shapes[0],
                requires_grad=True,
                device=torch.cuda.current_device(),
                dtype=self.config.pipeline_dtype,
            ) for x in self.communication_batch['RECV_NEXT']
        ]
        rp_tensors = [
            torch.empty(
                self.send_tensor_shapes[0],
                requires_grad=True,
                device=torch.cuda.current_device(),
                dtype=self.config.pipeline_dtype,
            ) for x in self.communication_batch['RECV_PREV']
        ]
        if get_args().profile:
            torch.cuda.nvtx.range_push(name)
        req = fused_pipeline_ops(
            sp_tensors,
            rp_tensors,
            sn_tensors,
            rn_tensors
        )
        if get_args().profile:
            torch.cuda.nvtx.range_pop()
        # We don't care about the reqs order here, all users need to all reqs to finish
        for x in self.communication_batch['RECV_NEXT']:
            self.buffer_map(x[0]).append(([rn_tensors.pop(0)], [req]))
        for x in self.communication_batch['RECV_PREV']:
            self.buffer_map(x[0]).append(([rp_tensors.pop(0)], [req]))
        self.send_handles.append([req])
        assert(not rn_tensors)
        assert(not rp_tensors)
        for direction in ['SEND_PREV', 'SEND_NEXT']:
            for id, x in enumerate(self.communication_batch[direction]):
                if x[0].type == 'SEND_FORWARD':
                    deallocate_output_tensor(sp_tensors[id] if direction == 'SEND_PREV' else sn_tensors[id],
                                             self.config.deallocate_pipeline_outputs)
        for k, v in self.communication_batch.items():
            v.clear()

    def add_communication(
        self,
        scheduled_node: auto_schedule.ScheduledNode,
        next_is_comm: bool,
        next_compute: auto_schedule.ScheduledNode
    ):
        if self.forward_only and 'BACKWARD' in scheduled_node.type:
            return
        self.communication_batch[self.direction_map(scheduled_node)].append(
            (scheduled_node, None))
        def is_consumer(scheduled_node, next_compute):
            if scheduled_node.minibatch == next_compute.minibatch:
                if scheduled_node.type == 'RECV_FORWARD' and next_compute.type == 'F':
                    return True
                if scheduled_node.type == 'RECV_BACKWARD' and next_compute.type == 'B':
                    return True
            return False
        if (next_compute is not None and is_consumer(scheduled_node, next_compute)) or not next_is_comm or self.forward_only:
            self.flush()

    def schedule_f(self, scheduled_node):
        if core.parallel_state.is_pipeline_first_stage():
            input_tensor = [None] * len(self.recv_tensor_shapes)
        else:
            input_tensor = self.recv_forward_buffer.pop(0)
            for h in input_tensor[1]:
                for hh in h:
                    hh.wait()
            input_tensor = input_tensor[0]
        if get_args().profile:
            torch.cuda.nvtx.range_push(f'F{scheduled_node.minibatch}')
        if self.run_timer:
            ScheduleTimers.for_chunk(0).f_cnt += 1
            ScheduleTimers.for_chunk(0).f.start()
            mem_before = torch.cuda.memory_allocated()
        output_tensor = forward_step(
            self.forward_step_func,
            self.data_iterator,
            self.model,
            self.num_microbatches,
            input_tensor,
            self.forward_data_store,
            self.config,
            self.collect_non_loss_data,
            checkpoint_activations_microbatch=None,
        )
        if self.run_timer:
            ScheduleTimers.for_chunk(0).f.stop()
            ScheduleTimers.for_chunk(0).f_mem += torch.cuda.memory_allocated() - mem_before
        if get_args().profile:
            torch.cuda.nvtx.range_pop()
        if not core.parallel_state.is_pipeline_last_stage():
            self.send_forward_buffer.append(output_tensor)
        if not self.forward_only:
            self.input_tensors.append(input_tensor)
            self.output_tensors.append(output_tensor)
            if core.parallel_state.is_pipeline_last_stage():
                deallocate_output_tensor(output_tensor[0], self.config.deallocate_pipeline_outputs)

    def schedule_b(self, scheduled_node):
        if not self.forward_only:
            input_tensor = self.input_tensors.pop(0)
            output_tensor = self.output_tensors.pop(0)

            if core.parallel_state.is_pipeline_last_stage():
                # Keep the original behavior when we do a dummy communication
                output_tensor_grad = [None] * len(self.send_tensor_shapes)
            else:
                output_tensor_grad = self.recv_backward_buffer.pop(0)
                for h in output_tensor_grad[1]:
                    for hh in h:
                        hh.wait()
                output_tensor_grad = output_tensor_grad[0]
            if get_args().profile:
                torch.cuda.nvtx.range_push(f'B{scheduled_node.minibatch}')
            if self.run_timer:
                ScheduleTimers.for_chunk(0).b_cnt += 1
                ScheduleTimers.for_chunk(0).b.start()
                mem_before = torch.cuda.memory_allocated()
            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, self.model_type,
                self.config
            )
            if self.run_timer:
                ScheduleTimers.for_chunk(0).b.stop()
                ScheduleTimers.for_chunk(0).b_mem += torch.cuda.memory_allocated() - mem_before
            if get_args().profile:
                torch.cuda.nvtx.range_pop()
            self.send_backward_buffer.append(input_tensor_grad)
            WeightGradStore.flush()

    def schedule_w(self, scheduled_node, non_w_pending):
        if not self.forward_only and non_w_pending:
            if get_args().profile:
                torch.cuda.nvtx.range_push(f'W{scheduled_node.minibatch}')
            if self.run_timer:
                ScheduleTimers.for_chunk(0).w_cnt += 1
                ScheduleTimers.for_chunk(0).w.start()
            WeightGradStore.pop()
            if self.run_timer:
                ScheduleTimers.for_chunk(0).w.stop()
            if get_args().profile:
                torch.cuda.nvtx.range_pop()

    def disable_grad_sync(self):
        """Disable asynchronous grad reductions"""
        if self.no_sync_context is None:
            self.no_sync_context = self.no_sync_func()
            self.no_sync_context.__enter__()

    def enable_grad_sync(self):
        """Enable asynchronous grad reductions"""
        if self.no_sync_context is not None:
            self.no_sync_context.__exit__(None, None, None)
            self.no_sync_context = None

    def prepare(
        self,
        schedule: List[auto_schedule.ScheduledNode],
        forward_step_func,
        data_iterator: Union[Iterator, List[Iterator]],
        model: Union[torch.nn.Module, List[torch.nn.Module]],
        num_microbatches: int,
        seq_length: int,
        micro_batch_size: int,
        decoder_seq_length: int = None,
        forward_only: bool = False,
        collect_non_loss_data: bool = False,
    ):
        if isinstance(model, list):
            assert (
                len(model) == 1
            ), "non-interleaved pipeline parallelism does not support model chunking"
            model = model[0]
        if isinstance(data_iterator, list):
            assert (
                len(data_iterator) == 1
            ), "non-pipeline-parallel schedule does not support model chunking"
            data_iterator = data_iterator[0]

        config = get_model_config(model)
        if config.overlap_p2p_comm:
            raise ValueError(
                "Non-interleaved pipeline parallelism does not support overlapping p2p communication"
            )
        # Disable async grad reductions
        no_sync_func = config.no_sync_func
        if no_sync_func is None:
            no_sync_func = contextlib.nullcontext
        self.no_sync_func = no_sync_func
        self.no_sync_context = None
        if not forward_only:
            ScheduleTimers.iter_counter += 1

        # Checkpoint the activations of partial Transformer layers in a number of micro-batches
        # within the maximum outstanding micro-batch backpropagations.
        # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
        # checkpoint partial Transformer layers (or skip checkpointing) and
        # the rest of micro-batches within a window of micro-batches checkpoint
        # all Transformer layers. The window of micro-batches is set by the maximum
        # outstanding backpropagations and becomes smaller at later pipeline stages.
        # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
        assert config.num_microbatches_with_partial_activation_checkpoints is None

        model_type = get_model_type(model)

        rank = parallel_state.get_pipeline_model_parallel_rank()
        recv_tensor_shapes = get_tensor_shapes(
            rank=rank - 1,
            model_type=model_type,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            config=config,
        )
        send_tensor_shapes = get_tensor_shapes(
            rank=rank,
            model_type=model_type,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            config=config,
        )
        bootstrap_and_profile_p2p_communication(config, send_tensor_shapes,
                                                recv_tensor_shapes)

        run_timer = (
            get_args().zero_bubble_pipeline_timers_end_iter
            >= ScheduleTimers.iter_counter
            >= get_args().zero_bubble_pipeline_timers_start_iter
        )

        self.config = config
        self.model_type = model_type
        self.recv_tensor_shapes = recv_tensor_shapes
        self.send_tensor_shapes = send_tensor_shapes
        self.run_timer = run_timer
        self.schedules = schedule
        self.forward_step_func = forward_step_func
        self.data_iterator = data_iterator
        self.model = model
        self.num_microbatches = num_microbatches
        self.collect_non_loss_data = collect_non_loss_data
        self.forward_only = forward_only
        self._reset()
        self.it = 0

    def run_until_post_validation(self):
        optimizer = self.optimizer
        updated, grad_norm, rollback, succeed = None, None, None, None
        it = 0
        if optimizer.do_this_step:
            assert optimizer.do_prev_step
            if self.data_iterator is not None:
                self.data_iterator.clear_buffer()
                self.data_iterator.save_to_buffer()
            while it < len(self.schedules):
                scheduled_node = self.schedules[it]
                if scheduled_node.type in ["SEND_FORWARD", "RECV_FORWARD"]:
                    next_is_comm = it + 1 < len(self.schedules) and self.schedules[it + 1].type in AUTO_SCHEDULE_COMMUNICATION_TYPES
                    next_compute = list(filter(lambda x: x.type in ['F', 'B', 'W'], self.schedules[it + 1:]))
                    next_compute = next_compute[0] if len(next_compute) > 0 else None
                    self.add_communication(scheduled_node, next_is_comm, next_compute)
                elif scheduled_node.type == 'F':
                    self.schedule_f(scheduled_node)
                elif scheduled_node.type == "RECV_POST_VALIDATION":
                    optimizer.recv_post_validation()
                elif scheduled_node.type == "SEND_POST_VALIDATION":
                    optimizer.send_post_validation()
                elif scheduled_node.type == "POST_VALIDATION":
                    self.flush()
                    updated, grad_norm, rollback, succeed = optimizer.post_validation(self._free_buffers)
                    break
                else:
                    raise ValueError(f"Unexpected type {scheduled_node.type}")
                it += 1
            assert succeed is not None
        else:
            while it < len(self.schedules):
                scheduled_node = self.schedules[it]
                if scheduled_node.type in ["SEND_FORWARD", "RECV_FORWARD", "F"]:
                    if optimizer.do_prev_step and scheduled_node.type == "RECV_FORWARD":
                        next_is_comm = it + 1 < len(self.schedules) and self.schedules[it + 1].type in AUTO_SCHEDULE_COMMUNICATION_TYPES
                        next_compute = list(filter(lambda x: x.type in ['F', 'B', 'W'], self.schedules[it + 1:]))
                        next_compute = next_compute[0] if len(next_compute) > 0 else None
                        self.add_communication(scheduled_node, next_is_comm, next_compute)
                elif scheduled_node.type == "RECV_POST_VALIDATION":
                    optimizer.recv_post_validation()
                elif scheduled_node.type == "SEND_POST_VALIDATION":
                    optimizer.send_post_validation()
                elif scheduled_node.type == "POST_VALIDATION":
                    self.flush()
                    updated, grad_norm, rollback, succeed = optimizer.post_validation(self._free_buffers)
                    break
                else:
                    raise ValueError(f"Unexpected type {scheduled_node.type}")
                it += 1
            assert not succeed
        if not succeed:
            if optimizer.do_prev_step:
                # send dummy recv_forward to clear send_forward request of last rank
                while it < len(self.schedules):
                    scheduled_node = self.schedules[it]
                    if scheduled_node.type == "RECV_FORWARD" and scheduled_node.rollback:
                        self.add_communication(scheduled_node, False, None)
                    it += 1
            self._reset()
            it = 0
        if succeed and self.data_iterator is not None:
            self.data_iterator.clear_buffer()
        if self.data_iterator is not None:
            self.data_iterator.pop_from_buffer()
        self.it = it
        return updated, grad_norm, rollback

    def run(self):
        self.disable_grad_sync()

        if get_args().profile:
            torch.cuda.nvtx.range_push(f'iter_{torch.distributed.get_rank()}_{ScheduleTimers.iter_counter}')

        it = self.it
        while it < len(self.schedules):
            scheduled_node = self.schedules[it]
            if "POST_VALIDATION" in scheduled_node.type:
                pass
            elif scheduled_node.type in AUTO_SCHEDULE_COMMUNICATION_TYPES:
                next_is_comm = it + 1 < len(self.schedules) and self.schedules[it + 1].type in AUTO_SCHEDULE_COMMUNICATION_TYPES
                next_compute = list(filter(lambda x: x.type in ['F', 'B', 'W'], self.schedules[it + 1:]))
                next_compute = next_compute[0] if len(next_compute) > 0 else None
                self.add_communication(scheduled_node, next_is_comm, next_compute)
            elif scheduled_node.type == 'F':
                self.schedule_f(scheduled_node)
            elif scheduled_node.type == 'B':
                self.schedule_b(scheduled_node)
            elif scheduled_node.type == 'W':
                non_w_pending = any([node.type != 'W' for node in self.schedules[it + 1:]])
                self.schedule_w(scheduled_node, non_w_pending)
            else:
                raise ValueError(f"Unknown node type {scheduled_node.type}")
            it += 1
        self.it = it

        if get_args().profile:
            torch.cuda.nvtx.range_push('W')
        if not self.forward_only:
            pending_ws = WeightGradStore.weight_grad_queue[0].qsize()
            if self.run_timer:
                ScheduleTimers.for_chunk(0).w_cnt += pending_ws
                ScheduleTimers.for_chunk(0).w.start()
            WeightGradStore.clear(self.model)
            if self.run_timer:
                ScheduleTimers.for_chunk(0).w.stop()
        if get_args().profile:
            torch.cuda.nvtx.range_pop()  # W
            torch.cuda.nvtx.range_pop()  # iter

        for h in self.send_handles:
            for hh in h:
                for hhh in hh:
                    hhh.wait()

        if not self.forward_only:
            # Launch any remaining grad reductions
            if self.no_sync_context is not None:
                self.enable_grad_sync()

            if self.config.finalize_model_grads_func is not None:
                # Finalize model grads (perform full grad all-reduce / reduce-scatter for
                # data parallelism, layernorm all-reduce for sequence parallelism).
                self.config.finalize_model_grads_func([self.model])

            if get_args().zero_bubble_pipeline_timers_end_iter == ScheduleTimers.iter_counter:
                ScheduleTimers.concluded = True
        return self.forward_data_store

    def __call__(self, *args, **kwargs):
        if kwargs['forward_only']:
            self.prepare(*args, **kwargs)
            assert self.do_post_validation
            self.do_post_validation = True
            self.is_first_run = True
            return self.run()
        if not get_args().enable_optimizer_post_validation:
            self.prepare(*args, **kwargs)
            self.is_first_run = False
            self.do_post_validation = False
            return self.run()
        # enable_optimizer_post_validation == True
        if self.is_first_run:
            self.prepare(*args, **kwargs)
            self.is_first_run = False
            self.do_post_validation = False
        if self.do_post_validation:
            self.prepare(*args, **kwargs)
            result = self.run_until_post_validation()
            self.do_post_validation = False
        else:
            result = self.run()
            self.do_post_validation = True
        return result

zb_v_scheduler = ZeroBubbleVPipeScheduler()
zb_scheduler = ZeroBubbleScheduler()

def get_zb_scheduler_instance():
    if get_args().zero_bubble_v_schedule:
        global zb_v_scheduler
        return zb_v_scheduler
    else:
        global zb_scheduler
        return zb_scheduler


schedule = None
is_auto_schedule = False


def update_schedule(scheduler, f: List[int], b: List[int], w: List[int],
        c: int, f_mem: List[int], b_mem: List[int], w_mem: List[int],
        mem_limit: int):
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    ag_arguments = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(ag_arguments, (f,b,w,f_mem,b_mem,w_mem, mem_limit))
    assert len(ag_arguments) == torch.distributed.get_world_size()
    # Each value is an array of dimension (device, chunk)
    f,b,w,f_mem,b_mem,w_mem,mem_limit= zip(*ag_arguments)

    if is_second_last_pipeline_stage():
        print(f"rank {torch.distributed.get_rank()} Performing ILP with: f={f},\n b={b},\n w={w},\n c={c},\n f_mem={f_mem},\n b_mem={b_mem},\n w_mem={w_mem},\n mem_limit={mem_limit}")
        schedule = scheduler(
            pipeline_model_parallel_size,
            get_num_microbatches(),
            f, b, w,
            max(c, 1),
            f_mem, b_mem, w_mem,
            mem_limit,
        )
        ag_result = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(ag_result, schedule)
        
    else:
        ag_result = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(ag_result, None)
        schedule = list(filter(lambda x: x is not None, ag_result))
        assert len(schedule) == 1
        schedule = schedule[0]
    if torch.distributed.get_rank() == 0:
        print("updated schedule")
        for node in schedule[0]:
            print(f"{node.type}-{node.minibatch}", end=', ')
        print()
    return schedule


def get_zero_bubble_forward_backward_func():
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    assert pipeline_model_parallel_size > 1, "zero bubble must be enabled with pipeline parallelism"

    print_rank_0("@_@: get zero bubble forward backward func")
    args = get_args()
    hidden_size = args.hidden_size
    num_attention_heads = args.num_attention_heads
    seq_length = args.seq_length
    # TODO：确认正确性
    f_mem_approx = 34 * hidden_size + 5 * num_attention_heads * seq_length
    w_mem_approx = - 32 * hidden_size
    b_mem_approx = - f_mem_approx - w_mem_approx

    def wrapped_auto_schedule_forward_backward_func(func, scheduler):
        global schedule, is_auto_schedule
        # print_rank_0(f"log@_@: wrapped_auto_schedule_forward_backward_func, schedule: {schedule}, scheduler: {scheduler}")
        if schedule is None:
            schedule = update_schedule(scheduler,
                f=[1000],
                b=[1000],
                w=[1000],
                c=1,
                f_mem=[f_mem_approx],
                b_mem=[0],
                w_mem=[-f_mem_approx],
                mem_limit=f_mem_approx * parallel_state.get_pipeline_model_parallel_world_size())
                # Using fixed 1p schedule
        if ScheduleTimers.concluded and not is_auto_schedule:
            conclusion = ScheduleTimers.joint_conclusion()
            # TODO(wanxy): Maybe an all-reduce here to collect global stats?
            print(f"rank {torch.distributed.get_rank()} profiling conclusion: {conclusion}")
            def estimate_free_memory_on_this_rank(old_schedule):
                (memory_free, memory_all) = [x // 1000000 for x in torch.cuda.mem_get_info()]
                memory_all = memory_all * get_args().zero_bubble_adaptive_memory_limit_percentile / 100
                activation_cost = 0
                stage = parallel_state.get_pipeline_model_parallel_rank()
                max_activation = 0
                for node in old_schedule[stage]:
                    chunk = node.chunk if hasattr(node, 'chunk') else 0
                    if node.type == 'F':
                        activation_cost += conclusion[4][chunk]
                    elif node.type == 'B':
                        activation_cost += conclusion[5][chunk]
                    elif node.type == 'W':
                        activation_cost += conclusion[6][chunk]
                    max_activation = max(activation_cost, max_activation)
                free_mem = memory_all - (torch.cuda.max_memory_allocated() // 1000000 - max_activation)

                print(f'estimated max free memory for activations on rank {torch.distributed.get_rank()} \
                    memory_free: {memory_free}, memory_all: {memory_all}, max_activation: {max_activation}, \
                    max_allocated: {torch.cuda.max_memory_allocated() // 1000000}, \
                    current_allocated: {torch.cuda.memory_allocated() // 1000000}, \
                    free_mem: {free_mem}')

                print(f'rank {torch.distributed.get_rank()} mem summary {torch.cuda.memory_summary()}')
                return free_mem
            schedule = update_schedule(scheduler,
                *conclusion,
                mem_limit=estimate_free_memory_on_this_rank(schedule))
            is_auto_schedule = True

        def wrap_schedule(**kwargs):
            return func(
                schedule=schedule[parallel_state.get_pipeline_model_parallel_rank()], **kwargs
            )
            
        # if torch.distributed.get_rank() == 0:
        #     print(f"rank {torch.distributed.get_rank()} Using schedule: ")
        #     for node in schedule[parallel_state.get_pipeline_model_parallel_rank()]:
        #         print(f"{node.type}-{node.minibatch}", end=', ')
        #     print()
        return wrap_schedule

    if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
        def scheduler(nstages, nmb, f, b, w, c, f_mem, b_mem, w_mem, mem_limit):
            print_rank_0("log@_@: using virtualpp scheduler")
            def avg_then_mid(a: List[List[float]]):
                a = [sum(x) / len(x) for x in a]
                return max(sorted(a)[len(a) // 2], 1)
            # For V schedule, we take average on each stage and then use mid value cross each stage.
            f_mid = avg_then_mid(f)
            b_mid = avg_then_mid(b)
            w_mid = avg_then_mid(w)
            if get_args().zero_bubble_v_schedule_mem_setup != 'zb':
                # Use fixed schedule for now
                if torch.distributed.get_rank() == 2:
                    print(f"log@_@: Using fixed vzb schedule")
                ret = v_schedule_greedy.PipelineGraph(
                    nstages, nmb, get_args().zero_bubble_v_schedule_mem_setup, int(1000), int(1000), int(1000), int(1)
                ).get_schedule()
                return ret
            if torch.distributed.get_rank() == 2:
                print(f"log@_@: Using other vzb schedule")
            return v_schedule.PipelineGraph(
                nstages,
                nmb,
                f_mid,b_mid,w_mid,c,
                # V schedule does not consider memory differences between stages for now.
                f_mem=f_mem_approx, b_mem=b_mem_approx, w_mem=w_mem_approx,
                max_mem=None
                # Mem ignored for now
            ).get_v_schedule()
        if get_args().zero_bubble_v_schedule:
            if torch.distributed.get_rank() == 0:
                global schedule
                # print(f"log@_@: Using vzb schedule, schedule: {schedule}")
            global_zb_scheduler = get_zb_scheduler_instance()
            forward_backward_func = wrapped_auto_schedule_forward_backward_func(global_zb_scheduler, scheduler=scheduler)
            # forward_backward_func = wrapped_auto_schedule_forward_backward_func(forward_backward_pipelining_with_interleaving_auto_schedule,
            #                                                                     scheduler=scheduler)
        else:
            raise ValueError("got virtual pipeline parallel but v_schedule is disabled")
    else:
        def scheduler(nstages, nmb, f, b, w, c, f_mem, b_mem, w_mem, mem_limit):
            f = [x[0] for x in f]
            b = [x[0] for x in b]
            w = [x[0] for x in w]
            # Using uniform f/b/w timing for now.
            f = [sorted(f)[len(f) // 2]] * len(f)
            b = [sorted(b)[len(b) // 2]] * len(b)
            w = [sorted(w)[len(w) // 2]] * len(w)
            f_mem = [x[0] for x in f_mem]
            b_mem = [x[0] for x in b_mem]
            w_mem = [x[0] for x in w_mem]

            if args.zero_bubble_max_pending_backward != 'auto':
                print(f'manual mem limit: {args.zero_bubble_max_pending_backward * max(f_mem[:2])}')
                mem_limit = [args.zero_bubble_max_pending_backward * max(f_mem[:2])] * len(f_mem)
            else:
                print(f'adaptive mem limit: {mem_limit}')
            
            return auto_schedule.auto_schedule(
                nstages,
                nmb,
                auto_schedule.GraphConfig(
                    cost_f=f,
                    cost_b=b,
                    cost_w=w,
                    cost_comm=c,
                    mem_f=f_mem,
                    mem_b=b_mem,
                    mem_w=w_mem,
                    max_mem=mem_limit,
                    print_scaling=1000
                ),
            )
        if torch.distributed.get_rank() == 0:
            print(f"log@_@: Using zb schedule")
        global_zb_scheduler = get_zb_scheduler_instance()
        forward_backward_func = wrapped_auto_schedule_forward_backward_func(global_zb_scheduler, scheduler=scheduler)

    return forward_backward_func
