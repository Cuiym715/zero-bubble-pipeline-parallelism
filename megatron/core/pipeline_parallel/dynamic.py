import contextlib
import itertools
from typing import Iterator, List, Union

import torch

from megatron import core, get_args, get_num_microbatches, print_rank_0
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.utils import get_model_config, get_model_type
from megatron.core.pipeline_parallel import p2p_communication
from megatron.core.pipeline_parallel.schedules import (
    deallocate_output_tensor,
    forward_step,
    backward_step,
    get_tensor_shapes,
)
from megatron.core.weight_grad_store import WeightGradStore
from megatron.timers import Timer
import time


def w_step(config, model_chunk_id):
    raise NotImplementedError
    if config.timers is not None:
        config.timers('w-compute', log_level=2).start()
    log_file_name = get_args().timers_save+'/log%d.txt'%torch.distributed.get_rank()
    with open(log_file_name, 'a') as log_file:
        log_file.write(f"w_step, queue size of chunk {model_chunk_id}: {WeightGradStore.get_size(model_chunk_id)}\n")
    for _ in range(WeightGradStore.get_size(model_chunk_id)):
        WeightGradStore.pop(chunk=model_chunk_id)
    if not WeightGradStore.is_empty(chunk=model_chunk_id):
        raise Exception(f"rank {torch.distributed.get_rank()}, chunk id {model_chunk_id}, WeightGradStore is not empty.")
    if config.timers is not None:
        config.timers('w-compute').stop()
        
def wait_comm(handles, compute_stream, stored_activation_num, config):
    log_file_name = get_args().timers_save+'/log%d.txt'%torch.distributed.get_rank()
    compute_event = torch.cuda.Event(enable_timing=False, blocking=False)
    comm_stream = torch.cuda.Stream()
    for handle in handles:
        if handle is None or handle[0].is_completed():
            continue
        with open(log_file_name, 'a') as log_file:
            log_file.write(f"wait for handle, w queue size 1:{WeightGradStore.get_size(1)}, 0:{WeightGradStore.get_size(0)}\n")
        assert len(handle) == 1, "handle should be a list of one element"
        if config.timers is not None:
            config.timers('w-compute', log_level=2).start()
        with torch.cuda.stream(compute_stream):
            while not handle[0].is_completed():
                if WeightGradStore.get_size(1) > 0:
                    # with open(log_file_name, 'a') as log_file:
                    #     log_file.write(f"   w_step, chunk 1\n")
                    WeightGradStore.pop(1)
                    stored_activation_num -= 1
                    compute_event.record(compute_stream)
                elif WeightGradStore.get_size(0) > 0:
                    # with open(log_file_name, 'a') as log_file:
                    #     log_file.write(f"   w_step, chunk 0\n")
                    WeightGradStore.pop(0)
                    stored_activation_num -= 1
                    compute_event.record(compute_stream)
                else:
                    break
                if not compute_event.query():
                    compute_event.synchronize()
        if config.timers is not None:
            # config.timers('w-compute').stop(later_sync=True)
            config.timers('w-compute').stop()
        # 通信任务分配到 comm_stream
        with torch.cuda.stream(comm_stream):
            handle[0].wait() 
    if config.timers is not None:
        # config.timers('w-compute').syn()
        pass
    torch.cuda.synchronize()
    with open(log_file_name, 'a') as log_file:
        log_file.write(f"   done time: {time.time()}, w queue size 1:{WeightGradStore.get_size(1)}, 0:{WeightGradStore.get_size(0)}\n")
    return stored_activation_num


def check_memory(stored_activation_num, config):
    args = get_args()
    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    matmul_per_layer = 4 * args.micro_batch_size
    matmul_per_stage = args.num_layers / 2 / args.num_waves_per_pipeline / pipeline_parallel_size * matmul_per_layer
    if parallel_state.is_pipeline_last_stage():
        matmul_per_stage += 1
    memory_limit = args.num_layers * matmul_per_layer + 1
    
    log_file_name = get_args().timers_save+'/log%d.txt'%torch.distributed.get_rank()
    with open(log_file_name, 'a') as log_file:
        log_file.write(f"check memory, w queue size 1:{WeightGradStore.get_size(1)}, 0:{WeightGradStore.get_size(0)}\n")
    with open(log_file_name, 'a') as log_file:
        log_file.write(f"   stored_activation_num: {stored_activation_num}, matmul_per_stage: {matmul_per_stage}\n")        
        log_file.write(f"   memory limit: {memory_limit}\n")
    if stored_activation_num + matmul_per_stage <= memory_limit:
        return stored_activation_num
    with open(log_file_name, 'a') as log_file:
        log_file.write(f"   time: {time.time()}, w_step\n")    
    if config.timers is not None:
        config.timers('w-compute', log_level=2).start()
    while stored_activation_num + matmul_per_stage > memory_limit:
        if WeightGradStore.get_size(1) > 0:
            WeightGradStore.pop(1)
        elif WeightGradStore.get_size(0) > 0:
            WeightGradStore.pop(0)
        stored_activation_num -= 1
    if config.timers is not None:
        config.timers('w-compute').stop()
    # with open(log_file_name, 'a') as log_file:
    #     log_file.write(f"   done time: {time.time()}, w queue size 1:{WeightGradStore.get_size(1)}, 0:{WeightGradStore.get_size(0)}\n")
    return stored_activation_num

def forward_backward_pipelining_with_dynamicPP(
    *,
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
    """Run interleaved 1F1B schedule (model split into model chunks), with
    communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    assert isinstance(model, list), \
        "interleaved pipeline parallelism expected model chunking"
    assert all(isinstance(chunk, torch.nn.Module) for chunk in model), \
        "invalid model chunking"
    assert isinstance(data_iterator, list), \
        "interleaved pipeline parallelism expected each model chunk to have a data iterator"

    print_rank_0("DynamicPipe")
    
    config = get_model_config(model[0])
    if config.overlap_p2p_comm and config.batch_p2p_comm:
        raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if isinstance(no_sync_func, list):

        def multi_no_sync():
            stack = contextlib.ExitStack()
            for model_chunk_no_sync_func in config.no_sync_func:
                stack.enter_context(model_chunk_no_sync_func())
            return stack

        no_sync_func = multi_no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    if config.grad_sync_func is not None and not isinstance(config.grad_sync_func, list):
        config.grad_sync_func = [config.grad_sync_func for _ in model]

    if config.param_sync_func is not None and not isinstance(config.param_sync_func, list):
        config.param_sync_func = [config.param_sync_func for _ in model]

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()
    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None
    disable_grad_sync()

    # Model chunk IDs with synchronized grads
    synchronized_model_chunks = set()

    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    forward_data_store = []
    dependencies = {'forward':[], 'backward':[]}
   
    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]

    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
    
    model_chunk_ids = {i:0 for i in range(num_microbatches)}

    if num_microbatches % pipeline_parallel_size != 0:
        msg = f'number of microbatches ({num_microbatches}) is not divisible by '
        msg += f'pipeline-model-parallel-size ({pipeline_parallel_size}) '
        msg += 'when using interleaved schedule'
        raise RuntimeError(msg)

    model_type = get_model_type(model[0])
    if model_type == ModelType.encoder_and_decoder:
        raise RuntimeError("Interleaving is not supported with an encoder and decoder model.")

    if decoder_seq_length is not None and decoder_seq_length != tensor_shape[0]:
        raise RuntimeError("Interleaving is not supported with a different decoder sequence length.")

    tensor_shape = (seq_length, micro_batch_size, config.hidden_size)
    if config.sequence_parallel:
        tensor_shape[0] = tensor_shape[0] // parallel_state.get_tensor_model_parallel_world_size()
    rank = torch.distributed.get_rank()
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
    # wait_handles = []
    # wait_handles = []
    wait_handles = []
    compute_stream = torch.cuda.Stream(priority=-1)
    stored_activation_num = 0
    matmul_per_layer = 4 * micro_batch_size
    args = get_args()
    matmul_per_stage = args.num_layers / 2 / args.num_waves_per_pipeline / pipeline_parallel_size * matmul_per_layer

    # Compute number of warmup and remaining microbatches.
    num_model_chunks = len(model)
    total_num_microbatches = num_microbatches * num_model_chunks
    all_warmup_microbatches = False
  
    # TODO: 怎么没写forward_only
    # if forward_only:
    #     # num_warmup_microbatches = total_num_microbatches
    #     # num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches
    #     num_forward_warmup_microbatches = total_num_microbatches
    #     num_forward_steady_microbatches = 0
    #     num_microbatches_remaining = 0
    # else:
    assert num_microbatches % pipeline_parallel_size == 0, \
        "number of microbatches must be divisible by pipeline parallel size"
    # num_warmup_microbatches = num_model_chunks * pipeline_parallel_size - \
    #     (pipeline_parallel_rank - pipeline_parallel_rank)
    num_forward_warmup_microbatches = pipeline_parallel_size - pipeline_parallel_rank
    num_microbatches_remaining = pipeline_parallel_size - pipeline_parallel_rank
    num_forward_steady_microbatches = pipeline_parallel_size * num_model_chunks - num_forward_warmup_microbatches - num_microbatches_remaining
    num_iters = int(num_microbatches / pipeline_parallel_size)
    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    # TODO: 看hanayo能不能用这个
    # if config.num_microbatches_with_partial_activation_checkpoints is not None:
    #     max_outstanding_backprops = num_warmup_microbatches + 1
        
    # if torch.distributed.get_rank() == 0:
    # print(torch.distributed.get_rank(), 'num_warmup_microbatches', num_warmup_microbatches)
    rank = torch.distributed.get_rank()
    args = get_args()
    log_flag = True
    log_file_name = args.timers_save+'/log%d.txt'%rank
    if log_flag:
        with open(log_file_name, 'a') as log_file:
            log_file.write(f"--------------Dynamicpipe--------------------\n")
            log_file.write(f"pipeline_parallel_size: {pipeline_parallel_size}\n")
            log_file.write(f"pipeline_parallel_rank: {pipeline_parallel_rank}\n")
            log_file.write(f"""num microbatches\n
                        num_microbatches: {num_microbatches}\n
                        total_num_microbatches: {total_num_microbatches}\n
                        num_forward_warmup_microbatches: {num_forward_warmup_microbatches}\n
                        num_forward_steady_microbatches: {num_forward_steady_microbatches}\n
                        num_microbatches_remaining: {num_microbatches_remaining}\n
                        num_iters: {num_iters}\n""")
        
    # Synchronize params for first two model chunks
    if config.param_sync_func is not None:
        config.param_sync_func(model[0].parameters())
        config.param_sync_func(model[1].parameters())

    def get_model_chunk_id(microbatch_id, forward):
        if forward:
            model_chunk_id = model_chunk_ids[microbatch_id]
            model_chunk_ids[microbatch_id] += 1
        else:
            model_chunk_id = (num_model_chunks - model_chunk_id - 1)
        return model_chunk_id

    def is_first_microbatch_for_model_chunk(microbatch_id: int) -> bool:
        """Check if an iteration is the first for a model chunk."""
        microbatch_group_size = pipeline_parallel_size * num_model_chunks
        num_microbatch_groups = total_num_microbatches // microbatch_group_size
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        if microbatch_group_id == 0:
            return microbatch_id_in_group % pipeline_parallel_size == 0
        else:
            return False

    def is_last_microbatch_for_model_chunk(microbatch_id: int) -> bool:
        """Check if an iteration is the last for a model chunk."""
        microbatch_group_size = pipeline_parallel_size * num_model_chunks
        num_microbatch_groups = total_num_microbatches // microbatch_group_size
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        if microbatch_group_id == num_microbatch_groups - 1:
            return microbatch_id_in_group % pipeline_parallel_size == pipeline_parallel_size - 1
        else:
            return False
   
    memory = []
    memory.append((time.time(), torch.cuda.memory_allocated()))

    # Run warmup forward passes.
    parallel_state.set_virtual_pipeline_model_parallel_rank(0)
    # input_tensors[0].append(
    #     p2p_communication.recv_forward(tensor_shape,
    #                                    dtype=dtype,
    #                                    batch_p2p_comm=batch_p2p_comm,
    #                                    timers=timers))

    
    memory.append((time.time(), torch.cuda.memory_allocated()))
    
    # wavepipe参考了chimera包含两条流水线，需要分别记录modelID和batchID
    modelID1 = 0
    batchID1 = 0
    modelID2 = 1
    batchID2 = 0
    
    for k in range(num_forward_warmup_microbatches):

        model_chunk_id = modelID1
        microbatchID = batchID1
        batchID1 += 1
            
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)
        input_tensor, recv_handle = p2p_communication.recv_forward(tensor_shape, config, overlap_p2p_comm=True)    
        wait_handles.append(recv_handle)
        if not forward_only:
            input_tensors[model_chunk_id].append(input_tensor)
                
        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                k % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None        
        stored_activation_num = wait_comm(wait_handles, compute_stream, stored_activation_num, config)
        wait_handles = []
        stored_activation_num = check_memory(stored_activation_num, config)
        if log_flag:
            with open(log_file_name, 'a') as log_file:
                log_file.write(f"""time: {time.time()}, forward_step: batchID{microbatchID}, modelID{model_chunk_id}""")
                if input_tensor is not None:
                    log_file.write(f""", input_tensor: {input_tensor.shape}\n""")
                else:
                    log_file.write(f""", input_tensor: None\n""")
        output_tensor = forward_step(forward_step_func, data_iterator[model_chunk_id], 
                                     model[model_chunk_id], num_microbatches,
                                     input_tensor, forward_data_store,
                                     config, collect_non_loss_data, checkpoint_activations_microbatch,)
        if parallel_state.is_pipeline_last_stage():
            stored_activation_num += matmul_per_stage + 1
        else:
            stored_activation_num += matmul_per_stage
        # if get_dependency:
        #     dependencies['forward'].append((model_chunk_id, microbatchID))
        memory.append((time.time(), torch.cuda.memory_allocated()))
        
        if pipeline_parallel_rank == 0 and (model_chunk_id % 2) == 1:
            detached_output_tensor = output_tensor.detach()
            detached_output_tensor.requires_grad_()
            input_tensor=detached_output_tensor
        #     with open(log_file_name, 'a') as log_file:
        #         log_file.write(f"""time: {int(time.time())}, output tensor = {output_tensor.shape}, input tensor = {input_tensor.shape}\n""")
        elif pipeline_parallel_rank == (pipeline_parallel_size - 1) and (model_chunk_id % 2) == 0:
            detached_output_tensor = output_tensor.detach()
            detached_output_tensor.requires_grad_()
            input_tensor=detached_output_tensor
        #     with open(log_file_name, 'a') as log_file:
        #         log_file.write(f"""time: {int(time.time())}, output tensor = {output_tensor.shape}, input tensor = {input_tensor.shape}\n""")
        else:
            send_handle = p2p_communication.send_forward(output_tensor, config, overlap_p2p_comm=True)
            wait_handles.append(send_handle)
            # with open(log_file_name, 'a') as log_file:
            #     log_file.write(f"""time: {int(time.time())}, send forward\n""")

        if not forward_only:
            output_tensors[model_chunk_id].append(output_tensor)
        if isinstance(output_tensor, list):
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)
        else:
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
        
        if (batchID1 % pipeline_parallel_size) == 0:
            batchID1 -= pipeline_parallel_size
            modelID1 += 2      
    # parallel_state.set_virtual_pipeline_model_parallel_rank(modelID2)
    # if pipeline_parallel_rank != (pipeline_parallel_size - 1):
    #     with open(log_file_name, 'a') as log_file:
    #         log_file.write(f"   receiving from next stage\n")
    #     input_tensor = p2p_communication.recv_forward(tensor_shape, config)
    # input_tensors[modelID2].append(input_tensor)
    # with open(log_file_name, 'a') as log_file:
    #     log_file.write(f"   received %s\n"%str(input_tensor==None))
    
    for iter in range(num_iters):
        parallel_state.set_virtual_pipeline_model_parallel_rank(modelID2)
        if pipeline_parallel_rank != (pipeline_parallel_size - 1):
            # with open(log_file_name, 'a') as log_file:
            #     log_file.write(f"   receiving from next stage\n")
            input_tensor, recv_handle = p2p_communication.recv_forward(tensor_shape, config, overlap_p2p_comm=True)
            wait_handles.append(recv_handle)
        if log_flag:
            with open(log_file_name, 'a') as log_file:
                log_file.write(f"steady forward\n")
        for k in range(0, num_forward_steady_microbatches, 2):
            microbatchID = batchID2
            model_chunk_id = modelID2
            batchID2 += 1
            
            # TODO: Decide to checkpoint all layers' activations of the current micro-batch
            checkpoint_activations_microbatch = None
            parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)
            stored_activation_num = wait_comm(wait_handles, compute_stream, stored_activation_num, config)
            wait_handles = []
            stored_activation_num = check_memory(stored_activation_num, config)
            if log_flag:
                with open(log_file_name, 'a') as log_file:
                    log_file.write(f"""time: {time.time()}, forward_step: batchID{microbatchID}, modelID{model_chunk_id}""")
                    if input_tensor is not None:
                        log_file.write(f""", input_tensor: {input_tensor.shape}\n""")
                    else:
                        log_file.write(f""", input_tensor: None\n""")
            output_tensor = forward_step(forward_step_func, data_iterator[model_chunk_id],
                                        model[model_chunk_id], num_microbatches,
                                        input_tensor, forward_data_store,
                                        config, collect_non_loss_data, checkpoint_activations_microbatch,)
            if parallel_state.is_pipeline_last_stage():
                stored_activation_num += matmul_per_stage + 1
            else:
                stored_activation_num += matmul_per_stage
            if not forward_only:
                input_tensors[model_chunk_id].append(input_tensor)
                output_tensors[model_chunk_id].append(output_tensor)
            # if get_dependency:
            #     dependencies['forward'].append((model_chunk_id, microbatchID))
            memory.append((time.time(), torch.cuda.memory_allocated()))
            
            if pipeline_parallel_rank != 0:
                # with open(log_file_name, 'a') as log_file:
                #     log_file.write(f"   receiving from prev stage\n")
                parallel_state.set_virtual_pipeline_model_parallel_rank(modelID1)
                input_tensor, recv_handle = p2p_communication.recv_forward(tensor_shape, config, overlap_p2p_comm=True) 
                wait_handles.append(recv_handle)
                
                # with open(log_file_name, 'a') as log_file:
                #     log_file.write(f"   received {input_tensor.shape}\n")
                # with open(log_file_name, 'a') as log_file:
                #     log_file.write(f"   sending to prev stage {output_tensor.shape}\n")
                parallel_state.set_virtual_pipeline_model_parallel_rank(modelID2)
                send_handle = p2p_communication.send_forward(output_tensor, config, overlap_p2p_comm=True)
                wait_handles.append(send_handle)
                # with open(log_file_name, 'a') as log_file:
                #     log_file.write(f"   sent %s\n"%str(output_tensor))
            else:
                detached_output_tensor = output_tensor.detach()
                detached_output_tensor.requires_grad_()
                input_tensor=detached_output_tensor
            # input_tensors[modelID1].append(input_tensor)
            # output_tensors[modelID2].append(output_tensor)
            if isinstance(output_tensor, list):
                deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)
            else:
                deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
            
            # microbatchID = orderedID
            # orderedID = (orderedID + 1) % num_microbatches
            # model_chunk_id = get_model_chunk_id(microbatchID, forward=True)
            microbatchID = batchID1
            model_chunk_id = modelID1
            batchID1 += 1
                    
            # TODO: Decide to checkpoint all layers' activations of the current micro-batch
            checkpoint_activations_microbatch = None
            parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)
            stored_activation_num = wait_comm(wait_handles, compute_stream, stored_activation_num, config)
            wait_handles = []
            stored_activation_num = check_memory(stored_activation_num, config)
            if log_flag:
                with open(log_file_name, 'a') as log_file:
                    log_file.write(f"""time: {time.time()}, forward_step: batchID{microbatchID}, modelID{model_chunk_id}\n""")
            output_tensor = forward_step(forward_step_func, data_iterator[model_chunk_id], 
                                        model[model_chunk_id], num_microbatches,
                                        input_tensor, forward_data_store,
                                        config, collect_non_loss_data, checkpoint_activations_microbatch)
            if parallel_state.is_pipeline_last_stage():
                stored_activation_num += matmul_per_stage + 1
            else:
                stored_activation_num += matmul_per_stage
            # if get_dependency:
            #     dependencies['forward'].append((model_chunk_id, microbatchID))
            if not forward_only:
                input_tensors[model_chunk_id].append(input_tensor)    
                output_tensors[model_chunk_id].append(output_tensor)
            memory.append((time.time(), torch.cuda.memory_allocated()))
            
            
            if pipeline_parallel_rank != (pipeline_parallel_size - 1):
                # with open(log_file_name, 'a') as log_file:
                #     log_file.write(f"   sending to next stage %s\n"%(str(output_tensor==None)))
                send_handle = p2p_communication.send_forward(output_tensor, config, overlap_p2p_comm=True)
                wait_handles.append(send_handle)
                # with open(log_file_name, 'a') as log_file:
                #     log_file.write(f"   receiving from next stage\n")
                parallel_state.set_virtual_pipeline_model_parallel_rank(modelID2)
                input_tensor, recv_handle = p2p_communication.recv_forward(tensor_shape, config, overlap_p2p_comm=True)             
                wait_handles.append(recv_handle)
            else:
                detached_output_tensor = output_tensor.detach()
                detached_output_tensor.requires_grad_()
                input_tensor=detached_output_tensor
            # input_tensors[modelID2].append(input_tensor)
            # output_tensors[modelID1].append(output_tensor)
            if isinstance(output_tensor, list):
                deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)
            else:
                deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
            # with open(log_file_name, 'a') as log_file:
            #     log_file.write(f"   received %s\n"%str(input_tensor==None))
                
            if (batchID1 % pipeline_parallel_size) == 0:
                batchID1 -= pipeline_parallel_size
                modelID1 += 2
            if (batchID2 % pipeline_parallel_size) == 0 :
                batchID2 -= pipeline_parallel_size
                modelID2 += 2     

        back_modelID1 = len(model) - 2
        back_batchID1 = pipeline_parallel_rank * iter
        back_modelID2 = len(model) - 1
        back_batchID2 = pipeline_parallel_rank * iter
        # with open(log_file_name, 'a') as log_file:
        #     log_file.write(f"---input tensors---\n")
        #     for l in range(len(input_tensors)):
        #         log_file.write(f"%d, len=%d\n"%(l, len(input_tensors[l])))
        #     log_file.write(f"------\n")
            
        if log_flag:
            with open(log_file_name, 'a') as log_file:
                log_file.write(f"1f1b\n")
        for k in range(num_microbatches_remaining):
            microbatchID = batchID2
            model_chunk_id = modelID2
            batchID2 += 1
            
            # model_chunk_id = get_model_chunk_id(microbatchID, forward=True)
            
            # input_tensors[model_chunk_id].append(input_tensor)
            
            # TODO: Decide to checkpoint all layers' activations of the current micro-batch
            checkpoint_activations_microbatch = None
            parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)
            stored_activation_num = wait_comm(wait_handles, compute_stream, stored_activation_num, config)
            wait_handles = []
            stored_activation_num = check_memory(stored_activation_num, config)
            if log_flag:
                with open(log_file_name, 'a') as log_file:
                    log_file.write(f"""time: {time.time()}, forward_step: batchID{microbatchID}, modelID{model_chunk_id}""")
                    if input_tensor is not None:
                        log_file.write(f""", input_tensor: {input_tensor.shape}\n""")
                    else:
                        log_file.write(f""", input_tensor: None\n""")
            output_tensor = forward_step(forward_step_func, data_iterator[model_chunk_id],
                                            model[model_chunk_id], num_microbatches,
                                            input_tensor, forward_data_store,
                                            config, collect_non_loss_data, checkpoint_activations_microbatch)
            if parallel_state.is_pipeline_last_stage():
                stored_activation_num += matmul_per_stage + 1
            else:
                stored_activation_num += matmul_per_stage
            if not forward_only:
                input_tensors[model_chunk_id].append(input_tensor)
                output_tensors[model_chunk_id].append(output_tensor)
            # if get_dependency:
            #     dependencies['forward'].append((model_chunk_id, microbatchID))
            memory.append((time.time(), torch.cuda.memory_allocated()))
            
            # with open(log_file_name, 'a') as log_file:
            #     log_file.write(f"   sending & receiving with pre stage\n ")
            if pipeline_parallel_rank == 0:
                output_tensor_grad = None
            else:
                # with open(log_file_name, 'a') as log_file:
                #     log_file.write(f"   sending %s\n"%str(output_tensor==None))
                send_handle = p2p_communication.send_forward(output_tensor, config, overlap_p2p_comm=True)
                wait_handles.append(send_handle)
                if not forward_only:
                    # with open(log_file_name, 'a') as log_file:
                    #     log_file.write(f"   receiving\n")
                    parallel_state.set_virtual_pipeline_model_parallel_rank(back_modelID2)
                    output_tensor_grad, recv_handle = p2p_communication.recv_backward(tensor_shape, config, overlap_p2p_comm=True)
                    wait_handles.append(recv_handle)
            # if not forward_only:
            #     with open(log_file_name, 'a') as log_file:
            #         log_file.write(f"   received %s\n"%(output_tensor_grad==None))
                
            # output_tensors[modelID2].append(output_tensor)
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
            # output_tensor_grads[back_modelID2].append(output_tensor_grad)
            
            if not forward_only:
                microbatchID = back_batchID2
                model_chunk_id = back_modelID2
                back_batchID2 += 1
                
                input_tensor = input_tensors[back_modelID2].pop(0)
                output_tensor = output_tensors[back_modelID2].pop(0)
                
                parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)
                stored_activation_num = wait_comm(wait_handles, compute_stream, stored_activation_num, config)
                wait_handles = []
                if log_flag:
                    with open(log_file_name, 'a') as log_file:
                        log_file.write(f"""time: {time.time()}, backward_step: batchID{microbatchID}, modelID{model_chunk_id}\n""")
                input_tensor_grad = backward_step(
                    input_tensor, output_tensor, output_tensor_grad,
                    model_type, config
                )
                WeightGradStore.flush_as_matmul(chunk=model_chunk_id)
                # with open(log_file_name, 'a') as log_file:
                #     log_file.write(f"   w queue size 1:{WeightGradStore.get_size(1)}, 0:{WeightGradStore.get_size(0)}\n")
                # w_step(config, model_chunk_id)
                # if get_dependency:
                #     dependencies['backward'].append((model_chunk_id, microbatchID))
                # with open(log_file_name, 'a') as log_file:
                #     log_file.write(f"   input tensor grad %s\n"%str(input_tensor_grad))
            
                if k == (num_microbatches_remaining - 1):
                    if pipeline_parallel_rank != (pipeline_parallel_size - 1):
                        send_handle = p2p_communication.send_backward(input_tensor_grad, config, overlap_p2p_comm=True)
                        wait_handles.append(send_handle)
                        parallel_state.set_virtual_pipeline_model_parallel_rank(back_modelID1)
                        output_tensor_grad, recv_handle = p2p_communication.recv_backward(tensor_shape, config, overlap_p2p_comm=True)
                        wait_handles.append(recv_handle)
                    else:
                        output_tensor_grad = input_tensor_grad
                    # with open(log_file_name, 'a') as log_file:
                    #     log_file.write(f"   received output tensor grad %s\n"%str(output_tensor_grad))
                else:
                    # with open(log_file_name, 'a') as log_file:
                    #     log_file.write(f"   sending & receiving with next stage\n")
                    parallel_state.set_virtual_pipeline_model_parallel_rank(modelID2)
                    input_tensor, recv_handle = p2p_communication.recv_forward(tensor_shape, config, overlap_p2p_comm=True)
                    wait_handles.append(recv_handle)
                    parallel_state.set_virtual_pipeline_model_parallel_rank(back_modelID2)
                    send_handle = p2p_communication.send_backward(input_tensor_grad, config, overlap_p2p_comm=True)
                    wait_handles.append(send_handle)
                    # input_tensors[modelID2].append(input_tensor)
                    # with open(log_file_name, 'a') as log_file:
                    #     log_file.write(f"   received %s\n"%(input_tensor==None))
                    #     if isinstance(input_tensor, list):
                    #         log_file.write(f"   input tensor %s %d\n"%(str(input_tensor[0]==None), len(input_tensor)))
            else:
                if k != (num_microbatches_remaining - 1):
                    # with open(log_file_name, 'a') as log_file:
                    #     log_file.write(f"   receiving %s\n")
                    parallel_state.set_virtual_pipeline_model_parallel_rank(modelID2)
                    input_tensor, recv_handle = p2p_communication.recv_forward(tensor_shape, config, overlap_p2p_comm=True)
                    wait_handles.append(recv_handle)
            if (batchID2 % pipeline_parallel_size) == 0 :
                batchID2 -= pipeline_parallel_size
                modelID2 += 2
            if (back_batchID2 % pipeline_parallel_size) == 0 :
                back_batchID2 -= pipeline_parallel_size
                back_modelID2 -= 2
                
        if not forward_only:
            if log_flag:
                with open(log_file_name, 'a') as log_file:
                    log_file.write(f"steady backward\n")    

            for k in range(0, num_forward_steady_microbatches, 2):
                microbatchID = back_batchID1
                model_chunk_id = back_modelID1
                back_batchID1 += 1
                # model_chunk_id = get_model_chunk_id(k+num_forward_warmup_microbatches, forward=True)
                                
                input_tensor = input_tensors[model_chunk_id].pop(0)
                output_tensor = output_tensors[model_chunk_id].pop(0)  
                
                parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)
                stored_activation_num = wait_comm(wait_handles, compute_stream, stored_activation_num, config)
                wait_handles = []
                if log_flag:
                    with open(log_file_name, 'a') as log_file:
                        log_file.write(f"""time: {time.time()}, backward_step: batchID{microbatchID}, modelID{model_chunk_id}\n""")
                input_tensor_grad = backward_step(
                    input_tensor, output_tensor, output_tensor_grad,
                    model_type, config
                )
                WeightGradStore.flush_as_matmul(chunk=model_chunk_id)
                    # w_step(config, model_chunk_id)
                # if get_dependency:
                #     dependencies['backward'].append((model_chunk_id, microbatchID))                                    
                memory.append((time.time(), torch.cuda.memory_allocated()))
                
                if pipeline_parallel_rank != 0:
                    # with open(log_file_name, 'a') as log_file:
                    #     log_file.write(f"   receiving %s\n")
                    parallel_state.set_virtual_pipeline_model_parallel_rank(back_modelID2)
                    output_tensor_grad, recv_handle = \
                        p2p_communication.recv_backward(tensor_shape, config, overlap_p2p_comm=True)
                    wait_handles.append(recv_handle)
                    # with open(log_file_name, 'a') as log_file:
                    #     log_file.write(f"   received %s\n"%(output_tensor_grad==None))
                    #     log_file.write(f"   sending %s\n"%(input_tensor_grad==None))
                    parallel_state.set_virtual_pipeline_model_parallel_rank(back_modelID1)
                    send_handle = p2p_communication.send_backward(input_tensor_grad, config, overlap_p2p_comm=True)
                    wait_handles.append(send_handle)
                else:
                    output_tensor_grad = input_tensor_grad
                
                microbatchID = back_batchID2
                model_chunk_id = back_modelID2
                back_batchID2 += 1
                
                input_tensor = input_tensors[model_chunk_id].pop(0)
                output_tensor = output_tensors[model_chunk_id].pop(0)
                
                parallel_state.set_virtual_pipeline_model_parallel_rank(back_modelID2)
                stored_activation_num = wait_comm(wait_handles, compute_stream, stored_activation_num, config)
                wait_handles = []
                if log_flag:
                    with open(log_file_name, 'a') as log_file:
                        log_file.write(f"""time: {time.time()}, backward_step: batchID{microbatchID}, modelID{model_chunk_id}\n""")
                input_tensor_grad = backward_step(
                    input_tensor, output_tensor, output_tensor_grad,
                    model_type, config
                ) 
                WeightGradStore.flush_as_matmul(chunk=model_chunk_id)
                # if get_dependency:
                #     dependencies['backward'].append((model_chunk_id, microbatchID))
                if pipeline_parallel_rank != (pipeline_parallel_size - 1):
                    # with open(log_file_name, 'a') as log_file:
                    #     log_file.write(f"   sending %s\n"%(input_tensor_grad==None))
                    send_handle = p2p_communication.send_backward(input_tensor_grad, config, overlap_p2p_comm=True)
                    wait_handles.append(send_handle)
                    # with open(log_file_name, 'a') as log_file:
                    #     log_file.write(f"   receiving %s\n")
                    parallel_state.set_virtual_pipeline_model_parallel_rank(back_modelID1)
                    output_tensor_grad, recv_handle = p2p_communication.recv_backward(tensor_shape, config, overlap_p2p_comm=True)
                    wait_handles.append(recv_handle)
                else:
                    output_tensor_grad = input_tensor_grad
                # with open(log_file_name, 'a') as log_file:
                #         log_file.write(f"   received %s\n"%(output_tensor_grad==None))
                if (back_batchID1 % pipeline_parallel_size) == 0 :
                    back_batchID1 -= pipeline_parallel_size
                    back_modelID1 -= 2
                if (back_batchID2 % pipeline_parallel_size) == 0 :
                    back_batchID2 -= pipeline_parallel_size
                    back_modelID2 -= 2
        
        if iter == num_iters - 1:
            break
                    
        if log_flag:
            with open(log_file_name, 'a') as log_file:
                log_file.write(f"next iteration\n")    
        modelID1 = 0
        batchID1 += pipeline_parallel_size
        modelID2 = 1
        batchID2 += pipeline_parallel_size
        if log_flag:
            with open(log_file_name, 'a') as log_file:
                log_file.write(f"modelID1: {modelID1}, batchID1: {batchID1}\n")
                log_file.write(f"modelID2: {modelID2}, batchID2: {batchID2}\n")
                log_file.write(f"back_modelID1: {back_modelID1}, back_batchID1: {back_batchID1}\n")
                log_file.write(f"back_modelID2: {back_modelID2}, back_batchID2: {back_batchID2}\n")
        for k in range(num_forward_warmup_microbatches):
            if not forward_only:
                microbatchID = back_batchID1
                model_chunk_id = back_modelID1
                back_batchID1 += 1
                
                input_tensor = input_tensors[model_chunk_id].pop(0)
                output_tensor = output_tensors[model_chunk_id].pop(0)
                
                parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)
                stored_activation_num = wait_comm(wait_handles, compute_stream, stored_activation_num, config)
                wait_handles = []
                if log_flag:
                    with open(log_file_name, 'a') as log_file:
                        log_file.write(f"""time: {time.time()}, backward_step: batchID{microbatchID}, modelID{model_chunk_id}\n""")
                input_tensor_grad = backward_step(
                    input_tensor, output_tensor, output_tensor_grad,
                    model_type, config
                )
                WeightGradStore.flush_as_matmul(chunk=model_chunk_id)
                # if get_dependency:
                #     dependencies['backward'].append((model_chunk_id, microbatchID))
                send_handle = p2p_communication.send_backward(input_tensor_grad, config, overlap_p2p_comm=True)
                wait_handles.append(send_handle)
                
            model_chunk_id = modelID1
            microbatchID = batchID1
            batchID1 += 1
    
            parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)
            input_tensor, recv_handle = p2p_communication.recv_forward(tensor_shape, config, overlap_p2p_comm=True)    
            wait_handles.append(recv_handle)
            if not forward_only:
                input_tensors[model_chunk_id].append(input_tensor)
                    
            # TODO: Decide to checkpoint all layers' activations of the current micro-batch
            checkpoint_activations_microbatch = None     
            stored_activation_num = wait_comm(wait_handles, compute_stream, stored_activation_num, config)   
            wait_handles = []
            stored_activation_num = check_memory(stored_activation_num, config)
            if log_flag:
                with open(log_file_name, 'a') as log_file:
                    log_file.write(f"""time: {time.time()}, forward_step: batchID{microbatchID}, modelID{model_chunk_id}""")
                    if input_tensor is not None:
                        log_file.write(f""", input_tensor: {input_tensor.shape}\n""")
                    else:
                        log_file.write(f""", input_tensor: None\n""")
            output_tensor = forward_step(forward_step_func, data_iterator[model_chunk_id], 
                                        model[model_chunk_id], num_microbatches,
                                        input_tensor, forward_data_store,
                                        config, collect_non_loss_data, checkpoint_activations_microbatch,)
            if parallel_state.is_pipeline_last_stage():
                stored_activation_num += matmul_per_stage + 1
            else:
                stored_activation_num += matmul_per_stage
            memory.append((time.time(), torch.cuda.memory_allocated()))
            
            if not forward_only:
                parallel_state.set_virtual_pipeline_model_parallel_rank(back_modelID1)
                if k != (num_forward_warmup_microbatches-1):
                    output_tensor_grad, bwd_wait_handles = p2p_communication.\
                        recv_backward(tensor_shape, config, overlap_p2p_comm=True)
            
            parallel_state.set_virtual_pipeline_model_parallel_rank(modelID1)
            if pipeline_parallel_rank == 0 and (model_chunk_id % 2) == 1:
                detached_output_tensor = output_tensor.detach()
                detached_output_tensor.requires_grad_()
                input_tensor=detached_output_tensor
            #     with open(log_file_name, 'a') as log_file:
            #         log_file.write(f"""time: {int(time.time())}, output tensor = {output_tensor.shape}, input tensor = {input_tensor.shape}\n""")
            elif pipeline_parallel_rank == (pipeline_parallel_size - 1) and (model_chunk_id % 2) == 0:
                detached_output_tensor = output_tensor.detach()
                detached_output_tensor.requires_grad_()
                input_tensor=detached_output_tensor
            #     with open(log_file_name, 'a') as log_file:
            #         log_file.write(f"""time: {int(time.time())}, output tensor = {output_tensor.shape}, input tensor = {input_tensor.shape}\n""")
            else:
                send_handle = p2p_communication.send_forward(output_tensor, config, overlap_p2p_comm=True)
                wait_handles.append(send_handle)
                # with open(log_file_name, 'a') as log_file:
                #     log_file.write(f"""time: {int(time.time())}, send forward\n""")

            if not forward_only:
                output_tensors[model_chunk_id].append(output_tensor)
            if isinstance(output_tensor, list):
                deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)
            else:
                deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
            
            if (batchID1 % pipeline_parallel_size) == 0:
                batchID1 -= pipeline_parallel_size
                modelID1 += 2  
            if not forward_only:
                if (back_batchID1 % pipeline_parallel_size) == 0:
                    back_batchID1 -= pipeline_parallel_size
                    back_modelID1 -= 2         
            
                    
    if not forward_only:    
              
        for k in range(num_forward_warmup_microbatches):
            microbatchID = back_batchID1
            model_chunk_id = back_modelID1
            back_batchID1 += 1
            
            input_tensor = input_tensors[model_chunk_id].pop(0)
            output_tensor = output_tensors[model_chunk_id].pop(0)
            
            
            parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)
            if pipeline_parallel_rank != (pipeline_parallel_size - 1):
                stored_activation_num = wait_comm(wait_handles, compute_stream, stored_activation_num, config)
                wait_handles = []
                with open(log_file_name, 'a') as log_file:
                    log_file.write(f"""time: {time.time()}, backward_step: batchID{microbatchID}, modelID{model_chunk_id}\n""")
                input_tensor_grad = backward_step(
                    input_tensor, output_tensor, output_tensor_grad,
                    model_type, config
                )
                WeightGradStore.flush_as_matmul(chunk=model_chunk_id)
                # w_step(config, model_chunk_id)
            # if get_dependency:
            #     dependencies['backward'].append((model_chunk_id, microbatchID))
            send_handle = p2p_communication.send_backward(input_tensor_grad, config, overlap_p2p_comm=True)
            wait_handles.append(send_handle)
            if k != (num_forward_warmup_microbatches-1):
                output_tensor_grad, bwd_wait_handles = p2p_communication.\
                    recv_backward(tensor_shape, config, overlap_p2p_comm=True)
        
        if config.timers is not None:
            config.timers('w-compute', log_level=2).start()  
        # WeightGradStore.clear(model[model_chunk_id], chunk=model_chunk_id)
        for _ in range(WeightGradStore.get_size(1)):
            WeightGradStore.pop(1)
        for _ in range(WeightGradStore.get_size(0)):
            WeightGradStore.pop(0)
        if config.timers is not None:
            config.timers('w-compute').stop()
        
    enable_grad_sync()
    if config.grad_sync_func is not None:
        params = []
        for model_chunk_id in range(num_model_chunks):
            if model_chunk_id not in synchronized_model_chunks:
                params.extend(model[model_chunk_id].parameters())
                synchronized_model_chunks.add(model_chunk_id)
        if params:
            config.grad_sync_func(params)
            
    if config.finalize_model_grads_func is not None and not forward_only:
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func(model)
            
    memory.append((time.time(), torch.cuda.memory_allocated()))
    
    # args = get_args()
    # record_path = args.save+'/memorylog%d.json'%torch.distributed.get_rank()
    # with open(record_path, 'r') as file:
    #     data = json.load(file)
    # data['memory'].extend(memory)
    # with open(record_path, 'w') as json_file:
    #     json.dump(data, json_file, indent=4)
    
    for i in range(len(input_tensors)):
        assert len(input_tensors[i]) == 0,\
            f"input_tensors[{i}] is not empty"
    for i in range(len(output_tensors)):
        assert len(output_tensors[i]) == 0,\
            f"output_tensors[{i}] is not empty"

    with open(log_file_name, 'a') as log_file:
        log_file.write(f"---forward_data_store---\n")
        log_file.write(f"{len(forward_data_store)}\n")
        log_file.write(f"------\n")
    
    return forward_data_store