import pyopencl as cl
from pyopencl import array
import numpy as np
import matplotlib.pyplot as plt
import os
from OxygenPath import OxygenPath
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

steps = 1

platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
program = cl.Program(context, """
__kernel void run_msd(
__global const float *when_which_where,
__global const float *start_positions,
__global float *positions,
__global float *msd
)
{
    uint local_id = get_local_id(0);
    uint group_size = get_local_size(0);
    uint gid = get_global_id(0);


    __local float local_sums[16];
    local_sums[local_id] = positions[get_global_id(0)*3];

    for(uint stride = group_size/2;stride>0;stride/=2){
        barrier(CLK_LOCAL_MEM_FENCE);
        if(local_id < stride)
            local_sums[local_id] += local_sums[local_id + stride];
        if(local_id == 0){
            msd[get_group_id(0)] = local_sums[0];
            printf("\\n %d: %f", get_group_id(0), msd[get_group_id(0)]);
        }
    }
}
""").build()

oxygen_path = OxygenPath('atom_positions.xyz')
when_which_where = np.memmap('when_which_where.bin', dtype='float32', mode='r+', shape=(steps, 3))
positions = np.copy(oxygen_path.start_positions)
positions = np.pad(positions, ((0, 2**(oxygen_path.start_positions.shape[0]-1).bit_length()-oxygen_path.start_positions.shape[0]),
    (0, 0)), mode='constant', constant_values=0)
msd = np.zeros((steps, 4)).astype(np.float32)

print(np.sum(positions[:, 0]))

queue = cl.CommandQueue(context)
mem_flags = cl.mem_flags
when_which_where_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=when_which_where)
start_positions_buf = cl.Buffer(context,
                                mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR,
                                hostbuf=oxygen_path.start_positions)
positions_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=positions)
msd_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, msd.nbytes)

program.run_msd(queue,
                (2**(oxygen_path.start_positions.shape[0]-1).bit_length(), 1),
                (1<<4, 1),
                when_which_where_buf,
                start_positions_buf,
                positions_buf,
                msd_buf)

cl.enqueue_copy(queue, msd, msd_buf)

