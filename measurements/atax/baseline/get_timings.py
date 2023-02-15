import json
import os
import numpy as np



filenames = os.listdir('/home/sarahtr/dace/measurements/atax/baseline')
filenames.remove("get_timings.py")
filenames.remove("v++_atax_kernel_0_0_hw.log")
print(filenames)
state = []
kernel = []

for file in filenames:
    path_file = '/home/sarahtr/dace/measurements/atax/baseline/' + file

    

    dur_state = 0
    dur_kernel = 0
    
    with open(path_file) as jfile:
        data = json.load(jfile)
        

        for event in data['traceEvents']:
            #print(event['name'])
            if 'Full FPGA kernel runtime' in event['name']:
                dur = event['dur']
                dur_kernel += dur

            if 'Full FPGA state runtime' in event['name']:
                dur = event['dur']
                dur_state += dur
        
    state.append(dur_state)
    kernel.append(dur_kernel)

print('state: ' + str(state))
print('kernel: ' + str(kernel))

    
