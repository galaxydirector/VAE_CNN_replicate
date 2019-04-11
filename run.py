import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
from tensorflow.data.Dataset import from_generator

data_root=path.expanduser('/home/aitrading/Desktop/GLTransform/output/npy/')
np_files=glob(path.join(data_root,'*.npy'))  # list
mode = 'fully_connected'


def generate():
    for i,data in enumerate(np_files):
        data=np.load(data)
        yield data

if mode == 'fully_connected':
    ds = from_generator(generate, (tf.int32, ), (tf.TensorShape([]), ))

def trainer(model_object, learning_rate=1e-4, 
            batch_size=64, num_epoch=5, n_z=16, log_step=5, mode=mode, num_sample):
    model = model_object(learning_rate=learning_rate, batch_size=batch_size, n_z=n_z, mode= mode)

    for epoch in range(num_epoch):
        start_time = time.time()
        for i in range(num_sample // batch_size):
            # Get a batch
            batch = ds.take(1)

            # Execute the forward and backward pass 
            # Report computed losses
            losses = model.run_single_step(batch[0])
        end_time = time.time()
        
        if epoch % log_step == 0:
            log_str = '[Epoch {}] '.format(epoch)
            for k, v in losses.items():
                log_str += '{}: {:.3f}  '.format(k, v)
            log_str += '({:.3f} sec/epoch)'.format(end_time - start_time)
            print(log_str)
            
    print('Done!')
    return model

if __name__ == '__main__':
    trainer()