import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
from tensorflow.data.Dataset import from_generator
from vae import VAE

data_root=path.expanduser('/home/aitrading/Desktop/GLTransform/output/npy/')
np_files=glob(path.join(data_root,'*.npy'))  # list
mode = 'fully_connected'


def generate():
    for data in np_files:
        data=np.load(data)
        yield data

if mode == 'fully_connected':
    data_set = from_generator(generate, (tf.int32, ), output_shapes = tf.TensorShape([None,]))
elif mode == 'CNN':
    data_set = from_generator(generate, (tf.int32, tf.int32), output_shapes = tf.TensorShape([512,512]))

def trainer(model_object, learning_rate=1e-4, 
            batch_size=64, num_epoch=5, n_z=16, log_step=5, mode=mode, num_sample):
    model = model_object(learning_rate=learning_rate, batch_size=batch_size, n_z=n_z, mode= mode)

    for epoch in range(num_epoch):
        start_time = time.time()
        for i in range(num_sample // batch_size):
            # Get a batch
            batch = data_set.take(1)

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

def test_reconstruction(model, mnist, h=512, w=512, batch_size=16):
    # Test the trained model: reconstruction
    batch = data_set.take(1)
    x_reconstructed = model.reconstructor(batch[0])

    n = np.sqrt(batch_size).astype(np.int32)
    I_reconstructed = np.empty((h*n, 2*w*n))
    for i in range(n):
        for j in range(n):
            x = np.concatenate(
                (x_reconstructed[i*n+j, :].reshape(h, w), 
                 batch[0][i*n+j, :].reshape(h, w)),
                axis=1
            )
            I_reconstructed[i*h:(i+1)*h, j*2*w:(j+1)*2*w] = x

    plt.figure(figsize=(10, 20))
    plt.imshow(I_reconstructed, cmap='gray')

if __name__ == '__main__':
    trainer()