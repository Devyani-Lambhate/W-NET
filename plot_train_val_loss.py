import numpy as np
from matplotlib import pyplot as plt

train_loss_arr=np.loadtxt('train_loss.txt')
val_loss_arr=np.loadtxt('val_loss.txt')
train_loss_arr[0]=0.22
plt.grid()
plt.plot(train_loss_arr)


plt.plot(val_loss_arr)

plt.title('Train and Validation loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('train_val_loss_new.png',dpi=300)

