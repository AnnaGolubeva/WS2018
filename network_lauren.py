#####################################################################################
### Code by Lauren Hayward Sierens
#####################################################################################

from   itertools import product
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf

seed=1234
np.random.seed(seed)
random.seed(seed)
tf.set_random_seed(seed)

plt.ion() #

def step(u):
  out = np.array([1.0,0]) #0
  
  if u>0:
    out = np.array([0,1.0]) #1
  return out

def psi(u,gam):
  out = 0
  if -gam*u<700:
    out = 1.0/ (1.0 + np.exp(-gam*u))
  return out

theta = 34.0
gamma = 30.5

##############################################################################
############################## READ IN THE DATA ##############################
##############################################################################

N_data=2**12
n_neurons = [12,10,7,5,4,3,2]

##ANNA'S CODE FOR LABELLING:
#x_data = np.load('toy_dataset/configs.npy')
#y_data = [None for i in range(N_data)]
#y_orig_data      = np.load('toy_dataset/labels.npy')
#
## convert labels into 1-hot representation
## y=0 --> (1,0)
## y=1 --> (0,1)
#num_y1 = 0
#for (i,yy) in enumerate(y_orig_data):
#  if yy==0:
#    y_data[i] = np.array([1.0,0])
#  else:
#    y_data[i] = np.array([0,1.0])
#    num_y1 = num_y1 + 1

x_data = np.zeros((N_data,n_neurons[0]))
y_data = [None for i in range(N_data)]
fin = open('config_labels.txt', 'r')
count = 0
num_y1 = 0
for line in fin:
  tokens = line.split()
  c_str = tokens[0]
  
  for i in range(len(c_str)):
    x_data[count,i] = int(c_str[i])
  
  fx = int(tokens[1])
  #y_data[count] = step( fx - theta )
  p_y1condx = psi( fx - theta, gamma )
  if random.random() < p_y1condx:
    y_data[count] = np.array([0,1.0])
  else:
    y_data[count] = np.array([1.0,0])
    #num_y1 = num_y1 + 1
  if step( fx - theta )[0] == 1:
      num_y1 = num_y1 + 1
                      
  count = count + 1
fin.close()

print("Num with y = 1: %d out of %d" %(num_y1,len(y_data)))
y_data = np.array(y_data)
        
N_train = 3400
train_indices = random.sample(range(N_data),N_train)
test_indices = [ x for x in range(N_data) if x not in train_indices]

x_train = np.array( [x_data[i] for i in train_indices] )
y_train = np.array( [y_data[i] for i in train_indices] )
x_test  = np.array( [x_data[i] for i in test_indices] )
y_test  = np.array( [y_data[i] for i in test_indices] )

##############################################################################
######################### IMPLEMENTING THE REGRESSION ########################
##############################################################################

x = tf.placeholder(tf.float32, [None,n_neurons[0]])

W   = [None for i in range(len(n_neurons)-1)]
b   = [None for i in range(len(n_neurons)-1)]
out = [None for i in range(len(n_neurons)-2)]
for i in range(len(W)):
  W[i] = tf.Variable(tf.truncated_normal([n_neurons[i],n_neurons[i+1]],stddev=0.1))
  b[i] = tf.Variable(tf.truncated_normal([n_neurons[i+1]],stddev=0.1))

  if i==0: #first layer
    out[i] = tf.tanh( tf.matmul(x,W[i])+b[i] )
  elif i<(len(W)-1):
    out[i] = tf.tanh( tf.matmul(out[i-1],W[i])+b[i] )
  else: #output layer
    y = tf.matmul(out[i-1],W[i])+b[i]

##############################################################################
################################## TRAINING ##################################
##############################################################################

y_ = tf.placeholder(tf.float32, [None, n_neurons[-1]])
cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_,logits=y))
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
correct_prediction = tf.equal(tf.argmax(tf.sigmoid(y),1), tf.argmax(y_,1)) #list of booleans
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#l_rate=0.01 #learning rate
#train_step = tf.train.GradientDescentOptimizer(l_rate).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(l_rate).minimize(cross_entropy)

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.2
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           10000, 0.96, staircase=True)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy,global_step=global_step)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

minibatch_size = 20 #N_train needs to be divisible by batch_size
N_epochs = 10
permut = np.arange(N_train)

ep_list    = []
ce_tr_list = []
ce_te_list = []
ac_list    = []

for ep in range(N_epochs):
  np.random.shuffle(permut)
  configs = x_train[permut,:]
  labels  = y_train[permut,:]
  ce_tr = sess.run(cross_entropy, feed_dict={x: configs, y_: labels})
  ce_te = sess.run(cross_entropy, feed_dict={x: x_test, y_: y_test})
  ac = sess.run(accuracy,      feed_dict={x: x_test, y_: y_test})
  
  ep_list.append(ep)
  ce_tr_list.append(ce_tr)
  ce_te_list.append(ce_te)
  ac_list.append(ac)
  
  if ep%50==0:
    print('\nepoch = %d' %(ep))
    print('  LR         = %f' %sess.run(learning_rate))
    print('  Cross ent. (train) = %f' %ce_tr)
    print('  Cross ent. (test)  = %f' %ce_te)
    print('  Accuracy           = %f' %ac)
    
    plt.figure(1)
    plt.clf()
    plt.plot(ep_list,ac_list)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.pause(0.01)

    plt.figure(2)
    plt.clf()
    plt.plot(ep_list,ce_tr_list, label='Train')
    plt.plot(ep_list,ce_te_list, label='Test')
    plt.xlabel("Epochs")
    plt.ylabel("Cross Entropy")
    plt.legend(loc='upper right')
    plt.pause(0.01)

  for k in range(0, N_train, minibatch_size):
    batch_xs = configs[k:k+minibatch_size,:]
    batch_ys = labels[k:k+minibatch_size,:]
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

print
print(sess.run(cross_entropy, feed_dict={x: x_test, y_: y_test}))
print(sess.run(accuracy,      feed_dict={x: x_test, y_: y_test}))

plt.figure(1)
plt.savefig('accVsEpochs_sigmoidCE_%depochs_%.4fexpLR_%dseed.pdf' %(N_epochs,starter_learning_rate,seed))

plt.figure(2)
plt.savefig('ceVsEpochs_sigmoidCE_%depochs_%.4fexpLR_%dseed.pdf' %(N_epochs,starter_learning_rate,seed))

plt.show()



