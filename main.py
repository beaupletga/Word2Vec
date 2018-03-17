import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

WINDOWS_SIZE=2

dataset="He is the king . The king is royal . She is the royal queen"
ori_dataset=dataset
for i in ['.',';',',','?','!',':','(',')']:
    dataset = dataset.replace(i, "")
dataset_split=str.split(dataset)
dataset_split_unique = list(set(dataset_split))

DATASET_SIZE=len(dataset_split_unique)
HIDDEN_DIM=2
LR=1e-1
EPOCHS=1000
BATCH_SIZE=2**5

def create_dic():
    text_to_int={}
    int_to_text={}
    for index,i in enumerate(dataset_split_unique):
        text_to_int[i]=index
        int_to_text[index]=i
    return text_to_int,int_to_text


def create_hot_encoding(dataset,text_to_int,int_to_text):
    x=[]
    y=[]
    for i in range(len(dataset)):
        tmp1=np.zeros((len(text_to_int)))
        tmp2=np.zeros((len(text_to_int)))
        tmp1[text_to_int[dataset[i][0]]]=1
        tmp2[text_to_int[dataset[i][1]]]=1
        x.append(tmp1)
        y.append(tmp2)

    x=np.array(x)
    y=np.array(y)
    return x,y

def create_dataset():
    x=[]
    for i in range(0,len(dataset_split)):
        for j in range(i+1,i+WINDOWS_SIZE+1):
            if j<len(dataset_split):
                x.append([dataset_split[i],dataset_split[j]])
    dataset_split.reverse()
    for i in range(0,len(dataset_split)):
        for j in range(i+1,i+WINDOWS_SIZE+1):
            if j<len(dataset_split):
                x.append([dataset_split[i],dataset_split[j]])

    dataset_split.reverse()
    return x

def getKey(item):
    return item[1]

def display_neighbours(weights,int_to_text):
    distances=[]
    for i in range(weights.shape[0]):
        tmp=[[i,0]]
        for j in range(weights.shape[0]):
            if i!=j:
                distance=np.linalg.norm(weights[i]-weights[j])
                tmp.append([j,distance])
        distances.append(tmp)

    for i in range(len(distances)):
        distances[i]=sorted(distances[i], key=getKey)

    for i in range(len(distances)):
        for j in range(len(distances[i])):
            distances[i][j][0]=int_to_text[distances[i][j][0]]
        print distances[i]

dataset=create_dataset()
text_to_int,int_to_text=create_dic()
x_train,y_train=create_hot_encoding(dataset,text_to_int,int_to_text)

initializer=tf.contrib.layers.xavier_initializer()
X = tf.placeholder("float", [None, DATASET_SIZE])
Y = tf.placeholder("float", [None, DATASET_SIZE])

weights = {
    'w1': tf.Variable(initializer([DATASET_SIZE,HIDDEN_DIM])),
    'w2': tf.Variable(initializer([HIDDEN_DIM, DATASET_SIZE])),
}
biases = {
    'b1': tf.Variable(tf.zeros([HIDDEN_DIM])),
    'b2': tf.Variable(tf.zeros([DATASET_SIZE])),
}

def neural_net(X):
    h1=tf.matmul(X, weights['w1']) + biases['b1']
    output=tf.nn.softmax(tf.matmul(h1, weights['w2']) + biases['b2'])
    return output

output=neural_net(X)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=Y))

optimizer = tf.train.AdamOptimizer(LR).minimize(cost)
sess= tf.Session()
sess.run(tf.global_variables_initializer())
val_loss_list = []
for epoch in range(EPOCHS):
    for batch_i in range(0, x_train.shape[0], BATCH_SIZE):
        _,loss = sess.run([optimizer,cost], feed_dict={
            X: x_train[batch_i:batch_i + BATCH_SIZE], Y: y_train[batch_i:batch_i + BATCH_SIZE]})
print 'Original text : ',ori_dataset

print text_to_int

print 'Weights :'
print weights['w1'].eval(sess)
print''
print 'Words Distances'
display_neighbours(weights['w1'].eval(sess),int_to_text)





