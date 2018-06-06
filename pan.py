import tensorflow as tf
import tensorflow_hub as hub

def parse_function(filename, label):
    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)
    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    resized_image = tf.image.resize_images(image, [224,224])
    #resized_image = tf.image.resize_images(image, [64,64])

    return resized_image, label

def train_preprocess(image, label):
    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label

root_folder = '/media/hci-jw/Plextor1tb/workspace/data/'
save_folder = '/media/hci-jw/3TB/PAN_checkpoint/'
df = open(root_folder+"dataset.list",'r')
tempfile = df.read()
templist = tempfile.split('\n')
templist.pop()

filename_list = []
label_list = []

for temp in templist:
    name = temp.split(' ')[0]
    filename_list.append(root_folder+name)
    
    label = temp.split(' ')[1]
    label_list.append(int(label))

batch_size = 9
num_batches = 4000
max_epoch = 9 

step_size = 3
gamma = 0.1
base_lr = 0.00001
iter_size = 20
save_iter = 200

learning_rate = tf.placeholder(tf.float32, shape =[])

dataset = tf.data.Dataset.from_tensor_slices((filename_list, label_list))
#dataset = dataset.shuffle(len(filename_list))
dataset = dataset.map(parse_function, num_parallel_calls=4)
dataset = dataset.map(train_preprocess, num_parallel_calls=4)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(1)

# step 4: create iterator and final input tensor
iterator = dataset.make_initializable_iterator()
images, labels = iterator.get_next()
init_op = iterator.initializer

# Get the labels from the input data pipeline
labels = tf.cast(labels, tf.int64)

res_app = hub.Module("https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/1", trainable=True)
res_act = hub.Module("https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/1", trainable=True)
#height, width = hub.get_expected_image_size(module)

features_app = res_app(images)  # Features with shape [batch_size, num_features].
features_act = res_act(images)  # Features with shape [batch_size, num_features].

#slice feature app
app_sperate = tf.reshape(features_app, [-1, 9, 1, 2048])
seperated_bbox, seperated_ctx, seperated_part_app = tf.split(app_sperate, [1, 1, 7], 1)
reshaped_part_app = tf.reshape(seperated_part_app, [-1, 1, 7, 2048])
pool_part_app = tf.nn.max_pool(reshaped_part_app, ksize=[ 1, 1, 7, 1], strides=[1,1,7,1], padding='SAME')
#pool_part_app = tf.layers.max_pooling2d(reshaped_part_app, pool_size=[1, 7], strides=1, padding='SAME')

#slice feature act
act_sperate = tf.reshape(features_act, [-1, 9, 1, 2048])
_, _, seperated_part_act = tf.split(act_sperate, [1, 1, 7], 1)
reshaped_part_act_pre = tf.reshape(seperated_part_act, [-1, 1, 1, 2048])
reshaped_part_act = tf.reshape(reshaped_part_act_pre, [-1, 1, 7, 2048])
pool_part_act = tf.nn.max_pool(reshaped_part_act, ksize=[ 1, 1, 7, 1], strides=[1,1,7,1], padding='SAME')
#pool_part_act = tf.layers.max_pooling2d(reshaped_part_act, pool_size=[1, 7], strides=1, padding='SAME')

#slice label
reshaped_label = tf.reshape(labels, [-1, 1, 1, 9])
label_bbox, label_ctx, label_part = tf.split(reshaped_label, [1, 1, 7], 3)
reshaped_part_label = tf.reshape(label_part, [-1,1,1,1])

#fusion
fusion = tf.concat( [seperated_bbox, seperated_ctx, pool_part_app, pool_part_act], axis=3 )

#dense layers
fc_bbox = tf.layers.dense(seperated_bbox, 40)
fc_ctx = tf.layers.dense(seperated_ctx, 40)
fc_part_app = tf.layers.dense(pool_part_app, 40)
fc_act = tf.layers.dense(reshaped_part_act_pre, 75)
fc_part_act = tf.layers.dense(pool_part_act, 40)
fc_stanford2 = tf.layers.dense(fusion, 40, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.5))

# Define the prediction as the argmax of the scores
pred_bbox = tf.argmax(fc_bbox,1)
pred_ctx = tf.argmax(fc_ctx,1)
pred_part_app = tf.argmax(fc_part_app,1)
pred_act = tf.argmax(fc_act,1)
pred_part_act = tf.argmax(fc_part_act,1)
pred_stanford2 = tf.argmax(fc_stanford2,1)

# Define the loss
#loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
loss_bbox = tf.losses.sparse_softmax_cross_entropy(labels=label_bbox, logits=fc_bbox)
loss_ctx = tf.losses.sparse_softmax_cross_entropy(labels=label_ctx, logits=fc_ctx)
loss_act = tf.losses.sparse_softmax_cross_entropy(labels=reshaped_part_label, logits=fc_act)
loss_part_act = tf.losses.sparse_softmax_cross_entropy(labels=label_bbox, logits=fc_part_act)
loss_part_app = tf.losses.sparse_softmax_cross_entropy(labels=label_bbox, logits=fc_part_app)
loss_fuse = tf.losses.sparse_softmax_cross_entropy(labels=label_bbox, logits=fc_stanford2)

loss = loss_bbox + loss_ctx + 2*loss_act + 2*loss_part_act + loss_part_app + 0.5*loss_fuse

# Create an optimizer that will take care of the Gradient Descent
# Create the training operation
train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.4
#session = tf.Session(config=config)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(init_op)
    saver = tf.train.Saver()
    counter = 0
    for i in range(max_epoch):
        for j in range(num_batches):
            #sess.run([images, labels])
            _, loss_val = sess.run([train_op, loss], feed_dict={learning_rate: base_lr*(gamma**counter)})
            if j%iter_size == 0:
                print "epoch", i, "iteration", i*num_batches+j , "loss", loss_val
            if j%save_iter == 0 and j > 0:
                print "%d check point saved!"%(i*num_batches+j)
                saver.save(sess, save_folder+'PAN_%d.ckpt'%(i*num_batches+j)) 
        if i>0 and i%step_size ==0:
            counter += 1 
        sess.run(init_op)
    print "epoch", i, "iteration", i*num_batches+j , "loss", loss_val
    saver.save(sess, save_folder+'PAN_40000.ckpt') 

