import sys
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
from scipy import interp
from itertools import cycle
import tensorflow as tf
import tensorflow_hub as hub

def parse_function(filename, label):
    image_string = tf.read_file(filename)
    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)
    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    #resized_image = tf.image.resize_images(image, [364,448])
    resized_image = tf.image.resize_images(image, [448,448])

    return resized_image, label

def train_preprocess(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.random_crop(image, [224,224,3], seed=777)
    image = tf.image.per_image_standardization(image)
    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label

def test_preprocess(image, label):
    #crop middle of image
    #image = tf.image.crop_and_resize(image, boxes=[[0.25, 0.25, 0.75, 0.75]],box_ind = [9], crop_size=[224,224])
    #image = tf.image.crop_to_bounding_box(image,int(364*0.25), int(448*0.25), int(364*0.75), int(448*0.75)) 
    image = tf.image.crop_to_bounding_box(image,int(448*0.25), int(448*0.25), int(448*0.75), int(448*0.75)) 
    image = tf.image.resize_images(image, [224,224])

    image = tf.image.per_image_standardization(image)
    #image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


#hyper parameter
batch_size = 9
num_batches = 4000
max_epoch = 60

#step_size = 3 
step_size = 5 
#gamma = 0.5
gamma = 1
#base_lr = 0.0001
base_lr = 0.00001
iter_size = 200
save_iter = 2000

prev_epoch = 0
prev_iter = 357999

#check_point_file = ""
#Train = True

check_point_file = "PAN4-597998"
Train = False

root_folder = '/media/hci-jw/Plextor1tb/workspace/data/'
save_folder = '/media/hci-jw/3TB/PAN_checkpoint/'
test_result_save_folder = '/media/hci-jw/3TB/PAN_test_result/'

#load Train dataset
df = open(root_folder+"shuffled_dataset.list",'r')
dftempfile = df.read()
dftemplist = dftempfile.split('\n')
dftemplist.pop()

filename_list = []
label_list = []

for dftemp in dftemplist:
    name = dftemp.split(' ')[0]
    filename_list.append(root_folder+name)
    
    label = dftemp.split(' ')[1]
    label_list.append(int(label))

## Load validation dataset
#validf = open(root_folder+"sh_valid.list",'r')
#valid_tempfile = validf.read()
#valid_templist = valid_tempfile.split('\n')
#valid_templist.pop()
#
#valid_filename_list = []
#valid_label_list = []
#
#for vtemp in valid_templist:
#    vname = vtemp.split(' ')[0]
#    valid_filename_list.append(root_folder+vname)
#    
#    vlabel = vtemp.split(' ')[1]
#    valid_label_list.append(int(vlabel))

# Load Test dataset
testf = open(root_folder+"test_dataset.list",'r')
test_tempfile = testf.read()
test_templist = test_tempfile.split('\n')
test_templist.pop()

test_filename_list = []
test_label_list = []

for test_temp in test_templist:
    test_filename_list.append(root_folder+test_temp)
    test_label_list.append(int(0))


learning_rate = tf.placeholder(tf.float32, shape =[])

trainset = tf.data.Dataset.from_tensor_slices((filename_list, label_list))
trainset = trainset.map(parse_function, num_parallel_calls=4)
trainset = trainset.map(train_preprocess, num_parallel_calls=4)
trainset = trainset.batch(batch_size)
trainset = trainset.prefetch(1)

##valid set initialize
#validset = tf.data.Dataset.from_tensor_slices((valid_filename_list, valid_label_list))
#validset = validset.map(parse_function, num_parallel_calls=4)
#validset = validset.map(train_preprocess, num_parallel_calls=4)
#validset = validset.batch(batch_size)
#validset = validset.prefetch(1)

#test set initialize
testset = tf.data.Dataset.from_tensor_slices((test_filename_list, test_label_list))
testset = testset.map(parse_function, num_parallel_calls=4)
testset = testset.map(test_preprocess, num_parallel_calls=4)
testset = testset.batch(batch_size)
testset = testset.prefetch(1)

iterator = tf.data.Iterator.from_structure(trainset.output_types,trainset.output_shapes)

init_op = iterator.make_initializer(trainset)
#valid_init_op = iterator.make_initializer(validset)
test_init_op = iterator.make_initializer(testset)

images, labels = iterator.get_next()

#valid_iterator = validset.make_one_shot_iterator()
#valid_images, valid_labels = valid_iterator.get_next()
#valid_init_op = valid_iterator.initializer


# Get the labels from the input data pipeline
labels = tf.cast(labels, tf.int64)

# Training Graph
# Load Resnet V1-50
res_app = hub.Module("https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/1", trainable=True)
res_act = hub.Module("https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/1", trainable=True)

features_app = res_app(images)  # Features with shape [batch_size, num_features].
features_act = res_act(images)  # Features with shape [batch_size, num_features].

#slice feature app
app_sperate = tf.reshape(features_app, [-1, 9, 1, 2048])
seperated_bbox, seperated_ctx, seperated_part_app = tf.split(app_sperate, [1, 1, 7], 1)
reshaped_part_app = tf.reshape(seperated_part_app, [-1, 1, 7, 2048])
pool_part_app = tf.nn.max_pool(reshaped_part_app, ksize=[ 1, 1, 7, 1], strides=[1,1,7,1], padding='SAME')

#slice feature act
act_sperate = tf.reshape(features_act, [-1, 9, 1, 2048])
_, _, seperated_part_act = tf.split(act_sperate, [1, 1, 7], 1)
reshaped_part_act_pre = tf.reshape(seperated_part_act, [-1, 1, 1, 2048])
reshaped_part_act = tf.reshape(reshaped_part_act_pre, [-1, 1, 7, 2048])
pool_part_act = tf.nn.max_pool(reshaped_part_act, ksize=[ 1, 1, 7, 1], strides=[1,1,7,1], padding='SAME')

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
pred_bbox = tf.equal(tf.argmax(fc_bbox,axis=3), label_bbox)
pred_ctx = tf.equal(tf.argmax(fc_ctx,axis=3), label_ctx)
pred_part_app = tf.equal(tf.argmax(fc_part_app,axis=3), label_bbox)
pred_act = tf.equal(tf.argmax(fc_act,axis=3), reshaped_part_label)
pred_part_act = tf.equal(tf.argmax(fc_part_act,axis=3), label_bbox)
pred_stanford2 = tf.equal(tf.argmax(fc_stanford2,axis=3), label_bbox)

score = tf.reduce_max(tf.cast(pred_stanford2, tf.float32)) + tf.reduce_max(tf.cast(pred_part_act, tf.float32)) + tf.reduce_max(tf.cast(pred_act, tf.float32))+ tf.reduce_max(tf.cast(pred_part_app, tf.float32))+ tf.reduce_max(tf.cast(pred_ctx, tf.float32))+ tf.reduce_max(tf.cast(pred_bbox, tf.float32))

# Define the prediction as the argmax of the scores
#valid_pred_part_act = tf.equal(tf.argmax(fc_part_act,axis=3), label_bbox)
#valid_pred_stanford2 = tf.equal(tf.argmax(fc_stanford2,axis=3), label_bbox)
#tscore = tf.reduce_mean(tf.cast(valid_pred_stanford2, tf.float32)) + tf.reduce_mean(tf.cast(valid_pred_part_act, tf.float32))

#test_pred_part_act = tf.nn.top_k(fc_part_act, k=7)
test_pred_part_act = tf.argmax(fc_act, axis=3)

#original ratio raw result
#test_pred_score = (1*fc_stanford2 + 0.2*fc_part_act)/1.2

test_pred_score = fc_stanford2
test_pred_stanford2 = tf.argmax(test_pred_score,axis=3)

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


saver = tf.train.Saver(max_to_keep = 200, keep_checkpoint_every_n_hours=1)
with tf.Session() as sess:
    if Train: 
        if check_point_file != "":
            saver = tf.train.import_meta_graph(save_folder+check_point_file+".meta")
            saver.restore(sess, save_folder+check_point_file)
        else:
            sess.run(tf.global_variables_initializer())
        counter = 0
        avg_loss = 0
        acc = 0
        for i in range(max_epoch):

#            sess.run(valid_init_op)
#            valid_avg_loss = 0
#            valid_acc = 0
#            num_valid_batch = len(valid_filename_list)/9
#            for k in range(num_valid_batch):
#                valid_loss, valid_score = sess.run([loss, tscore])    
#                valid_avg_loss += valid_loss
#                valid_acc += valid_score
#            valid_avg_loss = valid_avg_loss/num_valid_batch 
#            valid_acc = valid_acc / (num_valid_batch*2) * 100
#            print "valid_loss {:.2f} valid_accuracy {:6.2f}%".format(valid_avg_loss, valid_acc)
#            sys.stdout.flush()

            sess.run(init_op)
            acc = 0
            avg_loss = 0
            for j in tqdm(range(num_batches)):
                _, loss_val, score_val = sess.run([train_op, loss, score], feed_dict={learning_rate: base_lr*(gamma**counter)})
                avg_loss += loss_val
                acc += score_val

                if j%iter_size == 0 and j > 0:
                    acc = acc / (iter_size*6) * 100
                    print "epoch", i, "iteration", i*num_batches+j+prev_iter , "loss {:6.2f}".format(avg_loss/iter_size), "accuracy {:6.2f}".format(acc)
                    sys.stdout.flush()
                    avg_loss = 0
                    acc = 0

                if j%save_iter == 0 :
                    print "%d check point saved!"%(i*num_batches+j+prev_iter)
                    sys.stdout.flush()
                    saver.save(sess, save_folder+'PAN4', global_step= i*num_batches+j+prev_iter) 

            if i>0 and i%step_size ==0:
                counter += 1 
        print "epoch", i, "iteration", i*num_batches+j+prev_iter , "loss {:6.2f}".format(avg_loss/iter_size), "accuracy {:6.2f}".format(acc/(iter_size*6)*100)
        print "%d check point saved!"%(i*num_batches+j+prev_iter)
        sys.stdout.flush()
        saver.save(sess, save_folder+'PAN4', global_step= i*num_batches+j+prev_iter) 

    else:
        if check_point_file != "":
            part_action_list = ["breathing", "drinking", "laughing", "looking down", "looking through", "looking up", "normal", "speaking", "brushing teeth", "bending", "fading away", "normal", "lying", "crouching", "forking", "running", "sitting", "action", "standing", "walking", "curving down", "curving up", "straight down", "straight up", "cutting", "half holding", "fully holding", "merging", "slack", "printing", "proping", "supporting", "washing", "waving", "writing", "not found"]
            
            action_list = ["applauding","blowing_bubbles","brushing_teeth","cleaning_the_floor","climbing","cooking","cutting_trees","cutting_vegetables","drinking","feeding_a_horse","fishing","fixing_a_bike","fixing_a_car","gardening","holding_an_umbrella","jumping","looking_through_a_microscope","looking_through_a_telescope","phoning","playing_guitar","playing_violin","pouring_liquid","pushing_a_cart","reading","riding_a_bike","riding_a_horse","rowing_a_boat","running","shooting_an_arrow","smoking","taking_photos","texting_message","throwing_frisby","using_a_computer","walking_the_dog","washing_dishes","watching_TV","waving_hands","writing_on_a_board","writing_on_a_book"]

            correct_list = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
            true_num = [ 184, 159, 100, 112, 195, 188, 103, 89, 156, 187, 173, 128, 151, 99, 192, 195, 91, 103, 159, 189, 160, 100, 135, 145, 193, 196, 85, 151, 114, 141, 97, 93, 102, 130, 193, 82, 123, 110, 83, 146]
            print "Test begin"
            saver = tf.train.import_meta_graph(save_folder+check_point_file+".meta")
            saver.restore(sess, save_folder+check_point_file)
            #saver.restore(sess, tf.train.latest_checkpoint(save_folder))
            print "Model restored."

            #Run test 
            sess.run(test_init_op)

            #set null list 
            y_score = np.ones([len(test_filename_list)/9,40])
            y_test = np.zeros([len(test_filename_list)/9,40])
            for i in tqdm(range(len(test_filename_list)/9)):
                for idx, name in enumerate(test_filename_list[i*9:(i+1)*9]):
                    if idx == 0:
                        img_total = cv2.imread(name)
                        img_total = cv2.cvtColor(img_total, cv2.COLOR_BGR2RGB)
                    elif idx == 1:
                        img_bbox = cv2.imread(name)
                        img_bbox = cv2.cvtColor(img_bbox, cv2.COLOR_BGR2RGB)
                    elif idx == 2:
                        img_head =cv2.imread(name)
                        img_head = cv2.cvtColor(img_head, cv2.COLOR_BGR2RGB)
                    elif idx == 3:
                        img_torso =cv2.imread(name)
                        img_torso = cv2.cvtColor(img_torso, cv2.COLOR_BGR2RGB)
                    elif idx == 4:
                        img_legs = cv2.imread(name)
                        img_legs = cv2.cvtColor(img_legs, cv2.COLOR_BGR2RGB)
                    elif idx == 5:
                        img_larm =cv2.imread(name)
                        img_larm = cv2.cvtColor(img_larm, cv2.COLOR_BGR2RGB)
                    elif idx == 6:
                        img_rarm =cv2.imread(name)
                        img_rarm = cv2.cvtColor(img_rarm, cv2.COLOR_BGR2RGB)
                    elif idx == 7:
                        img_lhand =cv2.imread(name)
                        img_lhand = cv2.cvtColor(img_lhand, cv2.COLOR_BGR2RGB)
                    elif idx == 8:
                        img_rhand = cv2.imread(name)
                        img_rhand = cv2.cvtColor(img_rhand, cv2.COLOR_BGR2RGB)

                class_folder = name.split('/')[7]
                save_file = name.split('/')[8][0:-11]+".jpg"
                

                pred_part_act, pred_stanford, pred_stanford_score = sess.run([test_pred_part_act, test_pred_stanford2,test_pred_score])

                pred_part_act = np.ravel(pred_part_act)
                pred_stanford = np.ravel(pred_stanford)
                pred_stanford_score = np.ravel(pred_stanford_score)

                #set scores
                y_score[i] *= pred_stanford_score

                true_idx = -1 
                for idx, act in enumerate(action_list):
                    if act in class_folder:
                        true_idx = idx
                        break

                y_test[i,true_idx] += 1
			   
		fig = plt.figure(figsize = (12,8))
		total = plt.subplot2grid((3, 6), (0, 0), colspan=3, rowspan=3)
		head = plt.subplot2grid((3, 6), (0, 4))
		torso = plt.subplot2grid((3, 6), (1, 4))
		legs = plt.subplot2grid((3, 6), (2, 4))
		larm = plt.subplot2grid((3, 6), (1, 3))
		rarm = plt.subplot2grid((3, 6), (1, 5))
		lhand = plt.subplot2grid((3, 6), (0, 3))
		rhand = plt.subplot2grid((3, 6), (0, 5))

		total.axis("off")
		head.axis("off")  
		torso.axis("off") 
		legs.axis("off") 
		larm.axis("off") 
		rarm.axis("off") 
		lhand.axis("off") 
		rhand.axis("off")

                total.imshow(img_total)
                head.imshow(img_head)
                torso.imshow(img_torso)
                legs.imshow(img_legs)
                larm.imshow(img_larm)
                rarm.imshow(img_rarm)
                lhand.imshow(img_lhand)
                rhand.imshow(img_rhand)
 
                total.set_title("Total Image")
                head.set_title("Head")
                torso.set_title("Torso")
                legs.set_title("Legs")
                larm.set_title("Left Arm")
                rarm.set_title("Right Arm")
                lhand.set_title("Left Hand")
                rhand.set_title("Right Hand")
	
		if pred_stanford[0] <0 and pred_stanford[0] >40:
                    total_result = "not found"
                    print "not found"
                    sys.stdout.flush()

	        else:
                    total_result = action_list[pred_stanford[0]] 
                     
                    if total_result in save_file:
                        #print "total_result:", total_result
                        #print "save_file:", save_file
                        correct_list[pred_stanford[0]] += 1
                        #print "pred : ", pred_stanford[0]
                        #print "corr : ", correct_list[pred_stanford[0]]
                        #sys.stdout.flush()

                
                pred_part_act = list(map(lambda x: x-40, pred_part_act))

                #check part
                if pred_part_act[0] <0 or pred_part_act[0] >34:
                    head_result = "not found"
                else:
                    head_result = part_action_list[pred_part_act[0]]

                if pred_part_act[1] <0 or pred_part_act[1] >34:
                    torso_result = "not found"
                else:
                    torso_result = part_action_list[pred_part_act[1]]

                if pred_part_act[2] <0 or pred_part_act[2] >34:
                    legs_result = "not found"
                else:
                    legs_result = part_action_list[pred_part_act[2]]

                if pred_part_act[3] <0 or pred_part_act[3] >34:
                    larm_result = "not found"
                else:
                    larm_result = part_action_list[pred_part_act[3]]

                if pred_part_act[4] <0 or pred_part_act[4] >34:
                    rarm_result = "not found"
                else:
                    rarm_result = part_action_list[pred_part_act[4]]

                if pred_part_act[5] <0 or pred_part_act[5] >34:
                    lhand_result = "not found"
                else:
                    lhand_result = part_action_list[pred_part_act[5]]

                if pred_part_act[6] <0 or pred_part_act[6] >34:
                    rhand_result = "not found"
                else:
                    rhand_result = part_action_list[pred_part_act[6]]

		total.text(.5,-0.1, total_result ,size=12, ha='center', transform=total.transAxes) 
		head.text(.5,-0.3, head_result,size=11, ha='center', transform=head.transAxes) 
		torso.text(.5,-0.3, torso_result,size=11, ha='center', transform=torso.transAxes) 
		legs.text(.5,-0.3, legs_result,size=11, ha='center', transform=legs.transAxes) 
		larm.text(.5,-0.3, larm_result,size=11, ha='center', transform=larm.transAxes) 
		rarm.text(.5,-0.3, rarm_result,size=11, ha='center', transform=rarm.transAxes) 
		lhand.text(.5,-0.3, lhand_result,size=11, ha='center', transform=lhand.transAxes) 
		rhand.text(.5,-0.3, rhand_result,size=11, ha='center', transform=rhand.transAxes) 

                #plt.show()

                result_path = test_result_save_folder + "/" + class_folder 
                if not os.path.exists(result_path):
                    os.makedirs(result_path)

                #save image with caption
                #plt.savefig(result_path +"/" + save_file)
                plt.close(fig) 

            class_acc = []
            for corr, tru in zip(correct_list, true_num):
                class_acc.append(float(float(corr)/float(tru)*100))

            for idx , k in enumerate(action_list):
                print k + ": {:6.2f}%".format(class_acc[idx])

	    #PR curve
            n_classes = 40

	    # For each class
	    precision = dict()
	    recall = dict()
	    average_precision = dict()
	    for i in range(n_classes):
		precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
		average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

	    # A "micro-average": quantifying score on all classes jointly
	    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
		y_score.ravel())
	    average_precision["micro"] = average_precision_score(y_test, y_score,
								 average="micro")
	    average_precision["macro"] = average_precision_score(y_test, y_score,
								 average="macro")
	    #print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))

	    plt.figure()
	    plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
		     where='post')
	    plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2,
			     color='b')

	    plt.xlabel('Recall')
	    plt.ylabel('Precision')
	    plt.ylim([0.0, 1.05])
	    plt.xlim([0.0, 1.0])
	    plt.title(
		'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
		.format(average_precision["micro"]))
	    #plt.show()
	    plt.savefig(check_point_file+"pr_curve.jpg")
            print average_precision

        else:
            print "check point is not exist!"

