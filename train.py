from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import glob
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from utils import *
from models import GAN
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from load_fake_data import *
from load_clean_data import *
import timeit




# Set random seed
seed=321
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'gan', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 5e-4, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 40, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1',16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 40, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
#Need to be changed by dataset
flags.DEFINE_string('datasetname', 'ml_error', 'Dataset to be used.')
flags.DEFINE_string('datapath',"dataset/ml/","Dataset path")

def calc_scores_error_detection(y_true, y_pred):
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    #print("y_true's shape is"+ str(y_true.shape))
    #print("y_pred's shape is"+ str(y_pred.shape))
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_true, y_pred, average=None)
    balanced_accuracy_score=metrics.accuracy_score(y_true, y_pred,normalize=True)
    return precision[-1], recall[-1], fscore[-1], support[-1],balanced_accuracy_score


#start marking down the time
start = timeit.default_timer()

learning_rate = FLAGS.learning_rate
#print("learning rate %14.6f"%learning_rate)


#load the preprocessed fake data and polluted node ids
feats_fake = load_fake_data(FLAGS.datapath,FLAGS.datasetname)
print(feats_fake.shape)


# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask= load_clean_data(FLAGS.datapath,FLAGS.datasetname)

# Some preprocessing
features = preprocess_features(features)
z_dim =features.shape[1]
if FLAGS.model =='gan':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GAN
    z_dim = z_dim
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))


# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.placeholder(tf.float32,shape=features.shape),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32) , # helper variable for sparse dropout
    'inputs_z': tf.placeholder(tf.float32, shape=(None, z_dim)) #placehoder for the inputing fake data into discriminator
}

# Create model

model = model_func(placeholders, real_input_dim=features.shape[1], z_input_dim=z_dim, logging=True)


saver = tf.train.Saver(max_to_keep=FLAGS.epochs)

checkpoint_path = FLAGS.datasetname+"training_1/"
checkpoint_dir = os.path.dirname(checkpoint_path)

sample_z = feats_fake
train_d_loss, val_d_loss =  [], []
train_g_loss =[]


############################################################Real implementation 
# Initialize session
sess = tf.Session()


# Define model evaluation function


def evaluate(features, support, labels, mask, sample_z, placeholders):
    t_test = time.time()
    num_examples = mask.sum()
    num_correct = 0
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    feed_dict_val.update({placeholders['inputs_z']: sample_z})
    val_correct,masked_v_pre, masked_v_label, v_discriminator_loss= sess.run([model.masked_correct,model.masked_pred_class, model.masked_labels,model.d_loss], feed_dict=feed_dict_val)
    num_correct += val_correct
    val_accuracy = num_correct / float(num_examples)
    y_true = np.argmax(masked_v_label, axis=1)
    y_pred = masked_v_pre
    v_precision, v_recall, v_f1_score, v_support = metrics.precision_recall_fscore_support(y_true, y_pred, average=None)

    return val_accuracy, v_precision[-1], v_recall[-1], v_f1_score[-1], v_discriminator_loss,(time.time() - t_test)



# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []
f1_score_val=[]

best = 0.0




best_epoch = 0
bad_counter = 0

test_f1_max =0.0
t_total = time.time()
# Train model
for epoch in range(FLAGS.epochs):

    print("Epoch",epoch)
    t1e = time.time()
    num_examples = train_mask.sum()
    num_correct = 0
    # Construct feed dictionary for the training part
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    feed_dict.update({placeholders['inputs_z']: sample_z})

    _, _, correct, masked_t_pre, masked_t_label,discriminator_loss, generator_loss =sess.run([model.d_opt, model.g_opt, model.masked_correct,
                                                                  model.masked_pred_class, model.masked_labels,model.d_loss, model.g_loss] , feed_dict=feed_dict)
    num_correct += correct
    ##add the training discriminator loss
    train_d_loss.append(discriminator_loss)
    train_g_loss.append(generator_loss)
    sess.run([model.shrink_lr])
    train_accuracy = num_correct / float(num_examples)
    y_true = np.argmax(masked_t_label, axis=1)
    y_pred = masked_t_pre
    # print("y_true's shape is"+ str(y_true.shape))
    # print("y_pred's shape is"+ str(y_pred.shape))
    t_precision, t_recall, t_f1_score, t_support = metrics.precision_recall_fscore_support(y_true, y_pred, average=None)
    print("\t\tClassifier train accuracy: ", train_accuracy)
    t_accu = metrics.accuracy_score(y_true, y_pred, normalize=True)
    print("\t\tClassifier train accuracy validation:", t_accu)
    print("\t\tClassifier train precision:", t_precision[-1])
    print("\t\tClassifier train recall:", t_recall[-1])
    print("\t\tClassifier train f1_score:", t_f1_score[-1])
    
    # Training step  is finished



    # Validation
    v_acc, v_pre, v_recall, v_f1, v_discriminator_loss, duration = evaluate(features, support, y_val, val_mask, sample_z, placeholders)
    print("\t\tClassifier validation accuracy: ", v_acc)
    print("\t\tClassifier validation precision:", v_pre)
    print("\t\tClassifier validation recall:", v_recall)
    print("\t\tClassifier validation f1_score:", v_f1)

    ##add the training discriminator loss
    val_d_loss.append(v_discriminator_loss)

    f1_score_val.append(v_f1)


    saver.save(sess, './' + checkpoint_path + 'my_test_model', global_step=epoch)
    if f1_score_val[-1] >= best:
        best = f1_score_val[-1]
        best_epoch = epoch
        bad_counter = 0
        print("the best epoch is %d" % best_epoch)
    else:
        bad_counter += 1

    if bad_counter == FLAGS.early_stopping:
        break

    files = glob.glob('./' + checkpoint_path + '*.index')
    for file in files:
        epoch_nb = file.split('.index')[0]
        epoch_nb = int(epoch_nb.split('-')[1])
        if epoch_nb < best_epoch:
            os.remove(file)

    files = glob.glob('./' + checkpoint_path + '*.meta')
    for file in files:
        epoch_nb = file.split('.meta')[0]
        epoch_nb = int(epoch_nb.split('-')[1])
        if epoch_nb < best_epoch:
            os.remove(file)

    files = glob.glob('./' + checkpoint_path + '*.data*')
    for file in files:
        epoch_nb = file.split('.data')[0]
        epoch_nb = int(epoch_nb.split('-')[1])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('./' + checkpoint_path + '*.index')
for file in files:
    epoch_nb = file.split('.index')[0]
    epoch_nb = int(epoch_nb.split('-')[1])
    if epoch_nb > best_epoch:
        os.remove(file)

files = glob.glob('./' + checkpoint_path + '*.meta')
for file in files:
    epoch_nb = file.split('.meta')[0]
    epoch_nb = int(epoch_nb.split('-')[1])
    if epoch_nb > best_epoch:
        os.remove(file)

files = glob.glob('./' + checkpoint_path + '*.data*')
for file in files:
    epoch_nb = file.split('.data')[0]
    epoch_nb = int(epoch_nb.split('-')[1])
    if epoch_nb > best_epoch:
        os.remove(file)


print(""
      ""
      ""
      "Optimization Finished!")


with open(FLAGS.datasetname+'test.txt', 'w') as f:
    train_time=format(time.time() - t_total)
    f.write("The training time is: \n")
    f.write("%s" % str(train_time)+ 's'+'\n')
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


# Testing
sess = tf.Session()
#enable to directly load pre-trained model
#best_epoch = 39
new_saver = tf.train.import_meta_graph('./'+ checkpoint_path+'my_test_model-'+str(best_epoch)+'.meta')
new_saver.restore(sess, './' + checkpoint_path + 'my_test_model-'+str(best_epoch))

# Now, let's access and create placeholders variables
graph = tf.get_default_graph()

print("The best epoch is %d" %best_epoch)

acc_test, pre_test, recall_test, f1_test, discriminator_loss_test, duration_test = evaluate(features, support, y_test, test_mask, sample_z,placeholders)
print("\t\tClassifier testing accuracy: ", acc_test)
print("\t\tClassifier testing precision:", pre_test)
print("\t\tClassifier testing recall:", recall_test)
print("\t\tClassifier testing f1_score:", f1_test)


###continue writing the test performance into a txt file
with open(FLAGS.datasetname+'test.txt', 'a') as f:
    f.write("The testing accuracy is: \n")
    f.write("%s" % str(acc_test)+ '\n')
    f.write("The testing precision is: \n")
    f.write("%s" % str(pre_test)+ '\n')
    f.write("The testing recall is: \n")
    f.write("%s" % str(recall_test)+ '\n')
    f.write("The testing f1_score is: \n" )
    f.write("%s" % str(f1_test) + '\n')




stop = timeit.default_timer()

print('The total time:', stop-start)


