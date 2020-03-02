import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import sys
from sklearn import metrics

# Parameters
# ==================================================
fold_i=0
# Data Parameters
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/%d_rt-polarity_val.pos"%fold_i, "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/%d_rt-polarity_val.neg"%fold_i, "Data source for the negative data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/2019/checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()
FLAGS._parse_flags()
#FLAGS(sys.argv)
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here


if FLAGS.eval_train:
    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    y_test = np.argmax(y_test, axis=1)
else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate

        scores = graph.get_operation_by_name("output/scores").outputs[0]

        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []
        all_scores=[]
        for x_test_batch in batches:
            batch_predictions,batch_scores = sess.run([predictions,scores], {input_x: x_test_batch, dropout_keep_prob: 1.0})

            all_scores=np.concatenate([all_scores, batch_scores[:,1]])

            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    #print(all_scores)
    #exit()
    auc=metrics.roc_auc_score(y_test,all_scores)
    print(all_scores)
    performance=[]
    for threshold in np.arange(-10,10,0.1): 
        all_predictions=np.zeros(len(y_test))
        all_predictions[all_scores>threshold]=1
        all_predictions=[1 if result>0 else 0 for result in all_scores]

        acc=metrics.accuracy_score(y_test,all_predictions)
        rec=metrics.recall_score(y_test,all_predictions)
        prec=metrics.precision_score(y_test,all_predictions)
        f1=metrics.f1_score(y_test,all_predictions)
        #auc=metrics.roc_auc_score(y_test,all_predictions)
        performance.append([threshold,acc,prec,rec,f1,auc])
        print(acc,prec,rec,f1,auc)
        exit()
    performance=np.array(performance)

    performance_sort=performance[performance[:, -2].argsort()]

    print(performance_sort[-20:])
    with open("result.txt", "ab") as f:
        #f.write(b"\n")
        np.savetxt(f, performance_sort[-1].reshape((-1,6)),fmt='%.4f')

# Save the evaluation to a csv
#predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
predictions_human_readable = np.row_stack(all_predictions)
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
