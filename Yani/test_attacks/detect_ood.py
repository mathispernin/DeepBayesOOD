from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from six.moves import xrange
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


import os, sys, pickle, argparse
from scipy.special import logsumexp
# Ensure paths are correct for Colab execution
sys.path.append('../utils/')
sys.path.append('load/')

# Import necessary utilities (assuming these are in the correct paths)
from model_eval import model_eval
import keras.backend
from load_classifier import load_classifier
from import_data_cifar10 import load_data_cifar10 # Import the confirmed data loading script


def comp_logp(logit, y, text, comp_logit_dist=False):
    """Computes marginal and conditional log-probabilities."""
    logpx = logsumexp(logit, axis=1)
    logpx_mean = np.mean(logpx)
    logpx_std = np.sqrt(np.var(logpx))
    
    # NOTE: The model only outputs 2 logits (plane/frog), but y is likely 10-dim if OOD is passed.
    # The logpxy calculation relies on the shape of y matching the model output (logit).
    # Since we are focusing on OOD, we primarily rely on logpx.
    # We will proceed with logpxy calculation on ID and OOD samples,
    # trusting that comp_logp will be called with correctly sized y_clean (2-dim).
    
    # Calculate log p(x|y) statistics (Conditional Likelihood)
    logpxy = np.sum(y * logit, axis=1)
    logpxy_mean = []; logpxy_std = []
    
    # Find mean/std per class (only works if y is the 2-class one-hot encoding)
    for i in xrange(y.shape[1]):
        ind = np.where(y[:, i] == 1)[0]
        if len(ind) > 0:
            logpxy_mean.append(np.mean(logpxy[ind]))
            logpxy_std.append(np.sqrt(np.var(logpxy[ind])))
        
    print('%s: logp(x) = %.3f +- %.3f, logp(x|y) = %.3f +- %.3f' \
          % (text, logpx_mean, logpx_std, 
             np.mean(logpxy_mean) if logpxy_mean else 0, 
             np.mean(logpxy_std) if logpxy_std else 0))
    
    results = [logpx, logpx_mean, logpx_std, logpxy, logpxy_mean, logpxy_std]
    
    # Note: Omitted comp_logit_dist logic for brevity and OOD focus
    return results

def comp_detect(x, x_mean, x_std, alpha, plus):
    """Computes detection rate based on standard deviation threshold."""
    if plus:
        # Detection for high scores (e.g., KL divergence)
        detect_rate = np.mean(x > x_mean + alpha * x_std)
    else:
        # Detection for low scores (e.g., Log-Likelihood)
        detect_rate = np.mean(x < x_mean - alpha * x_std)
    return detect_rate * 100
    
def search_alpha(x, x_mean, x_std, target_rate=5.0, plus=False):
    """Searches for alpha that gives a target false alarm rate."""
    alpha_min = 0.0
    alpha_max = 3.0
    alpha_now = 1.5
    detect_rate = comp_detect(x, x_mean, x_std, alpha_now, plus)
    T = 0
    while np.abs(detect_rate - target_rate) > 0.01 and T < 20:
        if detect_rate > target_rate:
            alpha_min = alpha_now
        else:
            alpha_max = alpha_now
        alpha_now = 0.5 * (alpha_min + alpha_max)
        detect_rate = comp_detect(x, x_mean, x_std, alpha_now, plus)
        T += 1
    return alpha_now, detect_rate


def test_ood(batch_size, guard_name, data_name, save):
    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)
    # Create TF session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    print("Created TensorFlow session.")

    # --- 1. Load ALL Data and Split into ID / OOD ---
    img_rows, img_cols, channels = 32, 32, 3
    
    # Load ALL 10 CIFAR-10 classes (labels=None)
    datapath = '../cifar_data/' 
    # X_all_test is actually the test set; X_train is the training set
    x_train_full, X_all_test, y_train_full, Y_all_test = load_data_cifar10(datapath, labels=None)

    # Filter the data into ID (Plane/Frog) and OOD (Other) sets
    id_labels = [0, 6]  # Plane and Frog
    all_labels = np.argmax(Y_all_test, axis=1) # Convert 10-dim one-hot to class index

    # Get ID indices (EITHER 0 OR 6)
    id_indices = np.where((all_labels == id_labels[0]) | (all_labels == id_labels[1]))[0]

    # Get OOD indices (NOT 0 AND NOT 6)
    ood_indices = np.where((all_labels != id_labels[0]) & (all_labels != id_labels[1]))[0]
    
    # Filter ID training set to get the actual training data (2 classes, 2-dim one-hot)
    # This must match what the trained model expects.
    x_train, _, y_train, _ = load_data_cifar10(datapath, labels=id_labels)

    # Final Test Sets
    x_clean = X_all_test[id_indices] # In-distribution (ID) test set
    y_clean = Y_all_test[id_indices]
    x_ood = X_all_test[ood_indices] # Out-of-Distribution (OOD) test set
    y_ood = Y_all_test[ood_indices]
    
    # Ensure y_clean is 2-dim one-hot for model evaluation later
    id_label_map = {0: 0, 6: 1} # Map 10-class indices to 2-class indices
    y_clean_labels = all_labels[id_indices]
    y_clean_2dim = np.zeros((len(y_clean_labels), 2))
    for i, label in enumerate(y_clean_labels):
        y_clean_2dim[i, id_label_map[label]] = 1.0

    nb_classes = y_train.shape[1] # Should be 2
    
    print('Loaded ID (Train/Test): %d/%d, OOD Samples: %d' % 
          (x_train.shape[0], x_clean.shape[0], x_ood.shape[0]))

    # --- 2. Define Placeholders and Load Model ---
    x = tf.placeholder(tf.float32, shape=(batch_size, img_rows, img_cols, channels))
    y = tf.placeholder(tf.float32, shape=(batch_size, nb_classes)) # 2 classes

    gen = load_classifier(sess, guard_name, data_name)
    
    if 'bayes' in guard_name and 'distill' not in guard_name and 'query' not in guard_name:
        guard_name += '_cnn'

    #keras.backend.set_learning_phase(1) if 'bnn' in guard_name else keras.backend.set_learning_phase(0)
    y_logit_op = gen.predict(x, softmax=False) # Logits output (2 dimensions)
    
    # --- 3. Compute Logits for OOD and Clean Samples ---
    def compute_logits(data_x):
        logits = []
        n_batch = int(data_x.shape[0] / batch_size)
        for i in xrange(n_batch):
            X_batch = data_x[i*batch_size:(i+1)*batch_size]
            logits.append(sess.run(y_logit_op, feed_dict={x: X_batch}))
        return np.concatenate(logits, 0)
    
    # Compute logits for train data (baseline)
    y_logit_train = compute_logits(x_train)
    y_train_truncated = y_train[:y_logit_train.shape[0]]

    # Compute logits for clean ID test data
    y_logit_clean = compute_logits(x_clean)
    y_clean_2dim_truncated = y_clean_2dim[:y_logit_clean.shape[0]]

    # Compute logits for OOD data
    y_logit_ood = compute_logits(x_ood)
    y_ood_truncated = y_ood[:y_logit_ood.shape[0]] # Truncate OOD labels just in case

    # --- 4. Compute Statistics ---
    print('-------------------------------------')
    print('Compute statistics on data')
    
    # Train set statistics (Used as the baseline mean/std)
    results_train = comp_logp(y_logit_train, y_train_truncated, 'Train', comp_logit_dist=True)

    # Clean ID test set statistics
    results_clean = comp_logp(y_logit_clean, y_clean_2dim_truncated, 'ID Test')

    # OOD test set statistics
    # NOTE: The OOD labels (y_ood) are 10-dimensional. We use y_clean_2dim_truncated
    # as a placeholder since the conditional likelihood on OOD is unreliable/irrelevant.
    # The primary OOD metric is the marginal likelihood (results_ood[0]).
    results_ood = comp_logp(y_logit_ood, y_ood_truncated[:, :2], 'OOD Test') 
    
    # --- 5. Detection Rate Calculation (Focus on Log P(x)) ---
    
    # Since VAEs use likelihood, low log-likelihood suggests OOD (plus=False)
    plus = False

    # 1. False Positive Rate (Target 5%) on Clean ID data
    alpha, detect_rate = search_alpha(results_clean[0], results_train[1], results_train[2], plus=plus)
    print("-------------------------------------")
    print('False Alarm Rate (ID rejection):')
    print('FP Rate (reject < mean of logp(x) - %.2f * std): %.4f' % (alpha, detect_rate))
    
    # 2. True Positive Rate (Detection Rate) on OOD data
    detect_rate_ood = comp_detect(results_ood[0], results_train[1], results_train[2], alpha, plus=plus)
    print('OOD Detection Rate (reject < mean of logp(x) - %.2f * std): %.4f' % (alpha, detect_rate_ood))
    
    results = {}
    results['FP_logpx'] = detect_rate
    results['TP_logpx_OOD'] = detect_rate_ood
    
    # Close TF session
    sess.close()
    
    # --- 6. Save Results ---
    if save:
        if not os.path.isdir('detection_results/'):
            os.makedirs('detection_results/', exist_ok=True)
            print('create path detection_results/')
        path = 'detection_results/' + guard_name + '/'
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
            print('create path ' + path)
            
        filename = data_name + '_' + guard_name + '_OOD' # OOD specific filename
        
        pickle.dump(results, open(path + filename + '.pkl', 'wb'))
        print("results saved at %s.pkl" % (path + filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run OOD detection experiments.')
    parser.add_argument('--batch_size', '-B', type=int, default=100)
    parser.add_argument('--data', '-D', type=str, default='plane_frog') # ID data name
    parser.add_argument('--guard', '-G', type=str, default='bayes_K10') # The trained model
    parser.add_argument('--save', '-S', action='store_true', default=False)
    
    # Argument for 'conv' is removed as CIFAR is always convolutional here
    # Arguments for 'targeted', 'attack', 'victim' are removed as they are OOD specific

    args = parser.parse_args()
    # Note: We hardcode data_name to 'plane_frog' to ensure ID labels are [0, 6]
    test_ood(args.batch_size, args.guard, 'plane_frog', args.save)