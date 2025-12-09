import numpy as np
from six.moves import xrange
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os, sys, pickle, argparse
import matplotlib.pyplot as plt # <--- NEW IMPORT

# Ensure paths are correct for Colab execution
sys.path.append('../utils/')
sys.path.append('load/')

# Import necessary utilities (assuming these are in the correct paths)
from load_classifier import load_classifier
from import_data_cifar10 import load_data_cifar10

# --- Typicality Test Functions ---

def compute_empirical_entropy(logit):
    """
    Computes the empirical entropy for a set of samples X: H_hat = 1/N * sum(-log p(x_n))
    This returns the negative log-likelihood scores.
    """
    # log p(x) for each sample is the input logit
    neg_logpx = -logit
    return neg_logpx

def compute_test_statistic(neg_logpx_batch, resubstitution_entropy_hat):
    """
    Computes the typicality test statistic (Eq. 3 in paper): 
    | 1/M * sum(-log p(x_m)) - H_hat_RESUB | = epsilon_hat
    """
    # Calculate the sample entropy for the batch
    sample_entropy_hat = np.mean(neg_logpx_batch)
    
    # Calculate the test statistic (epsilon_hat)
    epsilon_hat = np.abs(sample_entropy_hat - resubstitution_entropy_hat)
    return epsilon_hat

# --- New Helper Functions for Visualization ---

def compute_all_epsilon_scores(sess, y_logit_op, x_placeholder, data_x, M, H_hat_RESUB):
    """Computes and stores epsilon_hat scores for all full M-sized batches."""
    epsilon_scores = []
    n_samples = data_x.shape[0]
    n_batches = int(n_samples / M)
    
    for i in xrange(n_batches):
        X_batch = data_x[i*M:(i+1)*M]
        
        # Compute -log p(x) for the batch
        logit_batch = sess.run(y_logit_op, feed_dict={x_placeholder: X_batch})
        neg_logpx_batch = compute_empirical_entropy(logit_batch)
        
        # Compute the test statistic (epsilon_hat)
        epsilon_hat = compute_test_statistic(neg_logpx_batch, H_hat_RESUB)
        epsilon_scores.append(epsilon_hat)
            
    return epsilon_scores, n_batches

def detect_typicality_rate_from_scores(epsilon_scores, n_batches, epsilon_alpha_M, is_ood_set=False):
    """Calculates FP/TP rate based on collected epsilon scores."""
    if n_batches == 0:
        detection_rate = 0.0
    else:
        batches_classified_ood = np.sum(np.array(epsilon_scores) > epsilon_alpha_M)
        detection_rate = (batches_classified_ood / n_batches) * 100
    
    if is_ood_set:
        print('OOD Detection Rate (TP Rate) on %d batches: %.4f %%' % (n_batches, detection_rate))
    else:
        print('False Alarm Rate (FP Rate) on %d batches: %.4f %%' % (n_batches, detection_rate))
        
    return detection_rate

# --- Core OOD Detection Logic ---

def test_ood(batch_size, guard_name, data_name, save, alpha=0.99, K_bootstrap=50):
    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)
    # Create TF session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    print("Created TensorFlow session.")

    # --- 1. Load ALL Data and Split into ID / OOD ---
    img_rows, img_cols, channels = 32, 32, 3
    datapath = '../cifar_data/' 
    
    # Load FULL CIFAR-10 data (10 classes, 10-dim one-hot) for ID/OOD separation
    x_train_full, X_all_test, y_train_full, Y_all_test = load_data_cifar10(datapath, labels=None)

    # Filter the data into ID (Plane/Frog) and OOD (Other) sets
    id_labels = [0, 6]  # Plane and Frog
    all_labels = np.argmax(Y_all_test, axis=1)

    id_indices = np.where((all_labels == id_labels[0]) | (all_labels == id_labels[1]))[0]
    ood_indices = np.where((all_labels != id_labels[0]) & (all_labels != id_labels[1]))[0]
    
    # Load the ID training set used by the model (2 classes, 2-dim one-hot)
    x_train, X_test_full_ID, y_train, Y_test_full_ID = load_data_cifar10(datapath, labels=id_labels)

    # Final Test Sets (ID labels are NOT used here, only the images)
    x_clean = X_all_test[id_indices] # In-distribution (ID) test set
    x_ood = X_all_test[ood_indices] # Out-of-Distribution (OOD) test set
    
    nb_classes = y_train.shape[1] # Should be 2
    
    print('Loaded ID (Train/Test): %d/%d, OOD Samples: %d' % 
          (x_train.shape[0], x_clean.shape[0], x_ood.shape[0]))
    
    # --- 1.1 Create Validation Set for Bootstrap (M=batch_size) ---
    # We will use the second half of the ID test set (2000 samples total) as X'
    x_val = x_clean[1000:]
    x_clean = x_clean[:1000] # Use the first half for final ID evaluation
    
    M = batch_size # The batch size is the sequence length M in the paper
    
    # --- 2. Define Placeholders and Load Model ---
    x = tf.placeholder(tf.float32, shape=(M, img_rows, img_cols, channels))
    
    gen = load_classifier(sess, guard_name, data_name)
    
    # Get the logit operation (which outputs log p(x) for VAE/Flows)
    y_logit_op = gen.predict(x, softmax=False) # Logits output (log p(x) for generative models)
    
    # --- 3. Compute Logits and Entropy Estimator (Offline step 1) ---
    
    def compute_logits_all(data_x):
        """Computes logits for all samples, handling partial batching (M)."""
        logits = []
        n_samples = data_x.shape[0]
        n_batch = int(n_samples / M)
        
        # Process full batches
        for i in xrange(n_batch):
            X_batch = data_x[i*M:(i+1)*M]
            logits.append(sess.run(y_logit_op, feed_dict={x: X_batch}))
            
        # Handle the final partial batch (pad to size M if necessary)
        n_remaining = n_samples % M
        if n_remaining > 0:
             # Create a partial batch and pad it with a few samples from the beginning
             # to make it size M (This ensures a successful run of sess.run(y_logit_op)
             # with the fixed size M placeholder, though the statistics must be corrected).
            X_partial = data_x[n_batch*M:]
            X_padded = np.concatenate([X_partial, data_x[:M - n_remaining]], axis=0)
            
            # Run the padded batch
            logit_padded = sess.run(y_logit_op, feed_dict={x: X_padded})
            
            # Only keep the results for the actual remaining samples
            logits.append(logit_padded[:n_remaining])

        return np.concatenate(logits, 0)

    # 3.1 Compute logits for training data (to estimate model entropy)
    y_logit_train = compute_logits_all(x_train)
    
    # 3.2 Compute the Resubstitution Entropy Estimator (Eq. 5, Offline step 1)
    # H_hat_RESUB = 1/N * sum(-log p(x_n))
    neg_logpx_train = compute_empirical_entropy(y_logit_train)
    H_hat_RESUB = np.mean(neg_logpx_train)
    print("\n-------------------------------------")
    print('Resubstitution Entropy Estimate (H_hat_RESUB): %.3f' % H_hat_RESUB)

    # --- 4. Compute Bootstrap Distribution (Offline steps 2-4) ---
    
    print('Computing Bootstrap Distribution (K=%d, M=%d)...' % (K_bootstrap, M))
    
    epsilon_hat_bootstrap = []
    n_val = x_val.shape[0]

    # Perform K bootstrap samples
    for k in xrange(K_bootstrap):
        # 4.1 Sample M-sized batch from validation set X' (with replacement)
        idx = np.random.choice(n_val, size=M, replace=True)
        X_k_prime = x_val[idx]
        
        # 4.2 Compute -log p(x) for the batch X_k_prime (Requires fixed size M)
        logit_k = sess.run(y_logit_op, feed_dict={x: X_k_prime}) 
        neg_logpx_k = compute_empirical_entropy(logit_k)
        
        # 4.3 Compute the test statistic epsilon_hat_k (Eq. 6)
        epsilon_hat_k = compute_test_statistic(neg_logpx_k, H_hat_RESUB)
        epsilon_hat_bootstrap.append(epsilon_hat_k)
        
    # 4.4 Set the rejection threshold epsilon_alpha^M (Offline step 4)
    epsilon_alpha_M = np.quantile(epsilon_hat_bootstrap, alpha)
    print('Rejection Threshold (epsilon_%.2f^M): %.3f' % (alpha, epsilon_alpha_M))
    print("-------------------------------------")

    # --- 5. OOD Detection & Score Collection (Online steps) ---
    
    # Collect epsilon_hat for visualization
    id_epsilon_scores, n_batches_id = compute_all_epsilon_scores(sess, y_logit_op, x, x_clean, M, H_hat_RESUB)
    ood_epsilon_scores, n_batches_ood = compute_all_epsilon_scores(sess, y_logit_op, x, x_ood, M, H_hat_RESUB)

    # 5.2 Calculate FP Rate (ID Rejection) - Should be low (Type-I Error)
    fp_rate = detect_typicality_rate_from_scores(id_epsilon_scores, n_batches_id, epsilon_alpha_M, is_ood_set=False)
    
    # 5.3 Calculate TP Rate (OOD Detection) - Should be high
    tp_rate_ood = detect_typicality_rate_from_scores(ood_epsilon_scores, n_batches_ood, epsilon_alpha_M, is_ood_set=True)
    
    results = {}
    results['FP_typicality'] = fp_rate
    results['TP_typicality_OOD'] = tp_rate_ood
    
    # --- 6. Visualization ---
    print('\nGenerating Typicality Score Histogram...')

    plt.figure(figsize=(10, 6))
    
    # Plot OOD scores
    plt.hist(ood_epsilon_scores, bins=50, density=True, alpha=0.6, 
             label='OOD Batches (All CIFAR-10 except Plane/Frog)', color='red')
    
    # Plot ID scores
    plt.hist(id_epsilon_scores, bins=50, density=True, alpha=0.6, 
             label='ID Batches (Plane/Frog)', color='blue')
    
    # Draw the critical rejection threshold
    # Note: Using r"..." for raw strings to prevent SyntaxWarning on escape sequences like \h
    plt.axvline(epsilon_alpha_M, color='k', linestyle='--', linewidth=2, 
                label=r'Rejection Threshold $\epsilon_{%.2f}^M = %.3f$' % (alpha, epsilon_alpha_M))

    plt.xlabel(r'Typicality Score ($\hat{\epsilon}$) - Deviation from Model Entropy', fontsize=12)
    plt.ylabel('Normalized Frequency', fontsize=12)
    plt.title(r'Distribution of Typicality Scores ($\hat{\epsilon}$) - Batch Size M=%d' % M, fontsize=14)
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    
    # --- FIX: Save the plot to a file instead of showing it interactively ---
    plot_filename = 'typicality_histogram_%s_M%d_alpha%s.png' % (guard_name, M, str(alpha).replace('.', ''))
    plt.savefig(plot_filename)
    print("Histogram saved as %s" % plot_filename)
    
    sess.close() 
    
    # --- 7. Save Results ---
    if save:
        if not os.path.isdir('detection_results/'):
            os.makedirs('detection_results/', exist_ok=True)
        path = 'detection_results/' + guard_name + '/'
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
        
        filename = data_name + '_' + guard_name + '_typicality_OOD'
        
        pickle.dump(results, open(path + filename + '.pkl', 'wb'))
        print("results saved at %s.pkl" % (path + filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Typicality Test for OOD detection.')
    parser.add_argument('--batch_size', '-B', type=int, default=100)
    parser.add_argument('--data', '-D', type=str, default='plane_frog')
    parser.add_argument('--guard', '-G', type=str, default='bayes_K10')
    parser.add_argument('--save', '-S', action='store_true', default=False)
    parser.add_argument('--alpha', '-A', type=float, default=0.99, 
                        help='Confidence level for the bootstrap test (1 - FP rate).')
    parser.add_argument('--K_bootstrap', '-K', type=int, default=50, 
                        help='Number of bootstrap samples (K) for threshold setting.')
    
    args = parser.parse_args()
    
    test_ood(args.batch_size, args.guard, 'plane_frog', args.save, args.alpha, args.K_bootstrap)