from __future__ import division
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef, roc_curve, auc, cohen_kappa_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
from datetime import datetime
import time
import random
from efficiency_utils import (
    PROJECT_ROOT,
    PeakMemoryMonitor,
    create_run_output_dir,
    estimate_svm_complexity,
    save_efficiency_reports,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)



def calculate_feature_importance(model, X, y, feature_names, n_repeats=5):
    """Calculate SVM feature importance using permutation importance.

    Parameters:
    model: Trained SVM model.
    X: Feature data.
    y: Target labels.
    feature_names: List of feature names.
    n_repeats: Number of random permutations for each feature.

    Returns:
    normalized_importance: Normalized feature-importance scores.
    """
    # Calculate baseline accuracy.
    baseline_preds = model.predict(X)
    baseline_accuracy = accuracy_score(y, baseline_preds)

    importance_scores = []
    # Evaluate permutation importance for each feature.
    for i in range(X.shape[1]):
        feature_importance_sum = 0
        # Repeat the random permutation to reduce sampling variability.
        for _ in range(n_repeats):
            # Copy the data to avoid modifying the original array.
            X_permuted = X.copy()
            # Randomly permute the current feature.
            np.random.shuffle(X_permuted[:, i])
            # Evaluate the model on the permuted data.
            permuted_preds = model.predict(X_permuted)
            permuted_accuracy = accuracy_score(y, permuted_preds)
            # Importance is the decrease from baseline accuracy.
            importance = baseline_accuracy - permuted_accuracy
            feature_importance_sum += importance
        # Calculate mean importance.
        average_importance = feature_importance_sum / n_repeats
        importance_scores.append(average_importance)

    # Normalize importance scores to [0, 1] using MinMaxScaler.
    scaler = MinMaxScaler()
    normalized_importance = scaler.fit_transform(np.array(importance_scores).reshape(-1, 1)).flatten()
    
    return normalized_importance

def calculate_metrics(labels, predicted):
    accuracy = accuracy_score(labels, predicted)
    precision = precision_score(labels, predicted)
    recall = recall_score(labels, predicted)
    f1 = f1_score(labels, predicted)
    conf_matrix = confusion_matrix(labels, predicted)
    mcc = matthews_corrcoef(labels, predicted)
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp)
    kappa = cohen_kappa_score(labels, predicted)
    balanced_accuracy = balanced_accuracy_score(labels, predicted)
    return accuracy, precision, recall, f1, mcc, specificity, conf_matrix, kappa, balanced_accuracy

def train_and_evaluate(C, kernel, gamma, save_dir,
                        X_train, y_train, X_val,
                          y_val, X_test, y_test,
                            feature_names):
    """Train and evaluate the SVM model."""
    print("Starting SVM training...")
    # Initialize the SVM model.
    model = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        probability=True,
        random_state=SEED,
        verbose=True
    )
    
    # Save model parameters.
    params = {
        "Parameter": ["C", "kernel", "gamma", "random_state", "probability"],
        "Value": [C, kernel, gamma, SEED, True]
    }
    params_df = pd.DataFrame(params)
    params_file_path = os.path.join(save_dir, 'model_params.csv')
    params_df.to_csv(params_file_path, index=False)
    
    # Train the model.
    train_monitor = PeakMemoryMonitor()
    train_monitor.start()
    train_start_time = time.perf_counter()
    print("SVM training started; LibSVM will report internal iteration details.")
    model.fit(X_train, y_train)
    training_runtime_seconds = time.perf_counter() - train_start_time
    training_memory = train_monitor.stop()
    if hasattr(model, "n_iter_"):
        print(f"SVM training completed; internal iterations: {model.n_iter_}")
    
    # Obtain predicted probabilities and class labels.
    train_probs = model.predict_proba(X_train)[:, 1]
    train_predicted = model.predict(X_train)
    val_probs = model.predict_proba(X_val)[:, 1]
    val_predicted = model.predict(X_val)
    
    # Calculate evaluation metrics.
    train_accuracy, train_precision, train_recall, train_f1, train_mcc, train_specificity, train_conf_matrix, train_kappa, train_balanced_accuracy = calculate_metrics(y_train, train_predicted)
    val_accuracy, val_precision, val_recall, val_f1, val_mcc, val_specificity, val_conf_matrix, val_kappa, val_balanced_accuracy = calculate_metrics(y_val, val_predicted)
    
    # Calculate AUC.
    train_fpr, train_tpr, _ = roc_curve(y_train, train_probs)
    train_auc = auc(train_fpr, train_tpr)
    val_fpr, val_tpr, _ = roc_curve(y_val, val_probs)
    val_auc = auc(val_fpr, val_tpr)
    
    # Store metrics.
    train_metrics = [train_accuracy, train_precision, train_recall, train_f1, train_mcc, train_specificity, train_kappa, train_balanced_accuracy, train_auc, val_auc]
    val_metrics = [val_accuracy, val_precision, val_recall, val_f1, val_mcc, val_specificity, val_kappa, val_balanced_accuracy, train_auc, val_auc]
    
    print("Training-set metrics:")
    print(f"Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
    print(f"F1: {train_f1:.4f}, MCC: {train_mcc:.4f}, AUC: {train_auc:.4f}")
    
    print("Validation-set metrics:")
    print(f"Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
    print(f"F1: {val_f1:.4f}, MCC: {val_mcc:.4f}, AUC: {val_auc:.4f}")
    # Plot confusion matrices.
    datasets = ['Train', 'Validation']
    conf_matrices = [train_conf_matrix, val_conf_matrix]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, (dataset, conf_matrix) in enumerate(zip(datasets, conf_matrices)):
        ax = axes[i]
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{dataset} Confusion Matrix')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
    plt.tight_layout()
    save_path_confusion = os.path.join(save_dir, 'confusion_matrices.png')
    plt.savefig(save_path_confusion)
    
    # Plot ROC curves.
    plt.figure(figsize=(10, 8))
    plt.plot(train_fpr, train_tpr, label=f'Train ROC curve (area = {train_auc:.2f})')
    plt.plot(val_fpr, val_tpr, label=f'Validation ROC curve (area = {val_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves for Train, Validation sets')
    plt.legend(loc="lower right")
    save_path = os.path.join(save_dir, 'roc_curves.png')
    plt.savefig(save_path)
    
    # Save evaluation metrics.
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'MCC', 'Specificity', 'Kappa', 'Balanced Accuracy', 'Train AUC', 'Validation AUC']
    metrics_values = [train_metrics, val_metrics]
    metrics_df = pd.DataFrame(metrics_values, index=datasets, columns=metrics_names)
    metrics_csv_path = os.path.join(save_dir, 'evaluation_metrics.csv')
    metrics_df.to_csv(metrics_csv_path)
    print(f"Evaluation metrics saved to {metrics_csv_path}")

    inference_monitor = PeakMemoryMonitor()
    inference_monitor.start()
    inference_start_time = time.perf_counter()
    test_probs = model.predict_proba(X_test)[:, 1]
    test_predicted = model.predict(X_test)
    inference_runtime_seconds = time.perf_counter() - inference_start_time
    inference_memory = inference_monitor.stop()

    profile_batch_size = min(128, X_test.shape[0])
    complexity_metrics = estimate_svm_complexity(model, X_test.shape[1], profile_batch_size=profile_batch_size)
    efficiency_metrics = {
        'training_runtime_seconds': round(training_runtime_seconds, 4),
        'inference_runtime_seconds': round(inference_runtime_seconds, 4),
        'training_peak_memory_mb': training_memory['peak_memory_mb'],
        'training_peak_rss_mb': training_memory['peak_rss_mb'],
        'training_peak_cuda_allocated_mb': training_memory['peak_cuda_allocated_mb'],
        'training_peak_cuda_reserved_mb': training_memory['peak_cuda_reserved_mb'],
        'inference_peak_memory_mb': inference_memory['peak_memory_mb'],
        'inference_peak_rss_mb': inference_memory['peak_rss_mb'],
        'inference_peak_cuda_allocated_mb': inference_memory['peak_cuda_allocated_mb'],
        'inference_peak_cuda_reserved_mb': inference_memory['peak_cuda_reserved_mb'],
        'parameter_count': complexity_metrics['parameter_count'],
        'model_size_mb': complexity_metrics['model_size_mb'],
        'num_support_vectors': complexity_metrics['num_support_vectors'],
        'profile_batch_size': complexity_metrics['profile_batch_size'],
        'approx_flops_profile_batch': complexity_metrics['approx_flops_profile_batch'],
        'approx_flops_per_sample': complexity_metrics['approx_flops_per_sample'],
        'validation_recall': round(val_metrics[2], 4),
        'output_dir': save_dir,
    }
    efficiency_path, summary_path = save_efficiency_reports(save_dir, 'SVM', efficiency_metrics)
    print(f"Efficiency metrics saved to {efficiency_path}")
    print(f"Efficiency summary updated at {summary_path}")

    RUN_FEATURE_IMPORTANCE = False
    if RUN_FEATURE_IMPORTANCE:
        feature_importance = calculate_feature_importance(model, X_train, y_train, feature_names, n_repeats=5)
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        })
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        plt.figure(figsize=(12, 10))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, color='#66B2FF')
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        save_path = os.path.join(save_dir, 'feature_importance.png')
        plt.savefig(save_path)
        print(f"Feature-importance figure saved to {save_path}")
    else:
        print("Feature-importance analysis skipped.")
    # Predict and evaluate on the test set.
    RUN_PREDICTION_SAVE = True
    if RUN_PREDICTION_SAVE:
        # Calculate test-set metrics.
        test_accuracy, test_precision, test_recall, test_f1, test_mcc, test_specificity, test_conf_matrix, test_kappa, test_balanced_accuracy = calculate_metrics(y_test, test_predicted)
        
        # Calculate test-set AUC.
        test_fpr, test_tpr, _ = roc_curve(y_test, test_probs)
        test_auc = auc(test_fpr, test_tpr)
        print(f"Test AUC: {test_auc:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}, test recall: {test_recall:.4f}")
        
        # Plot the test-set ROC curve.
        plt.figure(figsize=(10, 8))
        plt.plot(test_fpr, test_tpr, label=f'Test ROC curve (area = {test_auc:.4f})', color='red')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve for Test set')
        plt.legend(loc="lower right")
        save_path_test_roc = os.path.join(save_dir, 'test_roc_curve.png')
        plt.savefig(save_path_test_roc)
        print(f"Test ROC curve saved to {save_path_test_roc}")
        
        # Save test-set predictions.
        global data
        if 'data' in globals():
            last_two_columns = data.iloc[:, -3:-1]
            result_df = pd.DataFrame({
                'Predicted_Probability': test_probs
            })
            final_result = pd.concat([result_df, last_two_columns], axis=1)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_file_path = os.path.join(save_dir, f'prediction_results_{timestamp}.csv')
            final_result.to_csv(save_file_path, index=False)
            print(f"Predictions saved to {save_file_path}")
    
    plt.close('all')
    return model, val_metrics[2]



def custom_split_samples(X, y, data):
    positive_indices = y == 1
    negative_indices = y == 0
    X_positive = X[positive_indices]
    y_positive = y[positive_indices]
    X_positive_train, X_positive_val, y_positive_train, y_positive_val = train_test_split(
        X_positive, y_positive, test_size=0.2, random_state=42
    )
    # Use the data argument instead of a global variable.
    negative_samples = data[negative_indices].nsmallest(33921, 'indicator')
    selected_negative_indices = negative_samples.index
    selected_X_negative = X[selected_negative_indices]
    selected_y_negative = y[selected_negative_indices]
    X_negative_train, X_negative_val, y_negative_train, y_negative_val = train_test_split(
        selected_X_negative, selected_y_negative, test_size=0.2, random_state=42
    )
    X_train = np.concatenate([X_positive_train, X_negative_train])
    y_train = np.concatenate([y_positive_train, y_negative_train])
    X_val = np.concatenate([X_positive_val, X_negative_val])
    y_val = np.concatenate([y_positive_val, y_negative_val])
    X_test = X
    y_test = y
    return X_train, X_val, X_test, y_train, y_val, y_test

def plot_feature_distribution(X, feature_names, dataset_name):
    df = pd.DataFrame(X, columns=feature_names)
    df.hist(bins=20, figsize=(15, 10))
    plt.suptitle(f'{dataset_name} Feature Distribution')
    plt.show()

def plot_label_distribution(y, dataset_name):
    positive_count = np.sum(y == 1)
    negative_count = np.sum(y == 0)
    label_dist = pd.DataFrame({
        'Sample Class': ['Positive Samples', 'Negative Samples'],
        'Count': [positive_count, negative_count]
    })
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x='Sample Class', y='Count', data=label_dist)
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.0f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 10),
                    textcoords='offset points')
    plt.title(f'{dataset_name} Positive and Negative Sample Distribution')
    plt.xlabel('Sample Class')
    plt.ylabel('Count')
    plt.show()

if __name__ == "__main__":
    start_time = time.perf_counter()
    dataset_path = os.path.join(PROJECT_ROOT, 'dataset.csv')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            "dataset.csv was not found in the release directory. "
            "Place the dataset file next to SVM.py before running the script."
        )
    data = pd.read_csv(dataset_path)
    feature_names = data.columns[2:-4]
    X = data.iloc[:, 2:-4].values
    y = data.iloc[:, -4].values
    coords = data.iloc[:, -3:-1].values
    positive_indices = np.where(y == 1)[0]
    for idx in positive_indices:
        current_point = coords[idx]
        distances = np.linalg.norm(coords - current_point, axis=1)
        sorted_indices = np.argsort(distances)
        nearest_9_indices = sorted_indices[1:11]
        y[nearest_9_indices] = 1
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    RUN_PREDICTION_SAVE = True
    RUN_plot_label_distribution = False
    # Pass the data argument explicitly.
    X_train, X_val, X_test, y_train, y_val, y_test = custom_split_samples(X, y, data)
    if RUN_plot_label_distribution:
        plot_label_distribution(y_train, 'Training Set')
        plot_label_distribution(y_val, 'Validation Set')
        plot_label_distribution(y_test, 'Test Set')
    
    # SVM hyperparameters.
    C = 1.0  # Regularization parameter.
    kernel = 'rbf'  # Kernel type: 'linear', 'rbf', 'poly', or 'sigmoid'.
    gamma = 'auto'  # Kernel coefficient: 'scale', 'auto', or a float.

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f'SVM_C_{C}_kernel_{kernel}_gamma_{gamma}_{timestamp}'
    save_dir = create_run_output_dir('SVM', run_name)
    
    # Train and evaluate the model.
    _, recall = train_and_evaluate(
        C, kernel, gamma, save_dir, X_train, y_train, X_val, y_val, X_test, y_test, feature_names
    )
    
    total_time = time.perf_counter() - start_time
    print(f"Validation recall: {recall:.4f}")
    print(f"Total runtime: {total_time / 60:.2f} min")
