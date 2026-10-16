from __future__ import division
from __future__ import print_function
import torch
import torch.nn.functional as F
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef, roc_curve, auc, cohen_kappa_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
from datetime import datetime
import time
import random
from efficiency_utils import (
    PROJECT_ROOT,
    PeakMemoryMonitor,
    count_trainable_parameters,
    create_run_output_dir,
    estimate_torch_flops,
    estimate_torch_model_size_mb,
    save_efficiency_reports,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

import torch.nn as nn



class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(
            0, 1
        )
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(
            A, B
        ).solution
        result = solution.permute(
            2, 0, 1
        )
        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.shape[-1] == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)
        splines = self.b_splines(x)
        splines = splines.permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight
        orig_coeff = orig_coeff.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )
        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1]
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for i, layer in enumerate(self.layers):
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

def calculate_feature_importance(model):
    num_features = model.layers[0].in_features
    feature_importance = np.zeros(num_features)
    for layer in model.layers:
        base_weight_importance = layer.base_weight.abs().sum(dim=0).detach().numpy()
        spline_weight_importance = layer.scaled_spline_weight.abs().sum(dim=(0, 2)).detach().numpy()
        layer_importance = base_weight_importance + spline_weight_importance
        if len(layer_importance) < num_features:
            padded_layer_importance = np.zeros(num_features)
            padded_layer_importance[:len(layer_importance)] = layer_importance
            layer_importance = padded_layer_importance
        elif len(layer_importance) > num_features:
            layer_importance = layer_importance[:num_features]
        feature_importance += layer_importance
    return feature_importance

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

def train_and_evaluate(lr, save_dir, num_epochs,
                        X_train, y_train, X_val,
                          y_val, X_test,y_test,
                            feature_names):
    train_losses = []
    train_metrics = []
    train_aucs = []
    train_accuracies = []
    val_losses = []
    val_metrics = []
    val_aucs = []
    val_accuracies = []
    model = KAN(
        layers_hidden=[X_train.shape[1], 64, 32, 2]
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    threshold = 0.5
    params = {
            "Parameter": ["threshold"],
            "Value": [threshold]
        }
    params_df = pd.DataFrame(params)
    params_file_path = os.path.join(save_dir, 'model_params.csv')
    params_df.to_csv(params_file_path, index=False)
    batch_size=7000
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True)
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    val_loader = DataLoader(val_dataset, batch_size=3500, shuffle=True)
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=30000, shuffle= False)
    train_monitor = PeakMemoryMonitor()
    train_monitor.start()
    train_start_time = time.perf_counter()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_probs = []
        train_labels = []
        all_inputs = []
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)[:, 1]
            loss = criterion(outputs, labels)
            total_loss = loss
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
            train_probs.extend(probabilities.cpu().tolist())
            train_labels.extend(labels.cpu().tolist())
            all_inputs.extend(inputs.cpu().numpy())
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        train_predicted = [1 if prob > threshold else 0 for prob in train_probs]
        train_accuracy, train_precision, train_recall, train_f1, train_mcc, train_specificity, train_conf_matrix, train_kappa, train_balanced_accuracy = calculate_metrics(train_labels, train_predicted)
        train_metrics.append([train_accuracy, train_precision, train_recall, train_f1, train_mcc, train_specificity, train_kappa, train_balanced_accuracy])
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy * 100:.2f}%, Train Recall: {train_recall:.4f}')
        train_fpr, train_tpr, _ = roc_curve(train_labels, train_probs)
        train_auc = auc(train_fpr, train_tpr)
        train_aucs.append(train_auc)
        train_accuracies.append(train_accuracy)
        
        model.eval()
        val_running_loss = 0.0
        val_probs = []
        val_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                probabilities = F.softmax(outputs, dim=1)[:, 1]
                loss = criterion(outputs, labels)
                val_probs.extend(probabilities.cpu().tolist())
                val_labels.extend(labels.cpu().tolist())
                val_running_loss += loss.item()
            val_loss = val_running_loss / len(val_loader)
            val_losses.append(val_loss)
        val_predicted = [1 if prob > 0.5 else 0 for prob in val_probs]
        val_accuracy, val_precision, val_recall, val_f1, val_mcc, val_specificity, val_conf_matrix, val_kappa, val_balanced_accuracy = calculate_metrics(val_labels, val_predicted)
        val_metrics.append([val_accuracy, val_precision, val_recall, val_f1, val_mcc, val_specificity, val_kappa, val_balanced_accuracy])
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy * 100:.2f}%, Validation Recall: {val_recall:.4f}')

        val_fpr, val_tpr, _ = roc_curve(val_labels, val_probs)
        val_auc = auc(val_fpr, val_tpr)
        val_aucs.append(val_auc)
        val_accuracies.append(val_accuracy)
    training_runtime_seconds = time.perf_counter() - train_start_time
    training_memory = train_monitor.stop()
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
    train_fpr, train_tpr, _ = roc_curve(train_labels, train_probs)
    train_auc = auc(train_fpr, train_tpr)
    val_fpr, val_tpr, _ = roc_curve(val_labels, val_probs)
    val_auc = auc(val_fpr, val_tpr)
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
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'MCC', 'Specificity', 'Kappa', 'Balanced Accuracy', 'Train AUC', 'Validation AUC']
    train_final_auc = train_aucs[-1]
    val_final_auc = val_aucs[-1]
    train_metrics[-1].extend([train_final_auc, val_final_auc])
    val_metrics[-1].extend([train_final_auc, val_final_auc])
    metrics_values = [train_metrics[-1], val_metrics[-1]]
    print("\nTraining and validation completed. Evaluation metrics:")
    for i, dataset in enumerate(datasets):
        print(f"\n{dataset} dataset metrics:")
        for j, metric_name in enumerate(metrics_names):
            print(f"{metric_name}: {metrics_values[i][j]:.4f}")
    metrics_df = pd.DataFrame(metrics_values, index=datasets, columns=metrics_names)
    metrics_csv_path = os.path.join(save_dir, 'evaluation_metrics.csv')
    metrics_df.to_csv(metrics_csv_path)
    print(f"\nEvaluation metrics saved to {metrics_csv_path}")

    RUN_PREDICTION_SAVE = True
    if RUN_PREDICTION_SAVE:
        model.eval()
        test_probs = []
        test_labels = []
        inference_monitor = PeakMemoryMonitor()
        inference_monitor.start()
        inference_start_time = time.perf_counter()
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                probabilities = F.softmax(outputs, dim=1)[:, 1]
                test_probs.extend(probabilities.cpu().tolist())
                test_labels.extend(labels.cpu().tolist())
        inference_runtime_seconds = time.perf_counter() - inference_start_time
        inference_memory = inference_monitor.stop()

    profile_batch_size = min(128, X_test.shape[0])
    sample_input = torch.tensor(X_test[:profile_batch_size], dtype=torch.float32)
    complexity_metrics = estimate_torch_flops(model, sample_input)
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
        'parameter_count': count_trainable_parameters(model),
        'model_size_mb': estimate_torch_model_size_mb(model),
        'profile_batch_size': complexity_metrics['profile_batch_size'],
        'approx_flops_profile_batch': complexity_metrics['approx_flops_profile_batch'],
        'approx_flops_per_sample': complexity_metrics['approx_flops_per_sample'],
        'validation_recall': round(val_metrics[-1][2], 4),
        'output_dir': save_dir,
    }
    efficiency_path, summary_path = save_efficiency_reports(save_dir, 'KAN', efficiency_metrics)
    print(f"Efficiency metrics saved to {efficiency_path}")
    print(f"Efficiency summary updated at {summary_path}")
    RUN_FEATURE_IMPORTANCE = False

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.tight_layout()
    save_path_loss_acc = os.path.join(save_dir, 'loss_accuracy_curves.png')
    plt.savefig(save_path_loss_acc)
    plt.figure(figsize=(10, 8))
    plt.plot(range(num_epochs), train_aucs, label='Train AUC')
    plt.plot(range(num_epochs), val_aucs, label='Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('AUC vs Epoch')
    plt.legend()
    save_path_auc = os.path.join(save_dir, 'auc_vs_epoch.png')
    plt.savefig(save_path_auc)
    train_recalls = [metric[2] for metric in train_metrics]
    val_recalls = [metric[2] for metric in val_metrics]
    plt.figure(figsize=(10, 8))
    plt.plot(range(num_epochs), train_recalls, label='Train Recall')
    plt.plot(range(num_epochs), val_recalls, label='Validation Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Recall vs Epoch')
    plt.legend()
    save_path_recall = os.path.join(save_dir, 'recall_vs_epoch.png')
    plt.savefig(save_path_recall)

    if RUN_FEATURE_IMPORTANCE:
        feature_importance = calculate_feature_importance(model)
        scaler = MinMaxScaler()
        feature_importance = scaler.fit_transform(feature_importance.reshape(-1, 1)).flatten()
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
    else:
        print("Feature-importance analysis skipped.")
    for layer_index, layer in enumerate(model.layers):
        x_values = np.linspace(layer.grid.min(), layer.grid.max(), 100)
        num_features = layer.in_features
        num_cols = 4
        num_rows = int(np.ceil(num_features / num_cols))
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 3))
        fig.suptitle(f'Single Variable Functions for All Input Features in Layer {layer_index}')
        for feature_index in range(num_features):
            row = feature_index // num_cols
            col = feature_index % num_cols
            ax = axes[row, col] if num_rows > 1 else axes[col]
            single_feature_x = torch.zeros((len(x_values), num_features))
            single_feature_x[:, feature_index] = torch.tensor(x_values, dtype=torch.float32)
            with torch.no_grad():
                single_feature_output = layer(single_feature_x)[:, 0].numpy()
            ax.plot(x_values, single_feature_output)
            ax.set_title(f'Feature {feature_index}')
            ax.set_xlabel('Input Value')
            ax.set_ylabel('Output Value')
        if num_features % num_cols != 0:
            for k in range(num_features % num_cols, num_cols):
                axes[-1, k].axis('off')
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'single_variable_functions_layer_{layer_index}.png')
        plt.savefig(save_path)
    test_fpr, test_tpr, _ = roc_curve(y_test, test_probs)
    test_auc = auc(test_fpr, test_tpr)
    print(f"Test AUC: {test_auc}")
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
    return model, val_metrics[-1][2]



def custom_split_samples(X, y):
    positive_indices = y == 1
    negative_indices = y == 0
    X_positive = X[positive_indices]
    y_positive = y[positive_indices]
    X_positive_train, X_positive_val, y_positive_train, y_positive_val = train_test_split(
        X_positive, y_positive, test_size=0.2, random_state=42
    )
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
            "Place the dataset file next to KAN.py before running the script."
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
    X_train, X_val, X_test, y_train, y_val, y_test = custom_split_samples(X, y)
    if RUN_plot_label_distribution:
        plot_label_distribution(y_train, 'Training Set')
        plot_label_distribution(y_val, 'Validation Set')
        plot_label_distribution(y_test, 'Test Set')
    lr = 1e-3
    num_epochs = 500 #100

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f'KAN_num_epochs_{num_epochs}_{timestamp}_lr_{lr}'
    save_dir = create_run_output_dir('KAN', run_name)
    _, recall = train_and_evaluate(
        lr, save_dir, num_epochs, X_train, y_train, X_val, y_val, X_test, y_test, feature_names
    )
    total_time = time.perf_counter() - start_time
    print(f"Validation recall: {recall:.4f}")
    print(f"Total runtime: {total_time / 60:.2f} min")
