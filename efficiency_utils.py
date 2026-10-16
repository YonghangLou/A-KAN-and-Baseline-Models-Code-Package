"""Shared efficiency profiling and output utilities for all released models."""

import os
import threading
import time

import pandas as pd

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def create_run_output_dir(model_name, run_name):
    model_root = ensure_dir(os.path.join(OUTPUT_ROOT, model_name))
    return ensure_dir(os.path.join(model_root, run_name))


class PeakMemoryMonitor:
    def __init__(self, interval=0.05):
        self.interval = interval
        self._running = False
        self._thread = None
        self._peak_rss_bytes = 0
        self._peak_cuda_allocated_bytes = 0
        self._peak_cuda_reserved_bytes = 0
        self._process = psutil.Process(os.getpid()) if psutil else None

    def _sample_once(self):
        if self._process is not None:
            rss_bytes = self._process.memory_info().rss
            self._peak_rss_bytes = max(self._peak_rss_bytes, rss_bytes)

        if torch is not None and torch.cuda.is_available():
            allocated_bytes = torch.cuda.memory_allocated()
            reserved_bytes = torch.cuda.memory_reserved()
            self._peak_cuda_allocated_bytes = max(self._peak_cuda_allocated_bytes, allocated_bytes)
            self._peak_cuda_reserved_bytes = max(self._peak_cuda_reserved_bytes, reserved_bytes)

    def _run(self):
        while self._running:
            self._sample_once()
            time.sleep(self.interval)

    def start(self):
        self._peak_rss_bytes = 0
        self._peak_cuda_allocated_bytes = 0
        self._peak_cuda_reserved_bytes = 0
        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._running = True
        self._sample_once()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join()
        self._sample_once()
        return self.snapshot()

    def snapshot(self):
        peak_rss_mb = self._peak_rss_bytes / (1024 ** 2)
        peak_cuda_allocated_mb = self._peak_cuda_allocated_bytes / (1024 ** 2)
        peak_cuda_reserved_mb = self._peak_cuda_reserved_bytes / (1024 ** 2)
        peak_memory_mb = max(peak_rss_mb, peak_cuda_reserved_mb)
        return {
            "peak_rss_mb": round(peak_rss_mb, 4),
            "peak_cuda_allocated_mb": round(peak_cuda_allocated_mb, 4),
            "peak_cuda_reserved_mb": round(peak_cuda_reserved_mb, 4),
            "peak_memory_mb": round(peak_memory_mb, 4),
        }


def count_trainable_parameters(model):
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def estimate_torch_model_size_mb(model):
    total_bytes = 0
    for parameter in model.parameters():
        total_bytes += parameter.numel() * parameter.element_size()
    for buffer in model.buffers():
        total_bytes += buffer.numel() * buffer.element_size()
    return round(total_bytes / (1024 ** 2), 4)


def _count_linear_flops(module, inputs, outputs):
    input_tensor = inputs[0]
    batch_items = 1
    for dim in input_tensor.shape[:-1]:
        batch_items *= dim
    return int(batch_items * module.in_features * module.out_features * 2)


def _count_conv1d_flops(module, inputs, outputs):
    output_tensor = outputs
    batch_size, out_channels, output_length = output_tensor.shape
    kernel_ops = (module.in_channels // module.groups) * module.kernel_size[0] * 2
    return int(batch_size * out_channels * output_length * kernel_ops)


def _count_relu_like_flops(outputs, ops_per_element=1):
    return int(outputs.numel() * ops_per_element)


def _count_maxpool1d_flops(module, outputs):
    batch_size, channels, output_length = outputs.shape
    kernel_size = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
    return int(batch_size * channels * output_length * kernel_size)


def _count_softmax_flops(outputs):
    return int(outputs.numel() * 5)


def _count_kanlinear_flops(module, inputs, outputs):
    input_tensor = inputs[0]
    batch_size = input_tensor.shape[0]
    basis_terms = module.grid_size + module.spline_order
    base_linear_flops = batch_size * module.in_features * module.out_features * 2
    spline_linear_flops = batch_size * module.in_features * basis_terms * module.out_features * 2
    spline_basis_flops = batch_size * module.in_features * basis_terms * (6 * module.spline_order + 4)
    return int(base_linear_flops + spline_linear_flops + spline_basis_flops)


def _count_toptheta_attention_flops(module, inputs, outputs):
    input_tensor = inputs[0]
    batch_size = input_tensor.shape[0]
    attention_width = module.value.out_features
    score_flops = batch_size * batch_size * attention_width * 2
    softmax_flops = batch_size * batch_size * 5
    masking_flops = batch_size * batch_size * 2
    weighted_sum_flops = batch_size * batch_size * attention_width * 2
    return int(score_flops + softmax_flops + masking_flops + weighted_sum_flops)


def estimate_torch_flops(model, sample_input):
    if torch is None or nn is None:
        return {"profile_batch_size": 0, "approx_flops_profile_batch": 0, "approx_flops_per_sample": 0}

    total_flops = 0
    hooks = []

    def hook_fn(module, inputs, outputs):
        nonlocal total_flops
        class_name = module.__class__.__name__
        if isinstance(module, nn.Linear):
            total_flops += _count_linear_flops(module, inputs, outputs)
        elif isinstance(module, nn.Conv1d):
            total_flops += _count_conv1d_flops(module, inputs, outputs)
        elif isinstance(module, nn.ReLU):
            total_flops += _count_relu_like_flops(outputs, ops_per_element=1)
        elif isinstance(module, nn.SiLU):
            total_flops += _count_relu_like_flops(outputs, ops_per_element=4)
        elif isinstance(module, nn.MaxPool1d):
            total_flops += _count_maxpool1d_flops(module, outputs)
        elif isinstance(module, nn.Softmax):
            total_flops += _count_softmax_flops(outputs)
        elif class_name == "KANLinear":
            total_flops += _count_kanlinear_flops(module, inputs, outputs)
        elif class_name == "TopThetaAttention":
            total_flops += _count_toptheta_attention_flops(module, inputs, outputs)

    for module in model.modules():
        hooks.append(module.register_forward_hook(hook_fn))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(sample_input)
    if was_training:
        model.train()

    for hook in hooks:
        hook.remove()

    profile_batch_size = int(sample_input.shape[0]) if hasattr(sample_input, "shape") else 0
    flops_per_sample = total_flops / profile_batch_size if profile_batch_size else 0
    return {
        "profile_batch_size": profile_batch_size,
        "approx_flops_profile_batch": int(total_flops),
        "approx_flops_per_sample": round(flops_per_sample, 4),
    }


def estimate_svm_complexity(model, input_dim, profile_batch_size=1):
    support_vectors = model.support_vectors_
    dual_coef = model.dual_coef_
    intercept = model.intercept_
    degree = getattr(model, "degree", 3)
    num_support_vectors = int(support_vectors.shape[0])
    parameter_count = int(support_vectors.size + dual_coef.size + intercept.size)
    model_size_mb = round(
        (support_vectors.nbytes + dual_coef.nbytes + intercept.nbytes) / (1024 ** 2),
        4,
    )

    kernel = getattr(model, "kernel", "rbf")
    if kernel == "linear":
        flops_per_sample = num_support_vectors * (2 * input_dim + 1)
    elif kernel == "poly":
        flops_per_sample = num_support_vectors * (2 * input_dim + degree + 8)
    elif kernel == "sigmoid":
        flops_per_sample = num_support_vectors * (2 * input_dim + 10)
    else:
        flops_per_sample = num_support_vectors * (4 * input_dim + 6)

    return {
        "profile_batch_size": int(profile_batch_size),
        "parameter_count": parameter_count,
        "model_size_mb": model_size_mb,
        "num_support_vectors": num_support_vectors,
        "approx_flops_profile_batch": int(flops_per_sample * profile_batch_size),
        "approx_flops_per_sample": round(float(flops_per_sample), 4),
    }


def save_efficiency_reports(save_dir, model_name, metrics):
    metrics_with_model = {"model_name": model_name}
    metrics_with_model.update(metrics)

    efficiency_path = os.path.join(save_dir, "efficiency_metrics.csv")
    pd.DataFrame([metrics_with_model]).to_csv(efficiency_path, index=False)

    summary_path = os.path.join(OUTPUT_ROOT, "model_efficiency_summary.csv")
    ensure_dir(OUTPUT_ROOT)
    summary_df = pd.DataFrame([metrics_with_model])
    if os.path.exists(summary_path):
        existing_df = pd.read_csv(summary_path)
        summary_df = pd.concat([existing_df, summary_df], ignore_index=True)
    summary_df.to_csv(summary_path, index=False)

    return efficiency_path, summary_path
