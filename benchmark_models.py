import os
import time
import torch
import psutil
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB

def benchmark_model(model_path, device, num_runs=100, warmup_runs=20):
    device_str = "GPU" if "cuda" in str(device) else "CPU"
    print(f"Benchmarking {model_path.name} on {device_str}...")
    
    # Measure file size
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    
    # Measure memory before loading
    mem_before = get_process_memory()
    gpu_mem_before = 0
    if "cuda" in str(device) and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gpu_mem_before = torch.cuda.memory_allocated() / (1024 * 1024)
    
    # Load model
    try:
        model = YOLO(str(model_path))
    except Exception as e:
        print(f"Error loading {model_path.name}: {e}")
        return None
        
    # Measure memory after loading
    mem_after_load = get_process_memory()
    gpu_mem_after_load = 0
    if "cuda" in str(device) and torch.cuda.is_available():
        gpu_mem_after_load = torch.cuda.memory_allocated() / (1024 * 1024)
    
    model_ram_mb = mem_after_load - mem_before
    model_vram_mb = gpu_mem_after_load - gpu_mem_before
    
    # Prepare dummy input (using the default bus.jpg or a blank tensor)
    img_path = Path("/home/luke/ai_training/ultralytics/bus.jpg")
    if not img_path.exists():
        img_input = np.zeros((640, 640, 3), dtype=np.uint8)
    else:
        img_input = str(img_path)
        
    # Warmup
    print("Running warmup...")
    for _ in range(warmup_runs):
        try:
            model.predict(img_input, device=device, verbose=False)
        except Exception as e:
            print(f"Warmup failed on {device_str}: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "file_size_mb": file_size_mb,
                "ram_loaded_mb": model_ram_mb,
                "vram_loaded_mb": model_vram_mb
            }
            
    # Benchmark loop
    print("Running benchmark...")
    latencies = []
    
    # Memory monitoring during inference
    peak_ram = mem_after_load
    peak_vram = gpu_mem_after_load
    
    for i in range(num_runs):
        t0 = time.perf_counter()
        model.predict(img_input, device=device, verbose=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms
        
        # Periodically sample memory
        if i % 10 == 0:
            peak_ram = max(peak_ram, get_process_memory())
            if "cuda" in str(device) and torch.cuda.is_available():
                peak_vram = max(peak_vram, torch.cuda.memory_allocated() / (1024 * 1024))
                
    latencies = np.array(latencies)
    
    # Calculate stats
    avg_latency = np.mean(latencies)
    median_latency = np.median(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    throughput = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    
    inference_ram_growth = peak_ram - mem_after_load
    inference_vram_growth = peak_vram - gpu_mem_after_load if "cuda" in str(device) else 0.0
    
    # Clean up model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return {
        "status": "success",
        "file_size_mb": file_size_mb,
        "ram_loaded_mb": model_ram_mb,
        "vram_loaded_mb": model_vram_mb,
        "inference_ram_peak_growth_mb": inference_ram_growth,
        "inference_vram_peak_growth_mb": inference_vram_growth,
        "latency_avg_ms": avg_latency,
        "latency_median_ms": median_latency,
        "latency_p95_ms": p95_latency,
        "latency_p99_ms": p99_latency,
        "throughput_fps": throughput
    }

def main():
    pt_path = Path("/home/luke/ai_training/ultralytics/yolo26n.pt")
    onnx_path = Path("/home/luke/ai_training/ultralytics/yolo26n.onnx")
    
    results = {}
    
    # Run CPU benchmarks
    results["PyTorch_CPU"] = benchmark_model(pt_path, device="cpu")
    results["ONNX_CPU"] = benchmark_model(onnx_path, device="cpu")
    
    # Run GPU benchmarks if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available. Benchmarking on GPU...")
        results["PyTorch_GPU"] = benchmark_model(pt_path, device="cuda:0")
        results["ONNX_GPU"] = benchmark_model(onnx_path, device="cuda:0")
    else:
        print("CUDA is not available. Skipping GPU benchmarks.")
        results["PyTorch_GPU"] = {"status": "skipped", "reason": "CUDA not available"}
        results["ONNX_GPU"] = {"status": "skipped", "reason": "CUDA not available"}
        
    # Write results
    out_path = Path("/home/luke/ai_training/ultralytics/benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results written to {out_path}")

if __name__ == "__main__":
    main()
