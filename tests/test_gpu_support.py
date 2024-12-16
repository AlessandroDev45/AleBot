import unittest
import torch
import torch.nn as nn
import time
import psutil
import os

class TestGPUSupport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize common test variables"""
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            # Set default device
            torch.cuda.set_device(0)
            # Clear cache
            torch.cuda.empty_cache()
    
    def test_gpu_info(self):
        """Test detailed GPU information"""
        print("\n=== GPU Information ===")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device name: {torch.cuda.get_device_name(0)}")
            print(f"GPU device count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            props = torch.cuda.get_device_properties(0)
            print(f"Device properties:")
            print(f"  - Total memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  - GPU compute capability: {props.major}.{props.minor}")
            print(f"  - Multi-processor count: {props.multi_processor_count}")
            print(f"  - Max memory allocation: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
            print(f"  - Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        self.assertTrue(torch.cuda.is_available(), "CUDA not available for PyTorch")

    def test_memory_management(self):
        """Test GPU memory management capabilities"""
        if torch.cuda.is_available():
            print("\n=== Memory Management Test ===")
            
            # Reset memory state
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Initial memory state
            initial_memory = torch.cuda.memory_allocated(0)
            print(f"Initial GPU memory allocated: {initial_memory / 1e6:.2f} MB")
            
            # Allocate large tensors
            tensors = []
            try:
                for i in range(5):
                    tensor = torch.randn(1000, 1000, device='cuda')
                    tensors.append(tensor)
                    current_memory = torch.cuda.memory_allocated(0)
                    print(f"After allocation {i+1}: {current_memory / 1e6:.2f} MB")
                
                peak_memory = torch.cuda.max_memory_allocated(0)
                print(f"Peak memory usage: {peak_memory / 1e6:.2f} MB")
                
                # Clear tensors
                del tensors
                torch.cuda.empty_cache()
                final_memory = torch.cuda.memory_allocated(0)
                print(f"Final GPU memory allocated: {final_memory / 1e6:.2f} MB")
                
                # Test if peak memory was significant
                self.assertTrue(peak_memory > initial_memory, 
                              "No significant memory was allocated")
                
                # Test if memory was freed (allowing for some overhead)
                self.assertTrue(final_memory < peak_memory, 
                              "Memory not properly freed")
            except Exception as e:
                self.fail(f"Memory management test failed: {str(e)}")

    def test_parallel_processing(self):
        """Test GPU parallel processing capabilities"""
        if torch.cuda.is_available():
            print("\n=== Parallel Processing Test ===")
            
            # Create large matrices for parallel operations
            size = 5000
            matrices = [torch.randn(size, size, device='cuda') for _ in range(3)]
            
            try:
                start_time = time.time()
                
                # Perform parallel matrix operations
                results = []
                for i in range(len(matrices)-1):
                    result = torch.mm(matrices[i], matrices[i+1])
                    results.append(result)
                
                torch.cuda.synchronize()
                duration = time.time() - start_time
                print(f"Parallel matrix operations time: {duration:.2f} seconds")
                
                # Test GPU utilization
                gpu_memory_used = torch.cuda.memory_allocated(0) / 1e9
                print(f"GPU memory used: {gpu_memory_used:.2f} GB")
                
                self.assertTrue(duration < 10, "Parallel processing too slow")
            except Exception as e:
                self.fail(f"Parallel processing test failed: {str(e)}")

    def test_deep_learning_performance(self):
        """Test deep learning model performance"""
        if torch.cuda.is_available():
            print("\n=== Deep Learning Performance Test ===")
            
            # Create a simple neural network
            class TestModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layers = nn.Sequential(
                        nn.Linear(100, 200),
                        nn.ReLU(),
                        nn.Linear(200, 100),
                        nn.ReLU(),
                        nn.Linear(100, 50)
                    )
                
                def forward(self, x):
                    return self.layers(x)
            
            try:
                # Initialize model and move to GPU
                model = TestModel().to(self.device)
                
                # Create test data
                batch_size = 1000
                input_data = torch.randn(batch_size, 100, device=self.device)
                
                # Warm-up run
                _ = model(input_data)
                torch.cuda.synchronize()
                
                # Timed run
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                for _ in range(100):  # 100 forward passes
                    output = model(input_data)
                end.record()
                
                torch.cuda.synchronize()
                duration = start.elapsed_time(end)
                
                print(f"Deep learning inference time (100 batches): {duration:.2f} ms")
                print(f"Average time per batch: {duration/100:.2f} ms")
                
                self.assertTrue(duration/100 < 10, 
                              "Deep learning inference too slow")
            except Exception as e:
                self.fail(f"Deep learning test failed: {str(e)}")

    def test_gpu_stress(self):
        """Stress test GPU capabilities"""
        if torch.cuda.is_available():
            print("\n=== GPU Stress Test ===")
            
            try:
                # Monitor initial state
                initial_memory = torch.cuda.memory_allocated(0)
                
                # Create increasingly large tensors
                sizes = [1000, 2000, 4000, 6000]
                results = []
                
                for size in sizes:
                    start_time = time.time()
                    
                    # Matrix multiplication with large matrices
                    a = torch.randn(size, size, device='cuda')
                    b = torch.randn(size, size, device='cuda')
                    c = torch.mm(a, b)
                    
                    torch.cuda.synchronize()
                    duration = time.time() - start_time
                    memory_used = torch.cuda.memory_allocated(0) / 1e9
                    
                    results.append({
                        'size': size,
                        'time': duration,
                        'memory': memory_used
                    })
                    
                    print(f"\nMatrix size: {size}x{size}")
                    print(f"Computation time: {duration:.2f} seconds")
                    print(f"GPU memory used: {memory_used:.2f} GB")
                    
                    # Clean up
                    del a, b, c
                    torch.cuda.empty_cache()
                
                # Verify memory cleanup
                final_memory = torch.cuda.memory_allocated(0)
                memory_diff = abs(final_memory - initial_memory) / 1e6
                print(f"\nMemory difference after cleanup: {memory_diff:.2f} MB")
                
                self.assertTrue(memory_diff < 1000, 
                              "Memory not properly cleaned up after stress test")
            except Exception as e:
                self.fail(f"Stress test failed: {str(e)}")

    def test_system_integration(self):
        """Test GPU integration with system resources"""
        if torch.cuda.is_available():
            print("\n=== System Integration Test ===")
            
            try:
                # Monitor system resources
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_info = psutil.virtual_memory()
                
                print(f"CPU Usage: {cpu_percent}%")
                print(f"System Memory: {memory_info.percent}% used")
                print(f"Available System Memory: {memory_info.available / 1e9:.2f} GB")
                
                # Test CPU-GPU data transfer
                size = 5000
                start_time = time.time()
                
                # Create tensor on CPU
                cpu_tensor = torch.randn(size, size)
                
                # Transfer to GPU
                gpu_tensor = cpu_tensor.cuda()
                
                # Perform computation
                result = torch.mm(gpu_tensor, gpu_tensor)
                
                # Transfer back to CPU
                cpu_result = result.cpu()
                
                torch.cuda.synchronize()
                duration = time.time() - start_time
                
                print(f"CPU-GPU transfer and compute time: {duration:.2f} seconds")
                
                # Clean up
                del cpu_tensor, gpu_tensor, result, cpu_result
                torch.cuda.empty_cache()
                
                self.assertTrue(duration < 10, 
                              "CPU-GPU data transfer too slow")
            except Exception as e:
                self.fail(f"System integration test failed: {str(e)}")

if __name__ == '__main__':
    unittest.main() 