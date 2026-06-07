import torch
import sys

print("🔍 GPU Detection Check")
print("=" * 40)

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"🚀 CUDA Available: {cuda_available}")

if cuda_available:
    device_count = torch.cuda.device_count()
    print(f"📊 CUDA Devices: {device_count}")
    
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        device_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"   Device {i}: {device_name} ({device_memory:.1f} GB)")
else:
    print("⚠️  No CUDA GPU detected")
    print("💡 Training will use CPU (slower)")

# Check MPS (Apple Silicon)
try:
    mps_available = torch.backends.mps.is_available()
    print(f"🍎 MPS Available: {mps_available}")
    if mps_available:
        print("   Apple Silicon GPU detected")
except:
    print("🍎 MPS: Not available")

print("\n🎯 Training Recommendation:")
if cuda_available:
    print("✅ Use CUDA for GPU training")
elif mps_available:
    print("✅ Use MPS for Apple Silicon training")
else:
    print("⚠️  CPU training only (will be slow)")
