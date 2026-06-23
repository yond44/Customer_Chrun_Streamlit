import os
import json

print("=" * 60)
print("📁 CHECKING MODELS FOLDER")
print("=" * 60)

models_dir = 'models'
required_files = [
    'ann_model.h5',
    'ann_model.keras',
    'final_pipeline.pkl',
    'model_config.json',
    'Customer_churn_rf_model.pkl',
    'encoder_ordinal.pkl',
    'encoder_ohe.pkl',
    'scaler.pkl',
    'model_metadata.json'
]

if os.path.exists(models_dir):
    print(f"✅ Folder '{models_dir}' ada\n")
    
    for file in os.listdir(models_dir):
        file_path = os.path.join(models_dir, file)
        file_size = os.path.getsize(file_path)
        status = "✅" if file in required_files or file.startswith('.') else "❓"
        print(f"{status} {file:40s} ({file_size:,} bytes)")
else:
    print(f"❌ Folder '{models_dir}' TIDAK ADA!")

print("\n" + "=" * 60)
print("📋 CHECKING JSON FILES")
print("=" * 60)

# Check model_config.json
config_path = 'models/model_config.json'
if os.path.exists(config_path):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✅ model_config.json valid")
        print(f"   Keys: {list(config.keys())}")
    except Exception as e:
        print(f"❌ model_config.json ERROR: {e}")
else:
    print(f"❌ model_config.json NOT FOUND")

# Check model_metadata.json
meta_path = 'models/model_metadata.json'
if os.path.exists(meta_path):
    try:
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        print(f"✅ model_metadata.json valid")
        print(f"   Keys: {list(metadata.keys())}")
        print(f"   test_accuracy: {metadata.get('test_accuracy')}")
    except Exception as e:
        print(f"❌ model_metadata.json ERROR: {e}")
else:
    print(f"❌ model_metadata.json NOT FOUND")

print("\n" + "=" * 60)