import sys
try:
    from mmcv import Config
except ImportError:
    try:
        from mmengine.config import Config
    except ImportError:
        print("Neither mmcv nor mmengine found.")
        sys.exit(1)

def verify_config(config_path):
    try:
        print(f"Verifying {config_path}...")
        cfg = Config.fromfile(config_path)
        print("Config parsed successfully.")
        # Check if vital keys exist
        assert 'model' in cfg
        assert 'train_dataloader' in cfg
        print("Basic keys present.")
    except Exception as e:
        print(f"FAILED to parse {config_path}")
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_config.py <config_path>")
        sys.exit(1)
    verify_config(sys.argv[1])
