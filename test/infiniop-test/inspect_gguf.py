import sys
from pathlib import Path

# Add the project's root directory to the Python path
# This allows us to import modules from other test directories
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Now the import should work
from gguf import GGUFReader

def inspect_file(filepath: Path):
    """Reads and prints the contents of a GGUF file."""
    if not filepath.is_file():
        print(f"Error: File not found at '{filepath}'")
        return

    print(f"--- Inspecting file: {filepath} ---")
    try:
        reader = GGUFReader(str(filepath), 'r')

        print(f"Found {len(reader.tensors)} tensors:")
        for i, tensor in enumerate(reader.tensors):
            print(f"  Tensor #{i}:")
            print(f"    Name: {tensor.name}")
            print(f"    Shape: {tensor.shape}")
            print(f"    Type: {tensor.tensor_type.name}")

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_gguf.py <path_to_your.gguf_file>")
        sys.exit(1)

    # Assume the gguf file is in a subdirectory relative to the script
    script_dir = Path(__file__).parent
    gguf_file = script_dir / sys.argv[1]
    inspect_file(gguf_file)