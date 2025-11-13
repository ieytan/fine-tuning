# Remove virtual environment that may have been copied from the local machine
rm -rf .venv

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3."
    exit 1
fi

# Create a new virtual envrionment, activate it, and install requirements
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_cuda.txt

# Verify torch was installed for cuda devices
output=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>&1)
status=$?

if [ $status -ne 0 ]; then
    echo "❌ PyTorch is not installed or failed to import:"
    echo "$output"
    exit 1
fi

if [ "$output" = "True" ]; then
    echo "✅ PyTorch is installed and CUDA is available."
else
    echo "⚠️ PyTorch is installed, but CUDA is NOT available."
fi