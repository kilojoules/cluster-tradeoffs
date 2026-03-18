#!/bin/bash
# LUMI setup for cluster-tradeoffs
# Run from inside the cloned repo directory.

set -e

# 1. Add pixi to PATH (also add to ~/.bashrc for persistence)
export PATH="$HOME/.pixi/bin:$PATH"
if ! grep -q '.pixi/bin' ~/.bashrc; then
    echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.bashrc
    echo "Added pixi to ~/.bashrc"
fi

# 2. Install ROCm environment
# Use pyproject.toml (not pixi.toml) — it has the rocm environment
pixi install -e rocm --manifest-path pyproject.toml

# 3. Quick smoke test (on login node, CPU only — just checks imports)
pixi run -e rocm --manifest-path pyproject.toml python -c "
import jax
jax.config.update('jax_enable_x64', True)
import pixwake
print('pixwake imported OK')
print('JAX version:', jax.__version__)
print('JAX devices:', jax.devices())
"

echo ""
echo "Setup complete. Submit a job with:"
echo "  sbatch lumi/run_greedy_gpu.sbatch"
