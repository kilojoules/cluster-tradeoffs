#!/bin/bash
# LUMI setup for cluster-tradeoffs
# Run this once on a login node (uan01)

# 1. Add pixi to PATH (also add to ~/.bashrc for persistence)
export PATH="$HOME/.pixi/bin:$PATH"
if ! grep -q '.pixi/bin' ~/.bashrc; then
    echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.bashrc
    echo "Added pixi to ~/.bashrc"
fi

# 2. Clone repo (if not already done)
cd "$HOME"
if [ ! -d cluster-tradeoffs ]; then
    git clone git@github.com:kilojoules/cluster-tradeoffs.git
    cd cluster-tradeoffs
    git checkout feature/gpu-multistart-bilevel
else
    cd cluster-tradeoffs
    git fetch && git checkout feature/gpu-multistart-bilevel && git pull
fi

# 3. Install ROCm environment
pixi install -e rocm

# 4. Quick smoke test (on login node, CPU only — just checks imports)
pixi run -e rocm python -c "
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
