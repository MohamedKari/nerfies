# nvidia-smi shows 11.2

conda install -c conda-forge cudatoolkit=11.2.2
conda install -c conda-forge cudnn=8.1.0.77
pip install --upgrade pip

pip install "jax[cuda11_cudnn805]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

python train.py --data_dir datasets/iphone-home --base_folder experiments/001 --gin_configs configs/test_vrig.gin