conda create --name nerfies python=3.8

source deactivate
conda activate nerfies
which python

pip install -r requirements.txt

# nvidia-smi shows 11.2
conda install -c conda-forge cudatoolkit=11.2.2
conda install -c conda-forge cudnn=8.1.0.77
pip install --upgrade pip

pip uninstall jax
pip install "jax[cuda11_cudnn805]==v0.3.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install tqdm

# see requirements-full.txt for reference
# pip freeze > requirements-full.txt


python train.py --data_dir datasets/iphone-home --base_folder experiments/001 --gin_configs configs/gpu_quarterhd.gin


python eval.py \
        --data_dir datasets/iphone-home \
        --base_folder experiments/001 \
        --gin_configs configs/gpu_quarterhd.gin