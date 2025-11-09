usage
pip install insightface onnxruntime-gpu

pip install torch torchvision scikit-learn pillow opencv-python imagehash tqdm


python pick_top3_per_set_insightface_gpu.py photos.zip --ref reference.jpg --gpu --export winners_out --csv winners.csv
