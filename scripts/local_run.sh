config=sample
save_path=snapshot/${config}

python train.py --config configs/${config}.yaml \
                --save_path ${save_path} \