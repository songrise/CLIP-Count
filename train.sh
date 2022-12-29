
# nohup python -u finetune_clip.py --epochs 70 --weight_decay 0.05 --lr 1e-4 --batch_size 32 --use_vpt --exp_name vitB32bvpt
nohup python -u finetune_clip.py --epochs 70 --weight_decay 0.05 --lr 1e-4 --batch_size 32 --use_coop --exp_name vitB32coop
nohup python -u finetune_clip.py --epochs 70 --weight_decay 0.05 --lr 1e-4 --batch_size 32 --use_coop --use_vpt --exp_name vitB32all

nohup python -u finetune_clip.py --epochs 200 --weight_decay 0.05 --lr 1e-4 --batch_size 16 --use_coop --use_vpt --exp_name vitB16decode