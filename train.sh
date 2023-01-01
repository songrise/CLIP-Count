
# nohup python -u finetune_clip.py --epochs 70 --weight_decay 0.05 --lr 1e-4 --batch_size 32 --use_vpt --exp_name vitB32bvpt
# nohup python -u finetune_clip.py --epochs 70 --weight_decay 0.05 --lr 1e-4 --batch_size 32 --use_coop --exp_name vitB32coop
# nohup python -u finetune_clip.py --epochs 70 --weight_decay 0.05 --lr 1e-4 --batch_size 32 --use_coop --use_vpt --exp_name vitB32all

# nohup python -u finetune_clip.py --epochs 200 --weight_decay 0.05 --lr 1e-4 --batch_size 16 --use_coop --use_vpt --exp_name vitB16decode
# nohup python -u finetune_clip.py --epochs 300 --weight_decay 0.05 --lr 1e-4 --batch_size 16 --use_coop --use_vpt --exp_name vitB16cross
# nohup python -u finetune_clip.py --epochs 300 --weight_decay 0.05 --lr 1e-4 --batch_size 16 --use_coop --use_vpt --exp_name vitB16rep

# nohup python -u finetune_clip.py --epochs 200 --weight_decay 0.05 --lr 1e-4 --batch_size 16 --exp_name vitB16val2
# nohup python -u finetune_clip.py --epochs 200 --weight_decay 0.05 --lr 1e-4 --batch_size 16 --use_coop  --exp_name vitB16val3

#best
# nohup python -u finetune_clip.py --epochs 200 --weight_decay 0.05 --lr 1e-4 --batch_size 16 --use_coop  --exp_name vitB16enc
nohup python -u finetune_clip.py --epochs 200 --weight_decay 0.05 --lr 1e-4 --batch_size 16 --use_coop  --exp_name vitB16enc_nocoop
nohup python -u finetune_clip.py --epochs 200 --weight_decay 0.05 --lr 1e-4 --batch_size 16 --use_coop --use_vpt  --exp_name vitB16enc_full




# nohup python -u finetune_clip_reg.py --epochs 200 --weight_decay 0.05 --lr 3e-6 --batch_size 16 --use_coop  --exp_name vitB16reg

# #test
# python -u finetune_clip.py --epochs 200 --weight_decay 0.05 --lr 1e-4 --batch_size 16  --exp_name vitB16enc --mode app
