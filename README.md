# TCH-Net
 The Official implementation of TCH-Net project.




## Dataset



## Training

```
python deit/main_suppix_hier_0223V2_hcast.py \
  --model cast_small \
  --batch-size 512 \
  --epochs 1000 \
  --num-superpixels 196 --num_workers 12 \
  --data-set things_plus  \
  --data-path /data/xiaomeng/dataset/things/  \
  --output_dir ./output/0523_things_plus-hcast-0.5 \
  --lr 0.001 --warmup-lr 0.0001 \
  --globalkl --gk_weight 0.5
```

## Baseline
Including [TransHP](https://github.com/WangWenhao0716/TransHP)、[Hier-ViT](https://github.com/pseulki/HCAST/tree/main#%EF%B8%8F--training)、[H-CAST](https://github.com/pseulki/HCAST/tree/main#%EF%B8%8F--training)
