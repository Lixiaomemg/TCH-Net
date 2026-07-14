# TCH-Net
 The Official implementation of TCH-Net project.




## Dataset
Including [Entity-13](https://www.image-net.org/download.php)、[Entity-30](https://www.image-net.org/download.php)、[Things-puls](https://things-initiative.org)、[SUN397](https://groups.csail.mit.edu/vision/SUN/hierarchy.html)、[MiniPlaces](https://www.kaggle.com/datasets/mittalshubham/images256)

Run the files in the folder ./Data_processing/ to obtain the hierarchical category topology of each dataset.

## Training

### Entity13
```
python deit/main_suppix_hier_0223V1.py \
   --model cast_small \
   --batch-size 256 \
   --epochs 150 \
   --num-superpixels 196 \
   --num_workers 12 \
   --data-set BREEDS-HIER-SUPERPIXEL \
   --breeds_sort entity13 \
   --data-path ./dataset/ImageNet \
   --output_dir ./output/entity13 \
   --lr 0.001 --warmup-lr 0.0001 \
   --globalkl --gk_weight 0.5
```
### Entity30
```
python deit/main_suppix_hier_0223V1.py \
   --model cast_small \
   --batch-size 256 \
   --epochs 150 \
   --num-superpixels 196 \
   --num_workers 12 \
   --data-set BREEDS-HIER-SUPERPIXEL \
   --breeds_sort entity30 \
   --data-path ./dataset/ImageNet \
   --output_dir ./output/entity30 \
   --lr 0.001 --warmup-lr 0.0001 \
   --globalkl --gk_weight 0.5
```
### Things-puls
```
python deit/main_suppix_hier_0223V1.py \
   --model cast_small \
   --batch-size 512 \
   --epochs 1000 \
   --num-superpixels 196 --num_workers 12 \
   --data-set things_plus  \
   --data-path ./dataset/things/  \
   --output_dir ./output/things_plus \
   --lr 0.001 --warmup-lr 0.0001 \
   --globalkl --gk_weight 0.5
```
### SUN397
```
python deit/main_suppix_hier_0223V1.py \
   --model cast_small \
   --batch-size 512 \
   --epochs 400 \
   --num-superpixels 196 \
   --num_workers 12 \
   --data-set sun397_2 \
   --data-path /data/xiaomeng/dataset/SUN397/ \
   --output_dir ./output/sun397_2 \
   --lr 0.001 --warmup-lr 0.0001 \
   --globalkl --gk_weight 1
```
### MiniPlaces
```
python deit/main_suppix_hier_0223V1.py \
   --model cast_small \
   --batch-size 512 \
   --epochs 150 \
   --num-superpixels 196 --num_workers 12 \
   --data-set miniplaces  \
   --data-path ./dataset/places365/datasets/benjaminkz/places365/versions/1/   \
   --output_dir ./output/miniplaces \
   --lr 0.001 --warmup-lr 0.0001 \
   --globalkl --gk_weight 1
```


## Baseline
Including [TransHP](https://github.com/WangWenhao0716/TransHP)、[Hier-ViT](https://github.com/pseulki/HCAST/tree/main#%EF%B8%8F--training)、[H-CAST](https://github.com/pseulki/HCAST/tree/main#%EF%B8%8F--training)


## Code Base
This repository is based on [H-CAST](https://github.com/pseulki/HCAST/tree/main#%EF%B8%8F--training).
