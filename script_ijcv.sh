CUDA_VISIBLE_DEVICES=0 python3 val.py --n_regions 1 --npoints 2048 4096 --model log/ijcv_shapenet_softpool_ns1/network.pth --dataset shapenet --methods softpool
mv pcds/softpool pcds/1
CUDA_VISIBLE_DEVICES=0 python3 val.py --n_regions 2 --npoints 2048 4096 --model log/ijcv_shapenet_softpool_ns2/network.pth --dataset shapenet --methods softpool
mv pcds/softpool pcds/2
CUDA_VISIBLE_DEVICES=0 python3 val.py --n_regions 4 --npoints 2048 4096 --model log/ijcv_shapenet_softpool_ns4/network.pth --dataset shapenet --methods softpool
mv pcds/softpool pcds/4
CUDA_VISIBLE_DEVICES=0 python3 val.py --n_regions 8 --npoints 2048 4096 --model log/ijcv_shapenet_softpool_ns8/network.pth --dataset shapenet --methods softpool
mv pcds/softpool pcds/8
CUDA_VISIBLE_DEVICES=0 python3 val.py --n_regions 16 --npoints 2048 4096 --model log/ijcv_shapenet_softpool_ns16/network.pth --dataset shapenet --methods softpool
mv pcds/softpool pcds/16
CUDA_VISIBLE_DEVICES=0 python3 val.py --n_regions 32 --npoints 2048 4096 --model log/ijcv_shapenet_softpool_ns32/network.pth --dataset shapenet --methods softpool
mv pcds/softpool pcds/32
CUDA_VISIBLE_DEVICES=0 python3 val.py --n_regions 64 --npoints 2048 4096 --model log/ijcv_shapenet_softpool_ns64/network.pth --dataset shapenet --methods softpool
mv pcds/softpool pcds/64
CUDA_VISIBLE_DEVICES=0 python3 val.py --n_regions 128 --npoints 2048 4096 --model log/ijcv_shapenet_softpool_ns128/network.pth --dataset shapenet --methods softpool
mv pcds/softpool pcds/128
CUDA_VISIBLE_DEVICES=0 python3 val.py --n_regions 256 --npoints 2048 4096 --model log/ijcv_shapenet_softpool_ns256/network.pth --dataset shapenet --methods softpool
mv pcds/softpool pcds/256
