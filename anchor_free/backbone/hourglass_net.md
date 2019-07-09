# class

## convolution

conv-bn-relu

## fully_connected

linear-bn-relu

## residual

conv-bn-relu-conv-bn + skip(x) + relu

## MergeUp

up1 + up2



# function

## make_layer

(input_dim, output_dim) {(output_dim,output_dim), ...}

## make_layer_revr

{(input_dim, input_dim) ,…} (input_dim,output_dim)

## make_merge_layer

MergeUp()

## make_pool_layer

nn.Sequential()

## make_unpool_layer

nn.Upsample(scale_factor=2)

## make_kp_layer

(256, cur_dim) ( cur_dim, output_dim)

## make_inter_layer

residual(3, dim, dim)

## make_cnv_layer

convolution(3, input_dim, output_dim)

## make_hg_layer

(input_dim, output_dim, stride=2) ->这里进行一次下采样

{(out_dim, out_dim), ...}

# 构造特殊的层

##  keep_module

up1 =>{（cur_dim, cur_dim) ..} => cur_dim

max1 => {} => x => cur_dim

low1 => (cur_dim, next_dim), {( next_dim, next_dim ), …} => next_dim 下采样

low2 => kp_module_next => (next_dim, … , next_dim) => next_dim

low3 => {(next_dim, next_dim), … }, (next_dim, cur_dim) => cur_dim

up2 => （cur_dim), 上采样

merger => up1 + up2



## install

```shell
cd $COCOAPI/PythonAPI
make
python setup.py install --user
```

```shell
pip install -r requirements.txt
```

```shell
cd $CenterNet_ROOT/src/lib/models/networks/DCNv2
./make.sh
```

```shell
cd $CenterNet_ROOT/src/lib/external
make
```

```shell
sed -i "1254s/torch\.backends\.cudnn\.enabled/False/g"  ~/anaconda3/envs/centernet/lib/python3.6/site-packages/torch/nn/functional.py
```

```python
python main.py --arch=hourglass	
```

# centetnet_key install

```shell
cd /train/results/CenterNet/models/py_utils/_cpools
python setup.py install --user

cd /train/trainset/1/coco/PythonAPI
make

cd /train/results/CenterNet/external
make
```

```shell
ln -snf 【新目标目录】 【软链接地址】
touch .tmux.conf
```

```
set -g prefix C-a
unbind C-b
bind r source-file ~/.tmux.conf
```

```python
conda install jupyter notebool
conda activate centernet
```

