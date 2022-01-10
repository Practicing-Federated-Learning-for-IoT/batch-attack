## 更改实验设置
在 utils/options.py中更改实验的具体参数，例如 batch size, lr, atk(oscillating in; oscillating out; lowhigh; highlow), attack_type (reorder; reshuffle)
![](https://codimd.xixiaoyao.cn/uploads/upload_37e0de727a674c963aee7c7509b23444.png)
在 main_fed.py 中可以通过更改下方控制参与攻击的客户端个数、和攻击的轮数。num代表选择第几个客户端，iter 代表 epoch 的轮数
![](https://codimd.xixiaoyao.cn/uploads/upload_ffc0d6df2f2ab61812bcfb173dbb659d.png)
## 怎样运行
```python=
python main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 100 --gpu 0
```