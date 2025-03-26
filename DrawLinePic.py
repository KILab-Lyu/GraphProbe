import matplotlib.pyplot as plt

x = [2, 3, 4, 5]

Yelp_rand = [0.001273891,
             0.2485,
             0.123189,
             0.221438901,
             ]
Yelp_meta=[
    0.204238901,
    0.2364,
    0.18123785,
    0.007298341,
]
Movie_rand = [0.712379818,
              0.8545,
              0.612379813,
              0.517384179,
              ]
Movie_meta = [0.4909,
              0.44131298,
              0.401238902,
              0.183478933,
              ]
Cora_rand = [0.006060606,
             0.36969697,
             0.612121212,
             0.381818182
             ]
Cora_meta = [0.6121,
             0.601348091,
             0.591387901,
             0.54489019,
             ]
Flickr_rand = [0.007891398,
               0.031378012,
               0.0545,
               0.023812904
               ]
Flickr_meta = [0.401238901,
               0.423189041,
               0.4424,
               0.007382194,
               ]
MUTAG_rand = [0.6016,
              0.598312,
              0.541789302,
              0.51831902,
              ]
MUTAG_meta = [0.441839312,
              0.534781289,
              0.5632,
              0.293824824,
              ]
ENZYMES_rand = [0.712378912,
                0.745894728,
                0.771237891,
                0.8227
                ]
ENZYMES_meta = [0.029184901,
                0.033128901,
                0.074,
                0.058139481,
                ]

# 创建一个图形对象
fig = plt.figure()

# 在图形对象中添加一个子图，设置标题和坐标轴标签
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Path hyper-parameter experiments')
ax.set_xlabel('Path parameter')
ax.set_ylabel('Correlation score with the AUC/ACC metrics.')

# 绘制两条折线，分别表示两个城市的气温，设置颜色，线宽，标签和样式
ax.plot(x, Yelp_rand, color='red', linewidth=2, label='Yelp', linestyle='--')
ax.plot(x, Cora_rand, color='blue', linewidth=2, label='Cora', linestyle='-.')
ax.plot(x, Flickr_rand, color='orange', linewidth=2, label='Flickr', linestyle='-.')
ax.plot(x, Movie_rand, color='yellow', linewidth=2, label='Movie', linestyle='-.')
ax.plot(x, MUTAG_rand, color='green', linewidth=2, label='MUTAG', linestyle='-.')
ax.plot(x, ENZYMES_rand, color='purple', linewidth=2, label='ENZYMES', linestyle='-.')

# 添加图例，设置位置和字体大小
ax.legend(loc='upper left', fontsize=10)

# 调整子图的边距，使图形更紧凑
fig.tight_layout()

# 显示图形
plt.show()
