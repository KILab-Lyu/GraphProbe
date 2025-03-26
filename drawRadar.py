import numpy as np

import pygal
from pygal.style import LightenStyle,LightColorizedStyle
mat = [
    [(6+4+6+6)/4,(6+6+6)/3,(2+3+1)/3, (7+7+8+8)/4,(5+6+7)/3,(1+2+1)/3],
    [(4+6+4+4)/4,(2+3+2)/3,(7+6+7)/3, (3+3+3+3)/4,(1+1+1)/3,(8+6+7)/3],
    [(5+5+3+5)/4,(4+4+3)/3,(5+2+2)/3, (5+5+7+7)/4,(2+3+3)/3,(4+4+3)/3],
    [(7+8+7+7)/4,(5+5+5)/3,(6+7+6)/3, (6+4+5+5)/4,(4+4+4)/3,(7+8+8)/3],
    [(3+1+5+3)/4,(8+8+7)/3,(1+1+3)/3, (4+6+4+4)/4,(7+7+5)/3,(2+1+2)/3],
    [(1+3+2+2)/4,(3+2+4)/3,(4+4+4)/3, (2+2+2+2)/4,(8+8+8)/3,(5+5+4)/3],
    [(2+2+1+1)/4,(1+1+1)/3,(8+8+8)/3, (1+1+1+1)/4,(3+2+2)/3,(6+7+6)/3],
    [(8+7+8+8)/4,(7+7+8)/3,(3+5+5)/3, (8+8+6+6)/4,(6+5+6)/3,(3+3+5)/3],
]

np_mat = np.array(mat)
np_mat = (9-np_mat)/9
style = LightenStyle("#A52A2A",base_style=LightColorizedStyle)
style.background = '#FFF'
# style.foreground = '#FFF'
style.opacity_hover= '.2'

config = pygal.Config()
config.show_legend = False
config.width = 240
config.height = 240
config.fill = True
config.range = (0,1)
config.show_y_guides = True
config.show_x_guides = False
config.margin=0
config.margin_top=5
config.margin_left=-23
config.margin_bottom=-20
# config.inverse_y_axis = True
# config.x_label_rotation = 0
# config.title_font_size = 30
# config.label_font_size = 10

radar_chart = pygal.Radar(config, style=style)
# radar_chart.title = 'Employee performance before and after the event' 
# radar_chart.x_labels = ['Degree','Closeness','Betweenness','Eigenvector','Distance', 'Structure', 'Cluster', 'Construstive']
radar_chart.x_labels = ['Node Classification', 'Link Prediction', 'Contrastive-Graph Classification', 'Contrastive-Node Classification', 'Contrastive-Link Prediction', 'Graph Classification']

# radar_chart.add('Chebyshev', np_mat[0])
# radar_chart.render_to_file('radar_Chebyshev.svg', dpi=400)
# radar_chart.add('SSGCN',     np_mat[1])
# radar_chart.render_to_file('radar_SSGCN.svg', dpi=400)
# radar_chart.add('GCN',       np_mat[2])
# radar_chart.render_to_file('radar_GCN.svg', dpi=400)
# radar_chart.add('LightGCN',  np_mat[3])
# radar_chart.render_to_file('radar_LightGCN.svg', dpi=400)
# radar_chart.add('GraphSAGE', np_mat[4])
# radar_chart.render_to_file('radar_GraphSAGE.svg', dpi=400)
# radar_chart.add('GAT',       np_mat[5])
# radar_chart.render_to_file('radar_GAT.svg', dpi=400)
# radar_chart.add('GIN',       np_mat[6])
# radar_chart.render_to_file('radar_GIN.svg', dpi=400)
radar_chart.add('MLP',       np_mat[7])
radar_chart.render_to_file('radar_MLP.svg', dpi=400)

for i in range(8):
    print(np_mat[i])
# radar_chart.render_to_png('radar_Chebyshev.png', dpi=4000)
radar_chart.render_to_file('radar_Chebyshev.svg', dpi=400)