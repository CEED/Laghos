#! /usr/bin/env python
# -*- coding: iso-8859-1 -*-

from pylab import *

#rc('lines',  linestyle=None, marker='.', markersize=3)
rc('legend', fontsize=10)

txt_pa  = loadtxt("timings_pa");
txt_fa  = loadtxt("timings_fa");

def make_plot(column, label_prefix, line_style, txt, title=None, fig=None):
  cm=get_cmap('Set1') # 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3'
  if '_segmentdata' in cm.__dict__:
    cm_size=len(cm.__dict__['_segmentdata']['red'])
  elif 'colors' in cm.__dict__:
    cm_size=len(cm.__dict__['colors'])
  colors=[cm(1.*i/(cm_size-1)) for i in range(cm_size)]

  if fig is None:
    fig = figure(figsize=(10,8))
  ax = fig.gca()
  orders = list(set([int(x) for x in txt[:,0]]))

  for i, p in enumerate(orders):
    dofs = []
    data = []
    for k in range(txt.shape[0]):
      o = txt[k,0]
      if o == p:
        dofs.append(txt[k, 2])
        data.append(1e3*txt[k, column])
    ax.plot(dofs, data, line_style, label=label_prefix + str(p),
            color=colors[i], linewidth=2)

  ax.grid(True, which='major')
  ax.grid(True, which='minor')
  ax.legend(loc='best', prop={'size':18})
  #axis([-0.005, 0.2, -0.15, 2.2])
  ax.set_xlabel('Points per node', fontsize=15)
  ax.set_xscale('log', basex = 10)
  ax.set_ylabel('[DOFs x iterations] / [nodes x seconds]', fontsize=15)
  ax.set_yscale('log', basex = 10)
  if title is not None:
    ax.set_title(title)
  return fig

f1 = make_plot(4, 'PA: p = ', 'o-', txt_pa, title='CG Execution Rate')
#f1.savefig('CG_rate.pdf', bbox_inches='tight')
f2 = make_plot(6, 'PA: p = ', 'o-', txt_pa, title='Force Execution Rate')
#f2.savefig('Force_rate.pdf', bbox_inches='tight')
f3 = make_plot(4, 'FA: p = ', 'o-', txt_fa, title='CG Execution Rate')
f4 = make_plot(6, 'FA: p = ', 'o-', txt_fa, title='Force Execution Rate')

show()
