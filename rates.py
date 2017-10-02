#! /usr/bin/env python
# -*- coding: iso-8859-1 -*-

from pylab import *

#rc('lines',  linestyle=None, marker='.', markersize=3)
rc('legend', fontsize=10)

txt_pa  = loadtxt("timings_3d_pa");
txt_fa  = loadtxt("timings_3d_fa");
txt_oc  = loadtxt("timings_3d_occa");

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
        data.append(txt[k, 2]/txt[k, column])
       #data.append(1e6*txt[k, column])
    pm1 = p-1
    ax.plot(dofs, data, line_style, label=label_prefix + 'Q' + str(p) + 'Q' + str(p-1),
            color=colors[i], linewidth=2)

  ax.grid(True, which='major')
  ax.grid(True, which='minor')
  ax.legend(loc='best', prop={'size':18})
  ax.set_autoscaley_on(False)
  ax.set_autoscalex_on(False)
  axis([10, 1e7, 1e4, 3e6])
 #axis([10, 1e7, 1e5, 1e9])
  ax.set_xlabel('H1 DOFs', fontsize=18)
  ax.set_xscale('log', basex = 10)
  ax.set_ylabel('[DOFs x time steps] / [seconds]', fontsize=18)
  ax.set_yscale('log', basex = 10)
  xticks(fontsize = 18, rotation = 0)
  yticks(fontsize = 18, rotation = 0)
  if title is not None:
    ax.set_title(title, fontsize=18)
  return fig

f1 = make_plot(8, 'PA: ', 'o-', txt_pa, title='Total Rate')
f2 = make_plot(8, 'FA: ', 'o-', txt_fa, title='Total Rate')
f3 = make_plot(8, 'OCCA: ', 'o-', txt_oc, title='Total Rate')
#f1.savefig('laghos_3D_TT_PA.png', dpi=300, bbox_inches='tight')
#f2.savefig('laghos_3D_TT_FA.png', dpi=300, bbox_inches='tight')
#f3.savefig('laghos_3D_TT_OC.png', dpi=300, bbox_inches='tight')

show()
