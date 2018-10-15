#! /usr/bin/env python
# -*- coding: iso-8859-1 -*-

from pylab import *

#rc('lines',  linestyle=None, marker='.', markersize=3)
rc('legend', fontsize=10)

txt_master  = loadtxt("timings_tux_2d_master");
txt_kernels  = loadtxt("timings_tux_2d_kernels");
txt_raja  = loadtxt("timings_tux_2d_raja");
txt_raja_cuda  = loadtxt("timings_tux_2d_raja-cuda");
txt_occa  = loadtxt("timings_tux_2d_occa");
txt_occa_cuda  = loadtxt("timings_tux_2d_occa-cuda");

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
    #print("i="+repr(i)+", p="+repr(p))
    dofs = []
    data = []
    for k in range(txt.shape[0]):
      #print("   k="+repr(k))
      o = txt[k,0]
      #print("   o="+repr(o))
      if o == p:
        dofs.append(txt[k, 2])
        #data.append(txt[k, 2]/txt[k, column])
        data.append(txt[k, column])
    ax.plot(dofs, data, line_style, 
            label=label_prefix + 'Q' + str(p+1) + 'Q' + str(p),
            color=colors[i], linewidth=2)

  ax.grid(True, which='major')
  ax.grid(True, which='minor')
  ax.legend(loc='best', prop={'size':18})
  ax.set_autoscaley_on(False)
  ax.set_autoscalex_on(False)
  axis([1e2, 1e6, 1e-3, 1e1])
  ax.set_xlabel('H1 DOFs', fontsize=18)
  ax.set_xscale('log', basex = 10)
  ax.set_ylabel('[megadofs x time steps] / [seconds]', fontsize=18)
  ax.set_yscale('log', basex = 10)
  xticks(fontsize = 18, rotation = 0)
  yticks(fontsize = 18, rotation = 0)
  if title is not None:
    ax.set_title(title, fontsize=18)
  return fig

master = make_plot(9, 'master:', 'o-', txt_master, title='Master-Serial Total Rate')
kernels = make_plot(9, 'kernels:', 'o-', txt_kernels, title='Kernels-Serial Total Rate')
raja = make_plot(9, 'raja:', 'o-', txt_raja, title='Raja-Serial Total Rate')
raja_cuda = make_plot(9, 'raja_cuda:', 'o-', txt_raja_cuda, title='Raja-Cuda Total Rate')
occa = make_plot(9, 'occa:', 'o-', txt_occa, title='Occa-Serial Total Rate')
occa_cuda = make_plot(9, 'occa_cuda:', 'o-', txt_occa_cuda, title='Occa-Cuda Total Rate')

master.savefig('laghos_master_2D.png', dpi=300, bbox_inches='tight')
kernels.savefig('laghos_kernels_2D.png', dpi=300, bbox_inches='tight')
raja.savefig('laghos_raja_2D.png', dpi=300, bbox_inches='tight')
raja_cuda.savefig('laghos_raja_cuda_2D.png', dpi=300, bbox_inches='tight')
occa.savefig('laghos_occa_2D.png', dpi=300, bbox_inches='tight')
occa_cuda.savefig('laghos_occa_cuda_2D.png', dpi=300, bbox_inches='tight')

show()
