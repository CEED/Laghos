#! /usr/bin/env python
# -*- coding: iso-8859-1 -*-

from pylab import *

#rc('lines',  linestyle=None, marker='.', markersize=3)
rc('legend', numpoints=6, fontsize=10)


rE = loadtxt("rho_exact.out");
r  = loadtxt("rho.out");

###############
figure(1)
plot(rE[:,0], rE[:,1], 'g', label='Exact Density', linewidth = 1)
scatter(r[:,0], r[:,1], s = 10, c = 'r', label = 'Density', edgecolors = 'none')


grid('on')
legend(loc='best', prop = {'size':20})
axis([-0.1, 1.1, -0.5, 7.0])
xticks(fontsize = 20, rotation = 0)
yticks(fontsize = 20, rotation = 0)
xlabel('Position', fontsize = 30)
ylabel('Density', fontsize = 30)
savefig('Sod_r.png', dpi=300, format='png', bbox_inches='tight')

show()
