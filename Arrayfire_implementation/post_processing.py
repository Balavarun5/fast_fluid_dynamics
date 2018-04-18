#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from matplotlib import pyplot as pl
from tqdm import trange
import h5py
import numpy as np
import arrayfire as af
af.set_backend('opencl')
af.set_device(0)

import parameters as params

pl.rcParams['figure.figsize'  ] = 9.6, 6.
pl.rcParams['figure.dpi'      ] = 100
pl.rcParams['image.cmap'      ] = 'jet'
pl.rcParams['lines.linewidth' ] = 1.5
pl.rcParams['font.family'     ] = 'serif'
pl.rcParams['font.size'       ] = 20
pl.rcParams['font.sans-serif' ] = 'serif'
pl.rcParams['text.usetex'     ] = False
pl.rcParams['axes.linewidth'  ] = 1.5
pl.rcParams['axes.titlesize'  ] = 'medium'
pl.rcParams['axes.labelsize'  ] = 'medium'
pl.rcParams['xtick.major.size'] = 8
pl.rcParams['xtick.minor.size'] = 4
pl.rcParams['xtick.major.pad' ] = 8
pl.rcParams['xtick.minor.pad' ] = 8
pl.rcParams['xtick.color'     ] = 'k'
pl.rcParams['xtick.labelsize' ] = 'medium'
pl.rcParams['xtick.direction' ] = 'in'
pl.rcParams['ytick.major.size'] = 8
pl.rcParams['ytick.minor.size'] = 4
pl.rcParams['ytick.major.pad' ] = 8
pl.rcParams['ytick.minor.pad' ] = 8
pl.rcParams['ytick.color'     ] = 'k'
pl.rcParams['ytick.labelsize' ] = 'medium'
pl.rcParams['ytick.direction' ] = 'in'


def contour_plot(u, v, t_n):
    '''
    '''
    u_map = af.moddims(u, params.n, params.n)
    v_map = af.moddims(v, params.n, params.n)

    scale  = af.np_to_af_array(np.linspace(0, 1, params.n))
    x_tile = af.tile(scale, 1, params.n)
    y_tile = af.transpose(x_tile)

    speed_tile = (af.transpose(u_map) ** 2 + af.transpose(v_map) ** 2) ** 0.5
    x_tile     = np.array(x_tile)
    y_tile     = np.array(y_tile)
    speed_tile = np.array(af.flip(speed_tile, 0))

    pl.contourf(y_tile, x_tile, speed_tile, cmap='jet')
    pl.gca().set_aspect('equal')
    pl.title('Time = %.2f' %(t_n * params.delta_t))
    pl.xlabel('x')
    pl.ylabel('y')
    pl.colorbar()
    pl.savefig('results/images/%04d' %(t_n) + '.png')
    pl.close('all')


for i in trange(1000):
    with h5py.File('results/h5py/dump_timestep_%06d' %int(100 * i) + '.hdf5', 'r') as hf:
        u = hf['u'][:]
        v = hf['v'][:]
        u = af.np_to_af_array(u)
        v = af.np_to_af_array(v)
        contour_plot(u, v, 100 * i)
