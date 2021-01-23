#!/usr/bin/env python3

import h5py as h5
import numpy as np
import matplotlib.pyplot as ppl
from cycler import cycler
from scipy import signal
import mplcursors

sampt = 300

def normalize(x):
    return (x - x.mean())/x.std()

with h5.File("out.h5", "r") as f:
    data = f["dset"]
    chans = data.shape[1]
    fs = 1e9/(sampt*chans)
    dt = 1/fs
    filtf = 20e3/fs
    b, a = signal.butter(5, filtf)
    y = np.zeros((len(data), chans));
    for ch in range(chans):
        y[:, ch] = (normalize(signal.filtfilt(b, a, data[:,ch])))
    cm = ppl.get_cmap('gist_rainbow')
    fig = ppl.figure()
    ax = fig.add_subplot(111)
    custom_cycler = (cycler(color=[cm(1.*i/(chans/2)) for i in range(chans//2)]) *
                     cycler(linestyle=['-', '--']))

    ax.set_prop_cycle(custom_cycler)
    ppl.xlabel("Czas [s]")
    ppl.ylabel("Amplituda [V]")
    magconst = 4
    ppl.plot(np.arange(-4*magconst*sampt/1e9, dt*(len(data)), dt)[0:len(y)], y[:, 0])
    if chans > 1:
        ppl.plot(np.arange(16*magconst*sampt/1e9, dt*(len(data)+10), dt)[0:len(y)], y[:, 1])
    for ch in range(2, chans):
        t = np.arange((chans-ch)*magconst*sampt/1e9, dt*(len(data)+10), dt);
        ppl.plot(t[0:len(y)], y[:, ch])
    ppl.legend(["{}".format(chan) for chan in range(chans)], loc="lower center", ncol=chans//2);
    mplcursors.cursor()
    ppl.show()

