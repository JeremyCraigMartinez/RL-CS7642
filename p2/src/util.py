'''utility functions'''

from sys import argv
import matplotlib.pyplot as plt
import numpy as np

def logger(_str, level='low'):
    '''logger enabled with --verbose flag'''
    if '--verbose' in argv or level == 'high':
        print(_str)

def plot_values(values, xlabel, ylabel, filename):
    '''Plot figure'''
    plt.plot(np.arange(len(values)), values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)

def plot_multiple(values, labels, xlabel, ylabel, filename):
    '''Plot figure'''
    for value, label in zip(values, labels):
        plt.plot(np.arange(len(value)), value, label=label)
    plt.legend(labels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()

def plot_two_separate(values, xlabel, ylabels, filename):
    '''Plot figure'''
    fig, host = plt.subplots()
    fig.subplots_adjust(right=0.75)

    par1 = host.twinx()
    p1, = host.plot(range(len(values[0])), values[0], "b-", label=ylabels[0])
    p2, = par1.plot(range(len(values[1])), values[1], "r-", label=ylabels[1])

    host.set_ylim(min(values[0]), max(values[0]))
    par1.set_ylim(0, max(values[1]))

    host.set_xlabel(xlabel)
    host.set_ylabel(ylabels[0])
    par1.set_ylabel(ylabels[1])

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)

    lines = [p1, p2]

    host.legend(lines, [l.get_label() for l in lines])
    fig.savefig(filename)
    plt.close()
