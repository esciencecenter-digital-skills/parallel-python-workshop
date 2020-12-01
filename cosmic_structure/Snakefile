import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from cosmic_structure import (Box, random_density, compute_potential, run_model, plot_structure)

def power_spectrum(k):
    return np.exp(-k**2 / 100) / k**0.5

box = Box(N=256, L=50.0)

input_times = np.linspace(0.0, 4.0, num=10)
timestamps = [str(round(t * 1000)) for t in input_times]

rule all:
    input:
        "data/plot_all.gif"

rule initial_condition:
    output:
        "data/delta0.npy"
    run:
        delta = random_density(box, power_spectrum)
        np.save(output[0], delta)

rule compute_potential:
    input:
        "data/delta0.npy"
    output:
        "data/pot0.npy"
    run:
        delta = np.load(input[0])
        pot = compute_potential(box, delta)
        np.save(output[0], pot)

rule plot_result:
    input:
        "data/pot0.npy"
    output:
        "data/plot_{timestamp}.png"
    run:
        t = int(wildcards.timestamp) / 1000
        pot = np.load(input[0])
        structure = run_model(box, pot, t)
        fig, ax = plt.subplots(figsize=(12, 12))
        plot_structure(
            box, structure, xlim=[-20,20], ylim=[-20,20],
            ax=ax, point_scale=1, line_scale=2)
        fig.savefig(output[0])

rule animated_gif:
    input:
        expand("data/plot_{timestamp}.png", timestamp=timestamps)
    output:
        "data/plot_all.gif"
    shell:
        "convert -delay 50 -loop 0 {input} {output}"
