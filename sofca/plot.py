# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 17:33:58 2026

@author: s.gupta
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_distribution(
    df, 
    x_column='equivalent_diameter', 
    weight_column=None,  # optional, e.g., 'Volume'
    kind='hist',         # 'hist' or 'cumulative'
    metric='number',     # 'number' or 'volume'
    bins=20
):
    """
    General-purpose distribution plot.
    
    Parameters:
    - df: pandas DataFrame containing data
    - x_column: column name to plot on x-axis (size, shape, etc.)
    - weight_column: column to weight by (e.g., 'Volume'); required if metric='volume'
    - kind: 'hist' for histogram/line plot, 'cumulative' for cumulative curve
    - metric: 'number' for number percentage, 'volume' for weighted percentage
    - bins: number of bins
    """
    
    # Determine weights
    if metric == 'number':
        weights = None
    elif metric == 'volume':
        if weight_column is None:
            raise ValueError("weight_column must be provided for volume metric")
        weights = df[weight_column]
    else:
        raise ValueError("metric must be 'number' or 'volume'")
    
    # Compute histogram
    counts, bin_edges = np.histogram(
        df[x_column], 
        bins=bins,
        weights=weights
    )
    
    # Convert to percentage
    percentages = 100 * counts / counts.sum()
    
    # Bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Cumulative if requested
    if kind == 'cumulative':
        percentages = np.cumsum(percentages)
    
    # Plot
    plt.figure(figsize=(8,5))
    color = 'blue' if metric=='number' else 'green'
    
    plt.plot(bin_centers, percentages, linestyle='-', color=color, label=f"{metric.capitalize()} %")
    plt.xlabel(x_column.replace('_',' ').capitalize())
    
    # Y-axis label
    if metric == 'number':
        ylabel = "Percentage number of grains (%)"
    else:
        ylabel = "Weighted Percentage (%)"
    
    if kind == 'cumulative':
        ylabel = "Cumulative " + ylabel
    
    plt.ylabel(ylabel)
    plt.title(f"{kind.capitalize()} Distribution of {x_column.replace('_',' ').capitalize()}")
    plt.grid(False)
    plt.legend()
    plt.show()
    
plot_distribution(Properties, x_column='equivalent_diameter', weight_column = 'Volume', kind='hist', metric='number', bins=30)
plot_distribution(Properties, x_column='equivalent_diameter', weight_column = 'Volume', kind='cumulative', metric='number', bins=30)
plot_distribution(Properties, x_column='equivalent_diameter', weight_column = 'Volume', kind='hist', metric='volume', bins=30)
plot_distribution(Properties, x_column='equivalent_diameter', weight_column = 'Volume', kind='cumulative', metric='volume', bins=30)