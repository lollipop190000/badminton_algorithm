"""Plot fitted shot sensitivities and their conditional bootstrap intervals."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(run):
    coefficients=pd.read_csv(run/'coefficients.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,5),sharey=True,layout='constrained')
    shots=coefficients.shot_type.drop_duplicates().tolist()
    for ax,term,title in zip(axes,('beta','gamma'),('Hitter consumption sensitivity (beta)','Receiver consumption sensitivity (gamma)')):
        rows=coefficients[coefficients.term.eq(term)].set_index('shot_type').loc[shots]
        y=np.arange(len(shots))
        ax.hlines(y,rows.ci95_low,rows.ci95_high,color='#7b93a7',lw=2)
        ax.scatter(rows['final'],y,color='#155e75',zorder=3,label='Train + validation fit')
        ax.axvline(0,color='#cbd5e1',lw=1)
        ax.set_xlabel('Log odds coefficient per D / D*')
        ax.set_title(title,fontsize=11)
        ax.grid(axis='x',alpha=.2)
        ax.set_yticks(y,shots)
    axes[0].invert_yaxis()
    fig.suptitle('Shot-specific fixed-Q fit: 95% match-bootstrap intervals; no shot-order constraint',fontsize=12)
    fig.savefig(run/'coefficient_intervals.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    plot(parser.parse_args().run)
