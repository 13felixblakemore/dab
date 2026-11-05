"""Plotting utilities; import matplotlib only for plotting."""
import csv
from pathlib import Path
import numpy as np


def plot_training(csv_path,output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with open(csv_path,encoding="utf-8") as f:
        history=list(csv.DictReader(f))
    x=np.array([float(row["steps"]) for row in history]);
    fig,ax=plt.subplots(figsize=(9,5))
    for name in ("loss","fb_loss","reg_loss"):
        if name in history[0]:
            vals=np.array([float(row[name]) if row[name] else np.nan for row in history])
            ax.plot(x,vals,label=name,alpha=0.9)
    ax.set(xlabel="Environment transitions",ylabel="Epoch-mean loss",title="FB training objective")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path=Path(output_path);output_path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output_path,dpi=160)
    plt.close(fig)


def plot_trajectory(env,states,goal,output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    fig,ax=plt.subplots(figsize=(6,6))
    for w in env.wall_blocks:
        ax.add_patch(Rectangle((w.xmin,w.ymin),w.xmax-w.xmin,w.ymax-w.ymin,color="black"))
    states=np.asarray(states)
    if len(states):
        ax.plot(states[:,0],states[:,1],color="tab:blue",linewidth=1.4,alpha=0.8)
        ax.scatter(*states[0],marker="o",s=65,color="tab:green",zorder=4,label="Start")
    ax.add_patch(Circle(tuple(goal),radius=env.threshold_distance,fill=False,color="tab:red",linewidth=1.6))
    ax.scatter(*goal,marker="*",s=130,color="tab:red",zorder=4,label="Goal")
    ax.set(xlim=(0,env.size),ylim=(0,env.size),xlabel="x",ylabel="y",title="FB navigation trial")
    ax.set_aspect("equal");ax.legend(loc="upper left");fig.tight_layout()
    output_path=Path(output_path);output_path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output_path,dpi=160)
    plt.close(fig)
