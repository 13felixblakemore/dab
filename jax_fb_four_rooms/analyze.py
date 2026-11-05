"""Port of the supplied notebook's SR-distance clustering and adjacency analysis.

Run: python analyze.py --checkpoint results/default/fb_final.npz
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from fb_jax.agent import FBAgent
from fb_jax.environment import ContinuousFourRooms
from fb_jax.analysis import fb_features,sr_distance,kmedoids,observed_cluster_edges


def plot_clusters(env,states,labels,medoids,edges,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,ax=plt.subplots(figsize=(7,7))
    ax.scatter(states[:,0],states[:,1],c=labels,cmap="tab20",s=9,alpha=.75)
    for block in env.wall_blocks:
        ax.add_patch(Rectangle((block.xmin,block.ymin),block.xmax-block.xmin,
                               block.ymax-block.ymin,color="black",zorder=3))
    for (i,j),n in edges.items():
        ax.annotate("",xy=medoids[j],xytext=medoids[i],
                    arrowprops={"arrowstyle":"->","alpha":0.3,"lw":1},zorder=4)
    ax.scatter(medoids[:,0],medoids[:,1],marker="x",s=50,color="red",zorder=5)
    ax.set(xlim=(0,env.size),ylim=(0,env.size),title="FB successor-distance clusters and observed edges",
           xlabel="x",ylabel="y")
    ax.set_aspect("equal");fig.tight_layout()
    fig.savefig(output,dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--samples",type=int,default=400)
    parser.add_argument("--clusters",type=int,default=12)
    parser.add_argument("--edge-transitions",type=int,default=3000)
    parser.add_argument("--min-crossings",type=int,default=5)
    parser.add_argument("--seed",type=int,default=33)
    parser.add_argument("--output",default=None)
    args=parser.parse_args()
    if args.samples < 1 or not 1<=args.clusters<=args.samples:
        parser.error("Require samples >= clusters >= 1")
    agent=FBAgent.load(args.checkpoint)
    env=ContinuousFourRooms(size=agent.features.size,
                            include_stay_action=agent.config.n_actions==5,seed=args.seed)
    rng=np.random.default_rng(args.seed+22)
    states=np.asarray([env.sample_point() for _ in range(args.samples)])
    f,b=fb_features(agent,states)
    d=sr_distance(f,b)
    np.fill_diagonal(d,0)
    medoid_ids,labels,cost=kmedoids(d,args.clusters)
    medoids=states[medoid_ids]
    edges=observed_cluster_edges(agent,env,medoids,rng,args.edge_transitions,
                                 min_crossings=args.min_crossings)
    out=Path(args.output) if args.output else Path(args.checkpoint).parent
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/"clusters.npz",states=states,labels=labels,medoids=medoids,
                        distances=d)
    with (out/"cluster_edges.csv").open("w",newline="",encoding="utf-8") as handle:
        w=csv.writer(handle);w.writerow(["source","target","count"])
        w.writerows((a,b,n) for (a,b),n in edges.items())
    plot_clusters(env,states,labels,medoids,edges,out/"clusters.png")
    result={"num_samples":len(states),"clusters":args.clusters,"cost":cost,
            "observed_edges":len(edges),"output":str(out)}
    (out/"analysis_summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
