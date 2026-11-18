"""Paper figure with matched-seed intervals and separate market units."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from publication_stats import OUT, read


def main():
    first=read(OUT/"results.json")
    second=read(OUT/"delayed-results.json")
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.25),sharey=True)
    names=["Type disclosed at first decision","Type inferred after four rounds"]
    for i,(ax,m,title,unit) in enumerate(zip(axes,("econ","native"),("EconEvals duopoly","Fresh-food market"),
                                           ("Benchmark profit units","Simulated yuan"))):
        v=first[m]["0.99"]["policy_gap"]
        w=second[m]["test"]["inferred-uniform"]
        ax.axvline(0,color="#526173",lw=1,linestyle="--")
        for pos,row,key,color in ((1,v,"ci9875","#00796b"),(0,w,"ci975","#a34b18")):
            mean=row["mean"];lo,hi=row[key]
            ax.errorbar(mean,pos,xerr=[[mean-lo],[hi-mean]],fmt="o",capsize=5,color=color,lw=2)
        ax.set_title(title,fontsize=13)
        ax.set_ylim(-.6,1.6);ax.set_yticks([1,0],names)
        ax.set_xlabel(unit)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f"{x:,.0f}"))
        ax.spines[["top","right","left"]].set_visible(False)
        ax.grid(axis="x",alpha=.15)
        ax.tick_params(axis="y",length=0)
    fig.suptitle("Incremental terminal profit from opponent information",fontsize=15,fontweight="bold")
    fig.text(.02,.04,"Each estimate is paired against the same estimator with an uninformative prior. "
             "Independent protocols, 192 seeds per market each.\nIntervals adjusted within protocol: 98.75% for disclosure, 97.5% for inference. Profit discounted at 0.99.",
             fontsize=8,color="#455264")
    fig.tight_layout(rect=(0,.19,1,.91))
    fig.savefig(OUT/"information-profit.png",dpi=185,facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    main()
