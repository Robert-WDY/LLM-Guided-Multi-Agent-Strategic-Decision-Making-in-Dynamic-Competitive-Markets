"""Paper figure for independent LLM pricing and the predeclared deployment margins."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,FuncFormatter

from results import OUT,THRESH,read


def main():
    fig,axes=plt.subplots(1,2,figsize=(10.4,3.9),sharey=True)
    labels=["Qwen · menu","SmolLM · menu","Qwen · numeric","SmolLM · numeric"]
    for ax,m,title in zip(axes,("econ","native"),("EconEvals duopoly","Fresh-food market")):
        ax.axvline(0,color="#8b939b",lw=1)
        ax.axvline(THRESH[m],color="#996629",lw=1.5,ls="--")
        for i,(tag,name) in enumerate((("independent","qwen"),("independent","smol"),("numeric","qwen"),("numeric","smol"))):
            v=read(OUT/f"{tag}-results.json")[f"{m}-{name}"]["0.99"]["keep"]
            lo,hi=v["bounds"]
            ax.errorbar(v["mean"],3-i,xerr=[[v["mean"]-lo],[hi-v["mean"]]],fmt="o",
                        color=("#22567b" if i<2 else "#097769"),capsize=4)
        ax.set_title(title);ax.set_yticks(range(3,-1,-1),labels)
        ax.set_ylim(-.65,3.65);ax.grid(axis="x",alpha=.17)
        ax.spines[["right","top","left"]].set_visible(False);ax.tick_params(axis="y",length=0)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f"{x:,.0f}"))
        ax.set_xlabel("Benchmark units per opportunity" if m=="econ" else "Simulated yuan per opportunity")
    fig.suptitle("Does acting on advice beat keeping the model’s draft?",fontweight="bold",fontsize=14)
    fig.text(.02,.03,"Dots show mean terminal profit differences; bars are paired bootstrap 1.25th–98.75th percentiles.\n"
             "Dashed line marks the prespecified economic threshold. Model pricing occurs in simulated markets.",fontsize=8)
    fig.tight_layout(rect=(0,.15,1,.94))
    fig.savefig(OUT/"independent-pricing.png",dpi=185)
    plt.close(fig)


if __name__=="__main__":
    main()
