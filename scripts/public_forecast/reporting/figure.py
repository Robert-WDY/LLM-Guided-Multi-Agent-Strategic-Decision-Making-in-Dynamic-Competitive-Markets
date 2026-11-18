"""Fourth-settlement confusion tables for real focal firms; separate held-out opponent type."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from collect import OUT,read


def plot():
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    fig,axes=plt.subplots(1,2,figsize=(9.5,4.6))
    for m,ax,title in zip(('econ','native'),axes,('EconEvals 双寡头','原生生鲜')):
        cases=read(OUT/f'rtest-{m}-episodes.json.gz')
        matrix=[]
        for c in ('stable','demand','rival','reverse'):
            group=[r['argmax'][r['when']-7+3] for r in cases if r['condition']==c]
            matrix.append(np.bincount(group,minlength=3)/len(group))
        mat=np.array(matrix)
        ax.imshow(mat,vmin=0,vmax=1,cmap='Blues',aspect='auto')
        for (i,j),v in np.ndenumerate(mat):
            ax.text(j,i,f'{v:.1%}',ha='center',va='center',color='white' if v>.56 else '#152738',fontsize=12)
        ax.set_xticks(range(3),['判为稳定','判为需求','判为换策'])
        ax.set_yticks(range(4),['稳定','需求变化','跟价或压价','未见反向报价'])
        ax.set_title(title)
        for spine in ax.spines.values():spine.set_visible(False)
        ax.tick_params(length=0)
    fig.suptitle('市场变化后的第四次结算，归因器给出了什么判断',fontsize=15,fontweight='bold')
    fig.text(.5,.035,'每行 24 条真实草稿轨迹，包括两种模型的共同种子。单元格为行内占比。',ha='center',fontsize=9)
    fig.subplots_adjust(left=.13,right=.98,top=.80,bottom=.16,wspace=.42)
    fig.savefig(OUT/'attribution.png',dpi=155)
    plt.close(fig)


if __name__=='__main__':plot()

