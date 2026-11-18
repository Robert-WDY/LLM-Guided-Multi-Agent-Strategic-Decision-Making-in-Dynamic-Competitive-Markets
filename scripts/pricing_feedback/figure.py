"""Profit effects with seed-paired intervals, in separate market units."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path
from study import OUT,read


def draw():
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    offline=read(OUT/'units.json')
    real=read(OUT/'real-units.json')
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.2),sharey=True)
    positions=[5,4,2.8,1.8,.6,-.4]
    for ax,(m,title) in zip(axes,[('econ','EconEvals 双寡头'),('native','原生生鲜市场')]):
        scale=1 if m=='econ' else 1.e4
        for j,(source,comparison) in enumerate((s,a) for s in ('rule','deepseek','doubao') for a in ('feedback','wide')):
            k='conditional-'+comparison
            z=offline[m]['pairs'][k] if source=='rule' else real[source+'-'+m]['results'][k]['profit']
            color='#ab492f' if comparison=='feedback' else '#287270'
            lower,upper=[x/scale for x in z['ci']];point=z['mean']/scale
            ax.errorbar(point,positions[j],xerr=[[point-lower],[upper-point]],fmt='o',
                        capsize=3.5,color=color,linewidth=2,markersize=5)
        ax.axvline(0,color='#535953',linewidth=1)
        ax.grid(axis='x',color='#dededb',linewidth=.6)
        ax.set_axisbelow(True);ax.set_title(title,loc='left',fontweight='bold')
        ax.set_xlabel('整局折现利润差（基准单位）' if m=='econ' else '整局折现利润差（万模拟元）')
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='y',length=0)
    axes[0].set_yticks(positions,labels=['规则草稿 · 相对保留','相对宽搜索','DeepSeek · 相对保留',
                                          '相对宽搜索','豆包 · 相对保留','相对宽搜索'])
    fig.suptitle('修正价格单位后，对手建模的整局收益变化',fontweight='bold',fontsize=15)
    fig.text(.5,.025,'点为同种子利润均差，横线为 95% 配对区间。规则草稿 48 个种子，每种真实模型 32 个种子。',
             ha='center',fontsize=9)
    fig.subplots_adjust(left=.20,right=.97,bottom=.21,top=.80,wspace=.12)
    fig.savefig(OUT/'profit-effects.png',dpi=160,facecolor='white')
    plt.close(fig)


if __name__=='__main__':draw()

