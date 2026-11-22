from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(12, 4))
ax.set(xlim=(0,12),ylim=(0,4))
ax.axis("off")
ink, green = "#1e3741", "#125b55"

def box(x,y,w,title,detail,fill):
    ax.add_patch(FancyBboxPatch((x,y),w,.84,boxstyle="round,pad=0.05,rounding_size=.08",
                                facecolor=fill,edgecolor="#90aaa2"))
    ax.text(x+.13,y+.55,title,fontsize=12,color=ink,weight="bold")
    ax.text(x+.13,y+.19,detail,fontsize=10,color=green)

def arrow(start,end):
    ax.annotate("",xy=end,xytext=start,arrowprops=dict(arrowstyle="->",color="#4a6564",lw=1.5))

ax.text(.18,3.65,"Two views, one breast score",fontsize=20,weight="bold",color=ink)
for y,view in [(2.12,"CC view"),(.72,"MLO view")]:
    box(.2,y,1.8,view,"One breast","#e6f1ed")
    box(2.75,y,2.66,"Shared ResNet-18","Spatial feature maps","#e6f1ed")
    box(6.05,y,2.14,"Attention pooling","View descriptor","#edf1f5")
    arrow((2.08,y+.43),(2.64,y+.43))
    arrow((5.5,y+.43),(5.96,y+.43))
    arrow((8.29,y+.43),(9.03,1.85))
box(9.12,1.42,2.55,"Breast classifier","CC + MLO features","#f1e8df")
ax.text(.2,.20,"The single-view comparison follows the MLO branch and uses its own classifier.",fontsize=11,color=ink)
fig.tight_layout()
fig.savefig(Path(__file__).resolve().parents[1]/"reports/model.png",dpi=180,bbox_inches="tight")
