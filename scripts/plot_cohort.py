"""Plot Table 1 of the RSNA data paper. Counts cover the whole challenge."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
with (root/"reports/published-cohort.csv").open() as f:
    sites = list(csv.DictReader(f))
fig, axes = plt.subplots(1,2,figsize=(10,3.5),layout="constrained",sharey=True)
labels = ["Biopsy-confirmed cancer","Benign biopsy","Recalled, then negative or benign","Screen negative or benign"]
keys = list(sites[0])[1:]
for ax, row in zip(axes, sites):
    values = [int(row[k]) for k in keys]
    bars = ax.barh(range(4),values,color=["#7a383c","#9b7753","#53938a","#225960"])
    ax.bar_label(bars, labels=[f"{n:,}" for n in values],padding=4,fontsize=9)
    ax.set(yticks=range(4),yticklabels=labels,xlabel="Screening exams",title=row["site"],xlim=(0,8800))
    ax.spines[["right","top","left"]].set_visible(False)
axes[0].invert_yaxis()
fig.savefig(root/"reports/published-cohort.png",dpi=180)
