import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve


def plot_test_results(y, scores, out):
    fig, axes = plt.subplots(1,3,figsize=(12,4),layout="constrained")
    for (name, p), color in zip(scores.items(), ["#ae683c", "#14736d"]):
        precision, recall, _ = precision_recall_curve(y,p)
        fpr, tpr, _ = roc_curve(y,p)
        observed, predicted = calibration_curve(y,p,n_bins=8,strategy="quantile")
        axes[0].plot(recall,precision,label=name,color=color)
        axes[1].plot(fpr,tpr,color=color)
        axes[2].plot(predicted,observed,"o-",color=color)
    axes[0].axhline(y.mean(),ls="--",color="#666666",label="Sample prevalence")
    axes[0].legend(fontsize=8)
    axes[0].set(xlabel="Recall",ylabel="Precision",title="Precision and recall")
    axes[1].set(xlabel="False-positive rate",ylabel="True-positive rate",title="ROC")
    axes[2].set(xlabel="Mean predicted score",ylabel="Observed cancer fraction",title="Calibration")
    for ax in axes[1:]:
        ax.plot([0,1],[0,1],ls="--",lw=.8,color="#666666")
    for ax in axes:
        ax.set(xlim=(0,1),ylim=(0,1))
        ax.spines[["right","top"]].set_visible(False)
    fig.savefig(out/"heldout-site.png",dpi=180)
    plt.close(fig)
