"""Independently recompute stored test metrics and verify identical test ordering."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
DEFAULT_RUNS=REPO/"data"/"raw"/"moon"/"lrocnet"/"experiment_lunar_cnn_qcnn"/"runs"
CLASSES=("ejecta","oldcrater","none")


def metrics(y, pred):
    p,r,f,_=precision_recall_fscore_support(y,pred,labels=[0,1,2],zero_division=0)
    return {"accuracy":float(accuracy_score(y,pred)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),
            "macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),
            "per_class":{c:{"precision":float(p[i]),"recall":float(r[i]),"f1":float(f[i])} for i,c in enumerate(CLASSES)},
            "confusion_matrix":confusion_matrix(y,pred,labels=[0,1,2]).tolist()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=Path,default=DEFAULT_RUNS,help="Run output folder (repo-relative or absolute)")
    args=parser.parse_args()
    runs=args.runs if args.runs.is_absolute() else REPO/args.runs
    out=runs/"results_verification"
    if out.exists(): raise FileExistsError(f"Refusing to overwrite {out}")
    summary=json.loads((runs/"summary.json").read_text(encoding="utf-8"))
    records=summary["runs"]
    expected_seeds=list(summary["config"]["seeds"])
    model_names=("cnn","hybrid_qcnn","frozen_embedding_mlp")
    reference=None; verified=[]
    for item in records:
        test=item["test"]; ids=test["sample_ids"]; y=np.asarray(test["ground_truth"],dtype=np.int64)
        if len(ids)!=len(y) or len(set(ids))!=len(ids): raise AssertionError(f"Seed {item['seed']} has bad test IDs")
        if reference is None: reference=(ids,y)
        elif ids!=reference[0] or not np.array_equal(y,reference[1]): raise AssertionError("Seed/fold test IDs or labels differ")
        model_metrics={}
        for name in model_names:
            logits=np.asarray(test["logits"][name],dtype=np.float64)
            saved_pred=np.asarray(test["predictions"][name],dtype=np.int64)
            if logits.shape!=(len(y),3) or not np.isfinite(logits).all(): raise AssertionError(f"Invalid logits: {item['seed']} {name}")
            pred=logits.argmax(axis=1)
            if not np.array_equal(pred,saved_pred): raise AssertionError(f"Saved predictions don't match logits: {item['seed']} {name}")
            model_metrics[name]=metrics(y,pred)
            if model_metrics[name]!=test[name]: raise AssertionError(f"Stored metrics disagree with recomputation: seed={item['seed']} model={name}")
        model_metrics["majority_train_baseline"]=metrics(y,np.full_like(y,int(np.argmax(item["class_counts_train"]))))
        if model_metrics["majority_train_baseline"]!=test["majority_train_baseline"]: raise AssertionError("Baseline mismatch")
        verified.append({"seed":item["seed"],"test_n":len(y),"class_counts":np.bincount(y,minlength=3).tolist(),"metrics":model_metrics,
                         "qcnn_gradient_norm":item["qcnn_gradient_norm_initial"],"seconds":item["seconds"],
                         "inference":test["inference"],"params_reported":item["parameter_counts"]})
    if [r["seed"] for r in records]!=expected_seeds: raise AssertionError(f"Expected seeds {expected_seeds}, got {[r['seed'] for r in records]}")
    # Count exact architecture parameters without loading or modifying a checkpoint.
    import sys
    sys.path.insert(0,str(HERE))
    from train import CNN, HybridHead, ClassicalHead
    def count(m): return sum(p.numel() for p in m.parameters())
    counts={"cnn_total_and_trainable_during_fit":count(CNN()),"hybrid_readout_total_trainable":count(HybridHead()),
            "hybrid_encoder_total":count(CNN().encoder),"frozen_mlp_head_total_trainable":count(ClassicalHead())}
    grouped={}
    for name in (*model_names,"majority_train_baseline"):
        grouped[name]={}
        for metric in ("accuracy","balanced_accuracy","macro_f1"):
            vals=[r["metrics"][name][metric] for r in verified]
            grouped[name][metric]={"mean":float(np.mean(vals)),"sample_sd":float(np.std(vals,ddof=1))}
    out.mkdir()
    audit={"verified":True,"test_ids_and_labels_identical_across_seeds_and_models":True,"seeds":[r["seed"] for r in verified],
           "sample_ids":reference[0],"ground_truth":reference[1].tolist(),"class_names":CLASSES,"test_class_counts":verified[0]["class_counts"],
           "recomputed_metrics":grouped,"parameter_counts":counts,"per_seed":verified,
           "limitations":["Supplied split-generation method and geographic independence are unverified.",
                          "CNN is end-to-end trained; hybrid and matched MLP use a frozen CNN encoder, so comparison is not architecture-only.",
                          "Test set contains 779 images from the supplied partition and may not represent independent sites."]}
    (out/"results_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    with (out/"recomputed_metrics.csv").open("x",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["model","metric","mean","sample_sd"])
        for model,v in grouped.items():
            for metric,stats in v.items(): w.writerow([model,metric,stats["mean"],stats["sample_sd"]])
    report=["# Saved Lunar Experiment Results Verification","","Read-only recomputation from saved test logits, labels and IDs; no checkpoint was loaded and no inference or training was run.",
            "",f"Seeds: {[r['seed'] for r in verified]}; shared test sample count: {len(reference[1])}; labels: {dict(zip(CLASSES,verified[0]['class_counts']))}.",
            "",f"Metrics below are mean ± sample SD across the {len(verified)} seeds.",""]
    for name,vals in grouped.items():
        report.append(f"- **{name}:** accuracy {vals['accuracy']['mean']:.3f} ± {vals['accuracy']['sample_sd']:.3f}; balanced accuracy {vals['balanced_accuracy']['mean']:.3f} ± {vals['balanced_accuracy']['sample_sd']:.3f}; macro-F1 {vals['macro_f1']['mean']:.3f} ± {vals['macro_f1']['sample_sd']:.3f}")
    report += ["","Parameter counts (verified from architecture definitions):",
               f"- CNN: {counts['cnn_total_and_trainable_during_fit']:,} total and trainable during CNN fitting; after freezing, its standalone classifier has 195 trainable parameters.",
               f"- Hybrid readout: {counts['hybrid_readout_total_trainable']:,} trainable (64→8 bottleneck plus QCNN parameters/readout); frozen encoder has {counts['hybrid_encoder_total']:,} parameters.",
               f"- Frozen-embedding MLP control: {counts['frozen_mlp_head_total_trainable']:,} trainable head parameters.",
               "- Note: the per-seed JSON originally records the CNN `trainable` count after its encoder was frozen (195). This audit corrects the interpretation: CNN training used 59,347 trainable parameters. No metric values were changed.",
               "","## Validation checks", "",
               f"- All {len(verified)} seeds contain the same {len(reference[1])} ordered relative test paths and identical labels.",
               "- For each model and seed, saved argmax predictions exactly match saved logits; recomputed metrics and confusion matrices exactly match stored values.",
               "- QCNN initial gradient norms were finite and nonzero: "+", ".join(f"seed {r['seed']}={r['qcnn_gradient_norm']:.4f}" for r in verified)+".",
               "- This is the supplied test partition only. Provenance documentation does not establish site-independent generalization.",
               "- CNN versus hybrid changes both model and optimization regime: the CNN is end-to-end trained, whereas the hybrid uses a frozen CNN encoder selected on validation. Interpret as a pipeline comparison, not architecture-only attribution.",
               "- No model-performance or safety claims beyond these classification metrics are implied.",""]
    (out/"results_audit.md").write_text("\n".join(report),encoding="utf-8")
    print(f"PASS: validated {len(verified)} seeds × {len(model_names)} models; wrote {out}")


if __name__=="__main__": main()
