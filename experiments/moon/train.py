"""Reproducible LROCNet CNN/QCNN coursework experiment; isolated from Mars pipeline."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision.transforms import functional as TF

from src.quantum.circuits import BatchedQCNNHead

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]  # config paths are relative to the repository root
CLASSES = ("ejecta", "oldcrater", "none")
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}


class LunarImages(Dataset):
    def __init__(self, root: Path, split: str):
        self.items = []
        for label in CLASSES:
            folder = root / split / label
            paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"})
            self.items.extend((p, CLASS_TO_ID[label]) for p in paths)
        if not self.items:
            raise RuntimeError(f"No images in {root / split}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        with Image.open(path) as im:
            im.load()
            if im.size != (227, 227): raise ValueError(f"Unexpected image dimensions {im.size}: {path}")
            x = TF.to_tensor(im.convert("L"))
        return x, label


class Encoder(nn.Module):
    """Small CNN with 64-d global pooled representation."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, 5, stride=2, padding=2), nn.BatchNorm2d(24), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Sequential(nn.Flatten(), nn.Linear(96, 64), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.projection(self.features(x))


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.classifier = nn.Linear(64, 3)

    def forward(self, x):
        return self.classifier(self.encoder(x))


class HybridHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.bottleneck = nn.Linear(64, 8)
        self.qcnn = BatchedQCNNHead(n_qubits=8, encoding="angle", use_entanglement=True)

    def forward(self, embedding):
        angles = (torch.tanh(self.bottleneck(embedding)) + 1.0) * (torch.pi / 2.0)
        return self.qcnn(angles)


class ClassicalHead(nn.Module):
    """Matched frozen-embedding control: 64->8->3, 547 trainable parameters."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(64, 8), nn.GELU(), nn.Linear(8, 3))

    def forward(self, embedding):
        return self.layers(embedding)


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def class_weights(ds, device):
    counts = np.bincount([label for _, label in ds.items], minlength=3)
    if np.any(counts == 0): raise RuntimeError(f"Missing training class: {counts.tolist()}")
    weights = len(ds) / (3.0 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device), counts.tolist()


def loader(ds, batch_size, shuffle=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


def collect_predictions(model, batches, device, input_kind="image"):
    model.eval(); ys=[]; logits=[]
    with torch.no_grad():
        for x,y in batches:
            x=x.to(device, non_blocking=True)
            z=model(x)
            logits.append(z.detach().cpu().numpy()); ys.extend(y.numpy().tolist())
    return np.concatenate(logits), np.asarray(ys, dtype=np.int64)


def score(y, pred):
    p,r,f,_=precision_recall_fscore_support(y,pred,labels=[0,1,2],zero_division=0)
    return {"accuracy":float(accuracy_score(y,pred)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),
            "macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),
            "per_class":{c:{"precision":float(p[i]),"recall":float(r[i]),"f1":float(f[i])} for i,c in enumerate(CLASSES)},
            "confusion_matrix":confusion_matrix(y,pred,labels=[0,1,2]).tolist()}


def fit(model, train_batches, val_batches, optimizer, criterion, device, epochs, patience, pilot=False,
        checkpoint_path=None, resume=False, history_log_path=None):
    best=-1.0; best_state=None; stale=0; history=[]; start_epoch=0
    if resume and checkpoint_path is not None and checkpoint_path.exists():
        saved=torch.load(checkpoint_path,map_location=device,weights_only=False)
        model.load_state_dict(saved["current_state"]); optimizer.load_state_dict(saved["optimizer_state"])
        best=float(saved["best_metric"]); best_state=saved["best_state"]; stale=int(saved["stale"])
        history=saved["history"]; start_epoch=int(saved["epoch"])
        random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"]); torch.set_rng_state(saved["torch_rng"])
        if torch.cuda.is_available() and saved.get("cuda_rng") is not None: torch.cuda.set_rng_state_all(saved["cuda_rng"])
    for epoch in range(start_epoch,epochs):
        model.train(); loss_sum=0.; n=0; t0=time.perf_counter()
        for bi,(x,y) in enumerate(train_batches):
            if pilot and bi >= 3: break
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True); logits=model(x); loss=criterion(logits,y)
            loss.backward(); optimizer.step(); loss_sum += float(loss.detach())*len(y); n += len(y)
        model.eval(); pred=[]; truth=[]
        with torch.no_grad():
            for bi,(x,y) in enumerate(val_batches):
                if pilot and bi >= 2: break
                x=x.to(device,non_blocking=True); logits=model(x)
                pred.extend(logits.argmax(1).cpu().tolist()); truth.extend(y.tolist())
        metric=f1_score(truth,pred,average="macro",zero_division=0) if truth else 0.
        history.append({"epoch":epoch+1,"train_loss":loss_sum/max(n,1),"val_macro_f1":float(metric),"seconds":time.perf_counter()-t0})
        if history_log_path is not None and not pilot:
            history_log_path.parent.mkdir(parents=True,exist_ok=True)
            with history_log_path.open("a",encoding="utf-8") as log:
                log.write(json.dumps(history[-1])+"\n")
        if metric > best:
            best=metric; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else: stale+=1
        if checkpoint_path is not None and not pilot:
            checkpoint_path.parent.mkdir(parents=True,exist_ok=True)
            tmp=checkpoint_path.with_suffix(checkpoint_path.suffix+".tmp")
            torch.save({"current_state":model.state_dict(),"best_state":best_state,"optimizer_state":optimizer.state_dict(),
                        "epoch":epoch+1,"best_metric":best,"stale":stale,"history":history,
                        "python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),
                        "cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},tmp)
            os.replace(tmp,checkpoint_path)
        if not pilot and stale>=patience: break
    if best_state is not None: model.load_state_dict(best_state)
    return history


def embed_split(encoder, ds, device, batch_size):
    encoder.eval(); allx=[]; ally=[]
    with torch.no_grad():
        for x,y in loader(ds,batch_size):
            allx.append(encoder(x.to(device)).cpu()); ally.append(y)
    return TensorDataset(torch.cat(allx),torch.cat(ally))


def parameter_counts(model):
    return {"total":sum(p.numel() for p in model.parameters()),"trainable":sum(p.numel() for p in model.parameters() if p.requires_grad)}


def pilot_indices(ds, per_class):
    # Deterministic, class-balanced subset so the smoke pilot exercises all logits.
    chosen=[]
    for class_id in range(3):
        candidates=[i for i,(_,y) in enumerate(ds.items) if y==class_id]
        if len(candidates)<per_class: raise RuntimeError("Not enough examples for balanced pilot")
        chosen.extend(candidates[:per_class])
    return sorted(chosen)


def inference_timing(model, batches, device):
    model.eval(); total=0; elapsed=0.
    if device.type=="cuda": torch.cuda.synchronize(device)
    start=time.perf_counter()
    with torch.no_grad():
        for x,_ in batches:
            model(x.to(device,non_blocking=True)); total+=len(x)
    if device.type=="cuda": torch.cuda.synchronize(device)
    elapsed=time.perf_counter()-start
    return {"total_seconds":elapsed,"milliseconds_per_image":1000*elapsed/max(total,1),"samples":total}


def safe_save(obj, path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f: json.dump(obj,f,indent=2)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--pilot",action="store_true",help="Small train/validation-only runtime and gradient pilot; never opens test split")
    parser.add_argument("--resume",action="store_true",help="Resume partial seed/stage checkpoints in the configured output directory")
    parser.add_argument("--config",type=Path,default=HERE/"config.yaml")
    args=parser.parse_args()
    import yaml
    cfg=yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root=REPO/cfg["dataset"]; out=REPO/cfg["output"]
    if args.pilot: out=out.parent/f"pilot_{time.strftime('%Y%m%d_%H%M%S')}"
    if out.exists() and not args.resume: raise FileExistsError(f"Refusing to overwrite existing run directory: {out}")
    if args.resume and not out.exists(): raise FileNotFoundError(f"Cannot resume; output directory does not exist: {out}")
    device=torch.device("cuda" if cfg.get("device")=="cuda" and torch.cuda.is_available() else "cpu")
    train_ds=LunarImages(root,"train"); val_ds=LunarImages(root,"val")
    if len(train_ds)!=3608 or len(val_ds)!=613: raise RuntimeError(f"Unexpected split size train={len(train_ds)} val={len(val_ds)}")
    test_ds=None  # Construct only after train/validation model selection is complete for each seed.
    seeds=[42] if args.pilot else cfg["seeds"]
    epochs=min(2,int(cfg["max_epochs"])) if args.pilot else int(cfg["max_epochs"])
    batch_size=int(cfg["batch_size"]); all_results=[]; start_all=time.perf_counter()
    print(json.dumps({"device":str(device),"gpu":torch.cuda.get_device_name(0) if device.type=="cuda" else None,
                      "split_sizes":{"train":len(train_ds),"val":len(val_ds),"test":"locked_until_final_evaluation"},
                      "class_map":CLASS_TO_ID,"pilot":args.pilot},indent=2),flush=True)
    out.mkdir(parents=True,exist_ok=True)
    (out/"logs").mkdir(exist_ok=True); (out/"checkpoints").mkdir(exist_ok=True)
    for seed in seeds:
        seed_everything(seed); run_dir=out/"logs"/f"seed_{seed}"; run_dir.mkdir(exist_ok=True)
        prior_result=run_dir/"results.json"
        if args.resume and prior_result.exists():
            all_results.append(json.loads(prior_result.read_text(encoding="utf-8")))
            print(json.dumps({"seed":seed,"status":"already_complete_skipped"}),flush=True)
            continue
        w,counts=class_weights(train_ds,device)
        tr=loader(train_ds,batch_size,True); va=loader(val_ds,batch_size)
        cnn=CNN().to(device); cnn_opt=torch.optim.AdamW(cnn.parameters(),lr=float(cfg["learning_rate_cnn"]),weight_decay=float(cfg["weight_decay"]))
        criterion=nn.CrossEntropyLoss(weight=w)
        if args.pilot:
            # Use bounded, deterministic subsets; the pilot cannot inspect the held-out test partition.
            tr=loader(Subset(train_ds,pilot_indices(train_ds,64)),batch_size,True)
            va=loader(Subset(val_ds,pilot_indices(val_ds,32)),batch_size)
        cnn_t=time.perf_counter(); cnn_final=out/"checkpoints"/f"cnn_seed_{seed}.pt"
        cnn_stage=out/"checkpoints"/f"cnn_seed_{seed}_training.pt"
        if args.resume and cnn_final.exists():
            cnn.load_state_dict(torch.load(cnn_final,map_location=device,weights_only=False)["state_dict"])
            cnn_hist=[]; cnn_seconds=0.
        else:
            cnn_hist=fit(cnn,tr,va,cnn_opt,criterion,device,epochs,int(cfg["patience"]),args.pilot,cnn_stage,args.resume,run_dir/"cnn_epochs.jsonl")
            cnn_seconds=time.perf_counter()-cnn_t
        cnn_parameter_counts=parameter_counts(cnn)
        if not args.pilot:
            torch.save({"state_dict":cnn.state_dict(),"seed":seed,"class_map":CLASS_TO_ID},cnn_final)
        # Freeze encoder selected solely on training/validation data. Extract embeddings for supplied splits.
        encoder=cnn.encoder
        for p in encoder.parameters(): p.requires_grad=False
        splits={"train":embed_split(encoder,train_ds,device,batch_size),"val":embed_split(encoder,val_ds,device,batch_size)}
        if args.pilot:
            it=pilot_indices(train_ds,64); iv=pilot_indices(val_ds,32)
            splits["train"]=TensorDataset(splits["train"].tensors[0][it],splits["train"].tensors[1][it])
            splits["val"]=TensorDataset(splits["val"].tensors[0][iv],splits["val"].tensors[1][iv])
        # Head-only training on frozen, train-fitted CNN embeddings.
        head_weights=w.detach().cpu()
        qcnn=HybridHead(); qopt=torch.optim.AdamW(qcnn.parameters(),lr=float(cfg["learning_rate_head"]),weight_decay=float(cfg["weight_decay"]))
        qcriterion=nn.CrossEntropyLoss(weight=head_weights)
        qtr=loader(splits["train"],batch_size,True); qva=loader(splits["val"],batch_size)
        # Gradient verification is mandatory before a run can proceed.
        qcnn.train(); qx,qy=next(iter(qtr)); qlog=qcnn(qx); qloss=qcriterion(qlog,qy); qloss.backward()
        grad_norm=float(sum(p.grad.detach().norm().item() for p in qcnn.parameters() if p.grad is not None))
        if not np.isfinite(grad_norm) or grad_norm<=0: raise RuntimeError(f"QCNN gradient check failed: {grad_norm}")
        qcnn.zero_grad(set_to_none=True)
        q_t=time.perf_counter(); qfinal=out/"checkpoints"/f"hybrid_seed_{seed}.pt"; qstage=out/"checkpoints"/f"hybrid_seed_{seed}_training.pt"
        if args.resume and qfinal.exists():
            qcnn.load_state_dict(torch.load(qfinal,map_location="cpu",weights_only=False)["state_dict"]); qhist=[]; q_seconds=0.
        else:
            qhist=fit(qcnn,qtr,qva,qopt,qcriterion,torch.device("cpu"),epochs,int(cfg["patience"]),args.pilot,qstage,args.resume,run_dir/"qcnn_epochs.jsonl"); q_seconds=time.perf_counter()-q_t
        if not args.pilot:
            torch.save({"state_dict":qcnn.state_dict(),"seed":seed,"encoder_checkpoint":f"cnn_seed_{seed}.pt","class_map":CLASS_TO_ID},qfinal)
        # Matched classical control on the same frozen embeddings and optimization data.
        mlp=ClassicalHead(); mopt=torch.optim.AdamW(mlp.parameters(),lr=float(cfg["learning_rate_head"]),weight_decay=float(cfg["weight_decay"]))
        mfinal=out/"checkpoints"/f"frozen_mlp_seed_{seed}.pt"; mstage=out/"checkpoints"/f"frozen_mlp_seed_{seed}_training.pt"
        if args.resume and mfinal.exists(): mlp.load_state_dict(torch.load(mfinal,map_location="cpu",weights_only=False)["state_dict"]); mhist=[]
        else: mhist=fit(mlp,qtr,qva,mopt,qcriterion,torch.device("cpu"),epochs,int(cfg["patience"]),args.pilot,mstage,args.resume,run_dir/"frozen_mlp_epochs.jsonl")
        if not args.pilot: torch.save({"state_dict":mlp.state_dict(),"seed":seed,"class_map":CLASS_TO_ID},mfinal)
        row={"seed":seed,"class_counts_train":counts,"parameter_counts":{"cnn_during_fit":cnn_parameter_counts,
             "cnn_classifier_after_encoder_freeze":parameter_counts(cnn),"hybrid_head":parameter_counts(qcnn),"frozen_mlp_head":parameter_counts(mlp)},
             "qcnn_gradient_norm_initial":grad_norm,"seconds":{"cnn_fit":cnn_seconds,"qcnn_head_fit":q_seconds},
             "history":{"cnn":cnn_hist,"qcnn":qhist,"frozen_mlp":mhist}}
        if not args.pilot:
            # Test predictions are generated only after all training/model choices for this seed are fixed.
            test_ds=LunarImages(root,"test")
            splits["test"]=embed_split(encoder,test_ds,device,batch_size)
            cnn_logits,cy=collect_predictions(cnn,loader(test_ds,batch_size),device)
            te=splits["test"]; q_logits,qy=collect_predictions(qcnn,loader(te,batch_size),torch.device("cpu")); m_logits,_=collect_predictions(mlp,loader(te,batch_size),torch.device("cpu"))
            if not np.array_equal(cy,qy): raise RuntimeError("Models were evaluated on different test labels/order")
            row["test"]={"sample_count":int(len(cy)),"class_distribution":np.bincount(cy,minlength=3).tolist(),
                         "cnn":score(cy,cnn_logits.argmax(1)),"hybrid_qcnn":score(cy,q_logits.argmax(1)),"frozen_embedding_mlp":score(cy,m_logits.argmax(1)),
                         "majority_train_baseline":score(cy,np.full_like(cy,int(np.argmax(counts)))),
                         "inference":{"cnn":inference_timing(cnn,loader(test_ds,batch_size),device),
                                      "hybrid_qcnn":inference_timing(qcnn,loader(te,batch_size),torch.device("cpu")),
                                      "frozen_embedding_mlp":inference_timing(mlp,loader(te,batch_size),torch.device("cpu"))},
                         "sample_ids":[p.relative_to(root).as_posix() for p,_ in test_ds.items],
                         "ground_truth":cy.tolist(),"predictions":{"cnn":cnn_logits.argmax(1).tolist(),"hybrid_qcnn":q_logits.argmax(1).tolist(),"frozen_embedding_mlp":m_logits.argmax(1).tolist()},
                         "logits":{"cnn":cnn_logits.tolist(),"hybrid_qcnn":q_logits.tolist(),"frozen_embedding_mlp":m_logits.tolist()}}
        all_results.append(row)
        if not args.pilot:
            if prior_result.exists() and not args.resume: raise FileExistsError(prior_result)
            if args.resume and prior_result.exists(): prior_result.unlink()
            safe_save(row,prior_result)
            history_path=run_dir/"history.csv"
            if history_path.exists() and args.resume: history_path.unlink()
            with history_path.open("x",newline="",encoding="utf-8") as f:
                writer=csv.DictWriter(f,fieldnames=["model","epoch","train_loss","val_macro_f1","seconds"]); writer.writeheader()
                for model_name,hist in row["history"].items():
                    for h in hist: writer.writerow({"model":model_name,**h})
        print(json.dumps({"seed":seed,"cnn_seconds":cnn_seconds,"qcnn_seconds":q_seconds,"qcnn_gradient_norm":grad_norm,"pilot":args.pilot}),flush=True)
    summary={"config":cfg,"device":str(device),"pilot":args.pilot,"elapsed_seconds":time.perf_counter()-start_all,"runs":all_results}
    summary_path=out/"summary.json"
    if args.resume and summary_path.exists(): summary_path.unlink()
    safe_save(summary,summary_path)
    if not args.pilot:
        if args.resume and (out/"metrics.csv").exists(): (out/"metrics.csv").unlink()
        with (out/"metrics.csv").open("x",newline="",encoding="utf-8") as f:
            fields=["seed","model","accuracy","balanced_accuracy","macro_f1","test_samples","test_class_counts"]
            writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
            for item in all_results:
                for model_name in ("cnn","hybrid_qcnn","frozen_embedding_mlp","majority_train_baseline"):
                    values=item["test"][model_name]
                    writer.writerow({"seed":item["seed"],"model":model_name,"accuracy":values["accuracy"],
                                     "balanced_accuracy":values["balanced_accuracy"],"macro_f1":values["macro_f1"],
                                     "test_samples":item["test"]["sample_count"],"test_class_counts":item["test"]["class_distribution"]})
    print(json.dumps({"output":str(out),"elapsed_seconds":summary["elapsed_seconds"],"completed_seeds":len(seeds),"pilot":args.pilot},indent=2),flush=True)


if __name__=="__main__": main()
