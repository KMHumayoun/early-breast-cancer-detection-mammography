import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from .images import BreastImages
from .network import ScreeningModel


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(where):
    if where.type == "cuda":
        torch.cuda.synchronize()
    if where.type == "mps":
        torch.mps.synchronize()


def benchmark(height, batch, steps):
    where = device()
    torch.manual_seed(83)
    torch.set_num_threads(2)
    rows = []
    for views in (1, 2):
        model = ScreeningModel(views, pretrained=False).to(where).train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        images = torch.rand(batch, views, 1, height, round(height*2/3), device=where)
        labels = torch.arange(batch, device=where).remainder(2).float()
        for step in range(steps+2):
            if step == 2:
                synchronize(where)
                start = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(images), labels)
            loss.backward()
            opt.step()
        synchronize(where)
        rate = batch*steps / (time.perf_counter()-start)
        rows.append(dict(views=views, breasts_per_second=rate, measured_steps=steps))
        del model, opt, images, loss
    return dict(scope="Generated tensors; model training steps only. No clinical accuracy measurement.",
                device=str(where), machine=platform.machine(), torch=str(torch.__version__),
                height=height, width=round(height*2/3), batch=batch, rates=rows)


def read_index(path):
    raw = Path(path).read_bytes()
    return json.loads(raw)["breasts"], hashlib.sha256(raw).hexdigest()


def loader(rows, args, views, shuffle=False):
    dataset = BreastImages(rows, args.images, args.height, views)
    return DataLoader(dataset, batch_size=args.batch, shuffle=shuffle, num_workers=args.workers)


@torch.inference_mode()
def logits(model, batches, where):
    model.eval()
    return np.concatenate([model(x.to(where)).cpu().numpy() for x, _ in batches])


def train(args):
    rows, digest = read_index(args.index)
    fit = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "select"]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    views = ["CC", "MLO"] if args.model == "paired" else ["MLO"]
    torch.manual_seed(args.seed)
    where = device()
    model = ScreeningModel(len(views), pretrained=not args.random_init).to(where)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    pos = sum(r["cancer"] for r in fit)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(fit)-pos)/pos, device=where))
    train_batches = loader(fit, args, views, shuffle=True)
    val_batches = loader(validation, args, views)
    history, best = [], -1.
    for epoch in range(args.epochs):
        model.train()
        start = time.perf_counter()
        for x, y in train_batches:
            optim.zero_grad(set_to_none=True)
            loss = criterion(model(x.to(where)), y.to(where))
            loss.backward()
            optim.step()
        scores = logits(model, val_batches, where)
        ap = float(average_precision_score([r["cancer"] for r in validation], scores))
        history.append(dict(epoch=epoch+1, selection_ap=ap, seconds=time.perf_counter()-start))
        if ap > best:
            best = ap
            checkpoint = dict(state=model.state_dict(), views=views, height=args.height,
                              epoch=epoch+1, manifest=digest, seed=args.seed,
                              pretrained=not args.random_init)
            torch.save(checkpoint, out/"best.tmp")
            (out/"best.tmp").replace(out/"best.pt")
        (out/"history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"Epoch {epoch+1}/{args.epochs}  selection AP {ap:.4f}", flush=True)
    (out/"complete.json").write_text(json.dumps(dict(epochs=args.epochs, manifest=digest,
          torch=str(torch.__version__), device=str(where), training_breasts=len(fit),
          selection_breasts=len(validation))), encoding="utf-8")
