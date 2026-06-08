"""PILOT trainer.

The training loop uses all-item ranking evaluation and selects the best model by
validation NDCG@20.
"""

import copy
import numpy as np
import torch
import torch.optim as optim

try:
    from tqdm import tqdm
except ImportError:  # graceful fallback if tqdm is unavailable
    class _NoTqdm:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, *args, **kwargs):
            pass

    def tqdm(iterable, **kwargs):
        return _NoTqdm(iterable, **kwargs)


def optimizers(model, args):
    return optim.Adam(model.parameters(), lr=args.lr)


def _amp_config(args):
    use_amp = bool(int(getattr(args, 'use_amp', 0))) and str(args.device).startswith('cuda') and torch.cuda.is_available()
    if getattr(args, 'amp_dtype', 'bf16') == 'fp16':
        return use_amp, torch.float16, use_amp
    return use_amp, torch.bfloat16, False


def dcg(hit):
    log2 = torch.log2(torch.arange(1, hit.size()[-1] + 1) + 1).unsqueeze(0)
    return (hit / log2).sum(dim=-1)


def cal_hr(label, predict, ks):
    _, topk = torch.topk(predict, k=max(ks), dim=-1)
    hit = label == topk
    return [hit[:, :ks[i]].sum().item() / label.size()[0] for i in range(len(ks))]


def cal_ndcg(label, predict, ks):
    _, topk = torch.topk(predict, k=max(ks), dim=-1)
    hit = (label == topk).int()
    ndcg = []
    for k in ks:
        max_dcg = dcg(torch.tensor([1] + [0] * (k - 1)))
        predict_dcg = dcg(hit[:, :k])
        ndcg.append((predict_dcg / max_dcg).mean().item())
    return ndcg


def hrs_and_ndcgs_k(scores, labels, ks):
    metrics = {}
    ndcg = cal_ndcg(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    hr = cal_hr(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    for k, n, h in zip(ks, ndcg, hr):
        metrics[f'HR@{k}'] = h
        metrics[f'NDCG@{k}'] = n
    return metrics


@torch.no_grad()
def evaluate(model, data_loader, metric_ks, device):
    model.eval()
    acc = {f'HR@{k}': [] for k in metric_ks}
    acc.update({f'NDCG@{k}': [] for k in metric_ks})
    for batch in data_loader:
        batch = [x.to(device) for x in batch]
        rep = model(batch[0], batch[1], train_flag=False)
        scores = model.score(rep)
        metrics = hrs_and_ndcgs_k(scores, batch[1], metric_ks)
        for k, v in metrics.items():
            acc[k].append(v)
    return {k: round(float(np.mean(v)) * 100, 4) for k, v in acc.items()}


def model_train(tra_loader, val_loader, test_loader, model, args, logger):
    device = args.device
    metric_ks = args.metric_ks
    model = model.to(device)
    optimizer = optimizers(model, args)
    use_amp, amp_dtype, use_scaler = _amp_config(args)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    # Optional weight averaging.  It is disabled by default and can be enabled
    # explicitly with --ema_decay.
    ema_decay = float(getattr(args, 'ema_decay', 0.0))
    use_ema = ema_decay > 0
    ema_model = copy.deepcopy(model) if use_ema else model
    if use_ema:
        for p in ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def ema_update():
        for ep, mp in zip(ema_model.parameters(), model.parameters()):
            ep.mul_(ema_decay).add_(mp.detach(), alpha=1.0 - ema_decay)
        for eb, mb in zip(ema_model.buffers(), model.buffers()):
            eb.copy_(mb)

    best_ndcg20 = -1.0
    best_epoch = -1
    best_state = copy.deepcopy(ema_model.state_dict())
    bad_count = 0
    warmup_eval = int(getattr(args, 'ema_warmup_epochs', 0))

    # Optional smoothing for validation selection. It is disabled by default.
    sel_beta = float(getattr(args, 'sel_smooth_beta', 0.0))
    smooth_sel = None

    msg = (f"[PILOT] strict NDCG@20 selection "
           f"| optimizer=Adam lr={args.lr} "
           f"| AMP={'on' if use_amp else 'off'}({args.amp_dtype}) "
           f"| prototypes={args.num_prototypes if int(args.use_prototype) else 'off'} "
           f"| cond=FiLM+token "
           f"| EMA={'on(%.3f)' % ema_decay if use_ema else 'off'}")
    print(msg); logger.info(msg)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        last_parts = {}
        pbar = tqdm(tra_loader, desc=f"Epoch {epoch}", ncols=120, leave=True)
        for batch in pbar:
            batch = [x.to(device) for x in batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                _, loss, parts = model(batch[0], batch[1], train_flag=True)
            if use_scaler:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
            if use_ema:
                ema_update()
            running += float(loss.detach().cpu())
            last_parts = parts
            pbar.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}",
                             flow=f"{float(last_parts['flow']):.3f}")
        avg = running / max(1, len(tra_loader))
        logger.info(f"Epoch {epoch} avg_loss={avg:.4f} parts={{flow:{float(last_parts.get('flow',0)):.3f}, "
                    f"ce:{float(last_parts.get('ce',0)):.3f}, "
                    f"prior_ce:{float(last_parts.get('prior_ce',0)):.3f}}}")

        if epoch % args.eval_interval == 0:
            eval_model = ema_model if use_ema else model
            val = evaluate(eval_model, val_loader, metric_ks, device)
            sel_raw = val['NDCG@20']
            if sel_beta > 0:
                smooth_sel = sel_raw if smooth_sel is None else sel_beta * smooth_sel + (1.0 - sel_beta) * sel_raw
                sel = smooth_sel
            else:
                sel = sel_raw
            valid_label = 'valid(EMA)' if use_ema else 'valid'
            print(f"  {valid_label}: {val} | NDCG@20={sel_raw:.4f} sel={sel:.4f} (best {best_ndcg20:.4f} @ {best_epoch})")
            logger.info(f"valid epoch {epoch}: {val}")
            improved = sel > best_ndcg20 and epoch >= warmup_eval
            if improved:
                best_ndcg20 = sel; best_epoch = epoch
                best_state = copy.deepcopy(eval_model.state_dict())
                bad_count = 0
                print(f"  >> new best NDCG@20={sel:.4f} @ epoch {epoch}")
                logger.info(f"new best NDCG@20={sel:.4f} @ epoch {epoch}")
            else:
                bad_count += 1
                if bad_count >= args.patience:
                    print(f"Early stop @ epoch {epoch}; best NDCG@20={best_ndcg20:.4f} @ {best_epoch}")
                    logger.info(f"Early stop @ epoch {epoch}; best NDCG@20={best_ndcg20:.4f} @ {best_epoch}")
                    break

    eval_model = ema_model if use_ema else model
    eval_model.load_state_dict(best_state)
    test = evaluate(eval_model, test_loader, metric_ks, device)
    print('Test------------------------------------------------------')
    print(test)
    print(f"(selected by valid NDCG@20={best_ndcg20:.4f} @ epoch {best_epoch})")
    logger.info('Test------------------------------------------------------')
    logger.info(test)
    logger.info(f"selected_epoch={best_epoch} valid_NDCG@20={best_ndcg20:.4f}")
    selection = {'selected_epoch': best_epoch, 'valid_NDCG@20': best_ndcg20}
    return eval_model, test, selection
