"""Training and evaluation utilities for PILOT."""

import copy

import numpy as np
import torch
import torch.optim as optim

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def optimizers(model, args):
    return optim.Adam(model.parameters(), lr=args.lr)


def dcg(hit):
    log2 = torch.log2(torch.arange(1, hit.size()[-1] + 1) + 1).unsqueeze(0)
    return (hit / log2).sum(dim=-1)


def cal_hr(label, predict, ks):
    _, topk = torch.topk(predict, k=max(ks), dim=-1)
    hit = label == topk
    return [hit[:, :k].sum().item() / label.size()[0] for k in ks]


def cal_ndcg(label, predict, ks):
    _, topk = torch.topk(predict, k=max(ks), dim=-1)
    hit = (label == topk).int()
    values = []
    for k in ks:
        max_dcg = dcg(torch.tensor([1] + [0] * (k - 1)))
        predict_dcg = dcg(hit[:, :k])
        values.append((predict_dcg / max_dcg).mean().item())
    return values


def hrs_and_ndcgs_k(scores, labels, ks):
    metrics = {}
    scores = scores.clone().detach().to('cpu')
    labels = labels.clone().detach().to('cpu')
    hr = cal_hr(labels, scores, ks)
    ndcg = cal_ndcg(labels, scores, ks)
    for k, h, n in zip(ks, hr, ndcg):
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
        for key, value in metrics.items():
            acc[key].append(value)
    return {key: round(float(np.mean(value)) * 100, 4) for key, value in acc.items()}


def model_train(train_loader, val_loader, test_loader, model, args, logger):
    device = args.device
    metric_ks = args.metric_ks
    model = model.to(device)
    optimizer = optimizers(model, args)

    best_ndcg20 = -1.0
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    bad_count = 0

    msg = (f'[PILOT] select by Valid NDCG@20 | optimizer=Adam lr={args.lr} '
           f'| prototypes={args.num_prototypes if int(args.use_prototype) else "off"}')
    print(msg)
    logger.info(msg)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        last_parts = {'flow': 0.0, 'ce': 0.0, 'prior_ce': 0.0}
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', ncols=120, leave=True)
        for batch in pbar:
            batch = [x.to(device) for x in batch]
            optimizer.zero_grad(set_to_none=True)
            _, loss, parts = model(batch[0], batch[1], train_flag=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu())
            last_parts = parts
            if hasattr(pbar, 'set_postfix'):
                pbar.set_postfix(loss=f'{float(loss.detach().cpu()):.4f}', flow=f'{float(parts["flow"]):.3f}')

        avg_loss = running / max(1, len(train_loader))
        logger.info(
            f'Epoch {epoch} avg_loss={avg_loss:.4f} '
            f'parts={{flow:{float(last_parts.get("flow", 0)):.3f}, '
            f'ce:{float(last_parts.get("ce", 0)):.3f}, '
            f'prior_ce:{float(last_parts.get("prior_ce", 0)):.3f}}}'
        )

        if epoch % args.eval_interval == 0:
            val = evaluate(model, val_loader, metric_ks, device)
            current = val['NDCG@20']
            print(f'  valid: {val} | best NDCG@20={best_ndcg20:.4f} @ {best_epoch}')
            logger.info(f'valid epoch {epoch}: {val}')
            if current > best_ndcg20:
                best_ndcg20 = current
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                bad_count = 0
                print(f'  >> new best NDCG@20={best_ndcg20:.4f} @ epoch {epoch}')
                logger.info(f'new best NDCG@20={best_ndcg20:.4f} @ epoch {epoch}')
            else:
                bad_count += 1
                if bad_count >= args.patience:
                    print(f'Early stop @ epoch {epoch}; best NDCG@20={best_ndcg20:.4f} @ {best_epoch}')
                    logger.info(f'Early stop @ epoch {epoch}; best NDCG@20={best_ndcg20:.4f} @ {best_epoch}')
                    break

    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, metric_ks, device)
    print('Test------------------------------------------------------')
    print(test)
    print(f'(selected by valid NDCG@20={best_ndcg20:.4f} @ epoch {best_epoch})')
    logger.info('Test------------------------------------------------------')
    logger.info(test)
    logger.info(f'selected_epoch={best_epoch} valid_NDCG@20={best_ndcg20:.4f}')
    selection = {'selected_epoch': best_epoch, 'valid_NDCG@20': best_ndcg20}
    return model, test, selection
