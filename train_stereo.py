import os
import hydra
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from hydra.utils import instantiate
from accelerate.utils import set_seed, DistributedDataParallelKwargs
from accelerate.logging import get_logger

from model import fetch_model
from dataset import fetch_dataloader
from util.loss import sequence_loss
from util.padder import InputPadder


def freeze_bn_for_model(accelerator, model):
    accelerator.unwrap_model(model).freeze_bn()


def _is_valid_path(path):
    if path is None:
        return False
    path = str(path).strip()
    return path.lower() not in ["", "none", "null"]


def _strip_state_prefix(key):
    """Strip common wrappers from checkpoint keys."""
    for prefix in ("module.", "_orig_mod.", "model."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def _load_checkpoint_file(checkpoint_path, logger):
    """Load a PyTorch/safetensors checkpoint from a file or directory."""
    checkpoint_path = str(checkpoint_path)

    if os.path.isdir(checkpoint_path):
        candidates = [
            "model.safetensors",
            "pytorch_model.bin",
            "model.bin",
            "model.pt",
            "model.pth",
            "checkpoint.pth",
            "checkpoint.pt",
        ]
        ckpt_file = None
        for name in candidates:
            path = os.path.join(checkpoint_path, name)
            if os.path.exists(path):
                ckpt_file = path
                break
        if ckpt_file is None:
            # If the user passes a directory containing a single safetensors / pt / pth file
            # with a custom name, accept it.  This keeps Windows quick-validation convenient
            # for files such as sceneflow_192.safetensors.
            import glob
            wildcard_files = []
            for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth"):
                wildcard_files.extend(glob.glob(os.path.join(checkpoint_path, pattern)))
            if len(wildcard_files) == 1:
                ckpt_file = wildcard_files[0]
            else:
                raise FileNotFoundError(
                    f"No model checkpoint file found in directory: {checkpoint_path}. "
                    f"Expected one of: {candidates}, or a directory containing exactly one "
                    f"*.safetensors / *.bin / *.pt / *.pth file. Found: {wildcard_files}"
                )
    else:
        ckpt_file = checkpoint_path
        if not os.path.exists(ckpt_file):
            raise FileNotFoundError(f"Checkpoint file does not exist: {ckpt_file}")

    logger.info(f"Loading checkpoint file: {ckpt_file}")

    if ckpt_file.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except Exception as e:
            raise ImportError(
                "The checkpoint is a .safetensors file, but safetensors could not be imported. "
                "Install it with: pip install safetensors"
            ) from e
        state = load_file(ckpt_file)
    else:
        state = torch.load(ckpt_file, map_location="cpu")

    if isinstance(state, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break

    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint format from {ckpt_file}: {type(state)}")

    return state


def load_pretrained_partial(model, checkpoint_path, logger):
    """Partially load pretrained PromptStereo weights into RMSI-Stereo.

    This loading mode keeps all shape-compatible baseline parameters:
    - old PromptStereo weights with identical names/shapes are loaded;
    - new RMSI-Stereo MSCA/CVUM parameters are left initialized by the model;
    - incompatible or missing keys are skipped and logged.
    """
    logger.info(f"Partially loading pretrained weights from {checkpoint_path}")
    raw_state = _load_checkpoint_file(checkpoint_path, logger)
    model_state = model.state_dict()

    filtered_state = {}
    unexpected_keys = []
    shape_mismatch = []

    for raw_key, value in raw_state.items():
        key = _strip_state_prefix(raw_key)
        if key not in model_state:
            unexpected_keys.append(raw_key)
            continue
        if not torch.is_tensor(value):
            unexpected_keys.append(raw_key)
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            shape_mismatch.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        filtered_state[key] = value

    missing_keys, incompatible_unexpected = model.load_state_dict(filtered_state, strict=False)

    logger.info(f"Partial checkpoint load finished.")
    logger.info(f"  loaded tensors: {len(filtered_state)}")
    logger.info(f"  missing keys in checkpoint/new model params: {len(missing_keys)}")
    logger.info(f"  unexpected keys skipped from checkpoint: {len(unexpected_keys) + len(incompatible_unexpected)}")
    logger.info(f"  shape-mismatch keys skipped: {len(shape_mismatch)}")

    if missing_keys:
        logger.info("First 40 missing keys. New RMSI-Stereo keys here are expected:")
        for key in list(missing_keys)[:40]:
            logger.info(f"  [missing] {key}")

    if shape_mismatch:
        logger.info("First 20 shape-mismatch keys:")
        for item in shape_mismatch[:20]:
            logger.info(f"  [shape mismatch] {item}")

    return model


@hydra.main(version_base=None, config_path='config', config_name='train_sceneflow')
def main(cfg):
    set_seed(cfg.seed)
    logger = get_logger(__name__)

    accelerator = instantiate(cfg.accelerator, _partial_=True)(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    )
    accelerator.init_trackers(
        project_name=cfg.tracker.project_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        init_kwargs=cfg.tracker.init_kwargs,
    )

    logger.info(
        f"[Device] torch.cuda.is_available()={torch.cuda.is_available()}, "
        f"accelerator.device={accelerator.device}, "
        f"mixed_precision={accelerator.mixed_precision}, "
        f"num_processes={accelerator.num_processes}"
    )

    train_loader = fetch_dataloader(cfg, cfg.train_set, cfg.train_loader, logger)
    valid_loader = fetch_dataloader(cfg, cfg.valid_set, cfg.valid_loader, logger)
    model = fetch_model(cfg, logger)
    optimizer = instantiate(cfg.optimizer, _partial_=True)(model.parameters())
    scheduler = instantiate(cfg.scheduler, _partial_=True)(optimizer)

    if _is_valid_path(cfg.get("checkpoint", None)):
        model = load_pretrained_partial(model, cfg.checkpoint, logger)
        logger.info(f"Partially loaded checkpoint from {cfg.checkpoint}.")

    train_loader, model, optimizer, scheduler = accelerator.prepare(
        train_loader, model, optimizer, scheduler
    )
    for name in valid_loader:
        valid_loader[name] = accelerator.prepare_data_loader(valid_loader[name])

    set_seed(cfg.seed, device_specific=True)

    step = 0
    should_keep_training = True
    while should_keep_training:
        model.train()
        freeze_bn_for_model(accelerator, model)

        for data in tqdm(train_loader, dynamic_ncols=True, disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                _, left, right, disp_gt, valid = [x for x in data]

                model_cfg = cfg.model.instance.cfg
                msca_cfg = model_cfg.get('msca', {})
                rapc_cfg = cfg.get('rapc', {})
                msca_enabled = bool(msca_cfg.get('enabled', True))
                rapc_enabled = bool(rapc_cfg.get('enabled', True))
                # Aux carries per-iteration S_t/d^(t+1) for RAPC and scalar
                # CVUM/MSCA diagnostics for experiment verification.
                return_aux = msca_enabled or rapc_enabled

                with accelerator.autocast():
                    model_out = model(left, right, iters=cfg.model.train_iters, return_aux=return_aux)

                if return_aux:
                    init_disp, disp_pred, aux = model_out
                else:
                    init_disp, disp_pred = model_out
                    aux = None

                loss, metric = sequence_loss(
                    init_disp,
                    disp_pred,
                    disp_gt,
                    valid,
                    cfg.max_disp,
                    aux=aux,
                    rapc_cfg=rapc_cfg,
                    current_step=step,
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.max_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            if not accelerator.sync_gradients:
                continue

            step += 1

            loss = accelerator.reduce(loss.detach(), reduction='mean')
            metric = accelerator.reduce(metric, reduction='mean')
            accelerator.log(
                {
                    'train/loss': loss,
                    'train/learning_rate': optimizer.param_groups[0]['lr'],
                },
                step,
            )
            accelerator.log(metric, step)

            if (step > 0) and (step % cfg.save_freq == 0):
                accelerator.save_state(os.path.join(cfg.save_path, str(step)))

            if (step > 0) and (step % cfg.valid_freq == 0):
                for name in valid_loader:
                    model.eval()
                    total_elem, total_epe, total_out = 0.0, 0.0, 0.0

                    for data in tqdm(valid_loader[name], dynamic_ncols=True, disable=not accelerator.is_main_process):
                        _, left, right, disp_gt, valid = [x for x in data]
                        padder = InputPadder(left.shape, divis_by=32)
                        left, right = padder.pad(left, right)

                        with torch.no_grad():
                            with accelerator.autocast():
                                disp_pred = model(left, right, iters=cfg.model.valid_iters, test_mode=True)
                            disp_pred = padder.unpad(disp_pred)

                        epe = torch.abs(disp_pred - disp_gt)
                        out = (epe > cfg.valid_set[name].outlier).float()

                        if cfg.max_disp:
                            valid = (valid >= 0.5) & (disp_gt < cfg.max_disp)
                        else:
                            valid = (valid >= 0.5)

                        finite_mask = torch.isfinite(disp_pred) & torch.isfinite(disp_gt)
                        valid_mask = valid & finite_mask
                        num_valid = valid_mask.sum()

                        if num_valid.item() == 0:
                            continue

                        epe_sum = epe[valid_mask].sum()
                        out_sum = out[valid_mask].sum()
                        valid_count = num_valid.to(dtype=torch.float32)

                        epe_sum, out_sum, valid_count = accelerator.gather_for_metrics(
                            (
                                torch.nan_to_num(epe_sum.detach()),
                                torch.nan_to_num(out_sum.detach()),
                                valid_count.detach(),
                            )
                        )

                        total_elem += valid_count.sum().item()
                        total_epe += epe_sum.sum().item()
                        total_out += out_sum.sum().item()

                    if total_elem > 0:
                        accelerator.log(
                            {
                                f'valid/{name}/EPE': total_epe / total_elem,
                                f'valid/{name}/BP-{cfg.valid_set[name].outlier}': 100.0 * total_out / total_elem,
                            },
                            step,
                        )
                    else:
                        logger.warning(
                            f'[RANK {accelerator.process_index}] No valid pixels found during validation on {name}.'
                        )

                model.train()
                freeze_bn_for_model(accelerator, model)

            if step >= cfg.scheduler.total_steps:
                should_keep_training = False
                break

    accelerator.save_model(model, os.path.join(cfg.save_path, 'final'))
    accelerator.end_training()


if __name__ == '__main__':
    main()
