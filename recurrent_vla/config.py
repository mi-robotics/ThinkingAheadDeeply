"""Training-config schema (dataclasses), YAML loading with --key=value overrides, and the flat legacy config the
training loop reads."""
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class ActionHeadType(str, Enum):
    # HVLA: S1 = DINOv2 ViT-S over the frames (+ optional image / proprio history, or the mem-hvla
    # double encoder) + proprio + the S2 latent (mean-pooled VLM prefix: image tokens and the
    # instruction's text tokens); the VLM leaves the per-step path. Two decoders over that context:
    HVLA = "hvla"              # recurrent-depth decoder (prelude -> looped core -> coda)
    HVLA_FLOW = "hvla_flow"    # flow-matching decoder (one NFE costs about one recurrent iteration)


@dataclass
class ModelConfig:
    # Paths are relative to the repo root (scripts/train.sh runs from there). pretrained_models/ is a local link.
    vlm_path: str = "pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b"
    config_path: str = "pretrained_models/configs"     # HF config.json + tokenizer / processor files
    num_images_in_input: int = 2
    use_proprio: bool = True
    use_lora: bool = True
    lora_rank: int = 64
    lora_dropout: float = 0.0


@dataclass
class RecurrentConfig:
    hidden_dim: int = 896
    num_heads: int = 8
    prelude_vlm_layers: List[int] = field(default_factory=list)
    recurrent_vlm_layers: List[int] = field(default_factory=lambda: [6, 23])
    coda_vlm_layers: List[int] = field(default_factory=list)
    action_chunk_len: int = 8
    action_dim: int = 7
    mean_recurrence: int = 12
    backprop_depth: int = 8
    random_iterations: bool = True
    init_std: float = 0.632
    rms_norm_eps: float = 1e-6
    rope_base: float = 10000.0
    prelude_out_norm: bool = False   # RMSNorm prelude output before the adapter (off = authors' architecture)
    deep_supervision: bool = False   # train every gradient-tracked iterate (finetune.py weights them 1...n)
    state_init_std: float = 0.0      # 0 = use init_std; >0 scales ONLY the S_0 noise, not the action queries

    # --- HVLA S1 options ---
    # Depth of the observation encoder that fuses [vision tokens; S2 token; proprio] into ONE
    # context, computed once per forward and reused by every decoder layer / iteration.
    s1_encoder_layers: int = 4
    # The VLM layer whose prefix is mean-pooled into the S2 latent.
    s1_context_layer: int = 23

    # S1's own fast vision encoder. ViT-S/14 does both cameras in 1.61 ms against 4.43 ms
    # for the VLM's in-model DINOv2 ViT-L and 14.41 ms for the full fused backbone.
    s1_vit_model: str = "vit_small_patch14_dinov2.lvd142m"
    s1_vit_pretrained: bool = True
    s1_num_images: int = 2
    # How many S1 steps reuse one S2 latent at inference. 1 = refresh every step (no split);
    # the point of the split is that this can be 8 or more.
    s1_s2_refresh_every: int = 8
    # Keep S1's own DINOv2 tower at stock LVD-142M. MUST exist here as well as on
    # RecurrentConfigInternal: finetune.py copies these fields under `if hasattr(rec_cfg, f)`,
    # so a key that this dataclass does not declare is dropped from the YAML in silence.
    s1_freeze_vit: bool = False

    # --- flow decoder (type = hvla_flow) ---
    # Untied decoder layers. 4 = prelude 1 + core 2 + coda 1 of the recurrent head, so one
    # denoising step (NFE) costs about one recurrent iteration and the depth axes align.
    flow_num_layers: int = 4
    # Euler steps at inference when num_iter is not given.
    flow_num_inference_steps: int = 10


@dataclass
class ActionHeadConfig:
    type: ActionHeadType = ActionHeadType.HVLA
    recurrent: RecurrentConfig = field(default_factory=RecurrentConfig)


@dataclass
class OptimizerConfig:
    learning_rate: float = 2e-4
    lr_warmup_steps: int = 0
    num_steps_before_decay: int = 200000


@dataclass
class DataConfig:
    data_root_dir: str = "datasets/libero"             # datasets/ is a local link to the RLDS data
    dataset_name: str = "libero_spatial_no_noops"
    dataset_fraction: float = 1.0
    shuffle_buffer_size: int = 100_000
    image_aug: bool = True


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    action_head: ActionHeadConfig = field(default_factory=ActionHeadConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)

    batch_size: int = 16
    grad_accumulation_steps: int = 2
    max_steps: int = 200000
    save_freq: int = 5000
    save_latest_only: bool = False
    merge_lora: bool = True
    use_val_set: bool = False
    val_freq: int = 10_000
    val_time_limit: int = 180

    output_dir: str = "outputs"
    run_name: Optional[str] = None

    use_wandb: bool = True
    wandb_project: str = "rd-vla"
    wandb_entity: Optional[str] = None
    wandb_log_freq: int = 10

    resume: bool = False
    resume_path: Optional[str] = None
    resume_step: Optional[int] = None

    seed: int = 42


# ============================== YAML loading ==============================
import yaml
from pathlib import Path
from dataclasses import fields, is_dataclass
from typing import TypeVar, Type, Any, Dict, get_type_hints, get_origin
from enum import Enum



T = TypeVar('T')


def deep_update(base: Dict, updates: Dict) -> Dict:
    """Recursively update nested dict."""
    result = base.copy()
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _convert_value(value: Any, target_type: type) -> Any:
    """Convert value to target type, handling string representations of primitives."""
    if value is None:
        return None
    if isinstance(value, target_type):
        return value
    if target_type == float and isinstance(value, (int, str)):
        return float(value)
    if target_type == int and isinstance(value, (float, str)):
        return int(value)
    if target_type == bool and isinstance(value, str):
        return value.lower() in ('true', '1', 'yes')
    if target_type == str:
        return str(value)
    return value


def dict_to_dataclass(cls: Type[T], data: Dict[str, Any]) -> T:
    """Convert nested dict to nested dataclass."""
    if not is_dataclass(cls):
        if isinstance(data, cls):
            return data
        if issubclass(cls, Enum) and isinstance(data, str):
            return cls(data)
        return data

    hints = get_type_hints(cls)
    kwargs = {}

    for f in fields(cls):
        if f.name not in data:
            continue

        field_type = hints[f.name]
        value = data[f.name]

        origin = get_origin(field_type)
        if origin is not None:
            if value is None:
                kwargs[f.name] = None
            else:
                kwargs[f.name] = value
        elif is_dataclass(field_type):
            if isinstance(value, dict):
                kwargs[f.name] = dict_to_dataclass(field_type, value)
            else:
                kwargs[f.name] = value
        elif isinstance(field_type, type) and issubclass(field_type, Enum):
            kwargs[f.name] = field_type(value) if isinstance(value, str) else value
        elif isinstance(field_type, type) and field_type in (float, int, bool, str):
            kwargs[f.name] = _convert_value(value, field_type)
        else:
            kwargs[f.name] = value

    return cls(**kwargs)


def dataclass_to_dict(obj) -> Dict:
    """Convert dataclass to dict, handling enums."""
    if is_dataclass(obj):
        result = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            result[f.name] = dataclass_to_dict(value)
        return result
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, list):
        return [dataclass_to_dict(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj


def load_yaml_config(path: str) -> Dict:
    """Load YAML config file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def parse_cli_overrides(args: list) -> Dict:
    """Parse CLI args like --model.lora_rank=32 into nested dict."""
    overrides = {}
    for arg in args:
        if not arg.startswith('--'):
            continue
        if '=' in arg:
            key, value = arg[2:].split('=', 1)
        else:
            continue

        try:
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif '.' in value and value.replace('.', '').replace('-', '').isdigit():
                value = float(value)
            elif value.lstrip('-').isdigit():
                value = int(value)
        except:
            pass

        parts = key.split('.')
        current = overrides
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    return overrides


def load_config(config_class: Type[T], yaml_path: str = None, cli_args: list = None) -> T:
    """Load config from YAML with CLI overrides."""
    base = dataclass_to_dict(config_class())

    if yaml_path:
        yaml_config = load_yaml_config(yaml_path)
        base = deep_update(base, yaml_config)

    if cli_args:
        overrides = parse_cli_overrides(cli_args)
        base = deep_update(base, overrides)

    return dict_to_dataclass(config_class, base)


def save_config(config, path: str):
    """Save config to YAML file."""
    data = dataclass_to_dict(config)
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def get_legacy_config(cfg, mode: str = 'train'):
    """Convert the TrainConfig to the flat legacy FinetuneConfig that finetune.py reads."""
    if mode != 'train':
        raise ValueError("only training configs exist; evaluate with experiments/robot/libero/run_libero_eval.py")
    return _convert_train_config(cfg)


def _convert_train_config(cfg: TrainConfig):
    """Convert TrainConfig to legacy FinetuneConfig format."""
    from dataclasses import dataclass as dc

    @dc
    class LegacyConfig:
        pass

    legacy = LegacyConfig()

    legacy.config_file_path = cfg.model.config_path
    legacy.vlm_path = cfg.model.vlm_path
    legacy.resum_vla_path = cfg.resume_path or cfg.model.vlm_path

    legacy.data_root_dir = Path(cfg.data.data_root_dir)
    legacy.dataset_name = cfg.data.dataset_name
    legacy.dataset_fraction = cfg.data.dataset_fraction
    legacy.run_root_dir = Path(cfg.output_dir)
    legacy.shuffle_buffer_size = cfg.data.shuffle_buffer_size

    legacy.num_images_in_input = cfg.model.num_images_in_input
    legacy.use_proprio = cfg.model.use_proprio
    legacy.phase1_path = "None"

    legacy.batch_size = cfg.batch_size
    legacy.learning_rate = cfg.optimizer.learning_rate
    legacy.lr_warmup_steps = cfg.optimizer.lr_warmup_steps
    legacy.num_steps_before_decay = cfg.optimizer.num_steps_before_decay
    legacy.grad_accumulation_steps = cfg.grad_accumulation_steps
    legacy.max_steps = cfg.max_steps
    legacy.use_val_set = cfg.use_val_set
    legacy.val_freq = cfg.val_freq
    legacy.val_time_limit = cfg.val_time_limit
    legacy.save_freq = cfg.save_freq
    legacy.save_latest_checkpoint_only = cfg.save_latest_only
    legacy.resume = cfg.resume
    legacy.resume_step = cfg.resume_step
    legacy.image_aug = cfg.data.image_aug
    legacy.diffusion_sample_freq = 50

    legacy.use_lora = cfg.model.use_lora
    legacy.lora_rank = cfg.model.lora_rank
    legacy.lora_dropout = cfg.model.lora_dropout
    legacy.merge_lora_during_training = cfg.merge_lora

    legacy.use_wandb = cfg.use_wandb
    legacy.wandb_entity = cfg.wandb_entity
    legacy.wandb_project = cfg.wandb_project
    legacy.run_id_note = cfg.run_name
    legacy.run_id_override = None
    legacy.wandb_log_freq = cfg.wandb_log_freq

    legacy.phase = "Training"

    legacy.use_recurrent = False
    legacy.recurrent_cfg = None

    legacy.action_head = cfg.action_head

    # Training seed. This was defined on TrainConfig but never forwarded -- only the EVAL
    # converter set legacy.seed -- so every training run so far has been effectively
    # unseeded. The final runs pin seed 42, which requires this line.
    legacy.seed = cfg.seed


    return legacy
