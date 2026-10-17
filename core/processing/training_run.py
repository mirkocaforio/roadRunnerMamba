import argparse
import json
import os

import torch
from timm.data import Mixup

from core.data.dataloader import build_dataset, build_weighted_sampler
from core.processing import (
    DistributedConfig,
    ExperimentBuilder,
    ExperimentRunner,
    LossType,
    SchedulerConfig,
    SchedulerType,
    TrainingConfig,
)
from core.processing.experiment import ModelConfigFactory, WandbConfig
from core.utility import utils


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train rscd_mamba model with configurable parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model-type", type=str, default="RSCD_MAMBA", help="Type of model to train"
    )
    model_group.add_argument(
        "--img-size", type=int, default=224, help="Input image size"
    )
    model_group.add_argument(
        "--patch-size", type=int, default=4, help="Patch size for the model"
    )
    model_group.add_argument(
        "--in-chans", type=int, default=3, help="Number of input channels"
    )
    model_group.add_argument(
        "--num-classes", type=int, default=6, help="Number of output classes"
    )
    model_group.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=[2, 4, 8, 4],
        help="Depths of each RSCD layer",
    )
    model_group.add_argument(
        "--dims", type=int, nargs="+", default=64, help="Dimensions of each RSCD layer"
    )
    model_group.add_argument(
        "--d-state", type=int, default=1, help="D state for RSCD layers"
    )
    model_group.add_argument(
        "--dt-init",
        type=str,
        default="he_normal",
        help="DT initialization method for RSCD layers",
    )
    model_group.add_argument(
        "--mlp-ratio", type=float, default=4.0, help="MLP ratio for RSCD layers"
    )
    model_group.add_argument(
        "--drop-rate", type=float, default=0.0, help="Dropout rate for RSCD layers"
    )
    model_group.add_argument(
        "--attn-drop-rate",
        type=float,
        default=0.0,
        help="Attention drop rate for RSCD layers",
    )
    model_group.add_argument(
        "--drop-path-rate",
        type=float,
        default=0.3,
        help="Drop path rate for RSCD layers",
    )
    model_group.add_argument(
        "--norm-layer",
        type=str,
        default="LayerNorm",
        help="Normalization layer type for RSCD layers",
    )
    model_group.add_argument(
        "--patch-norm", action="store_true", help="Use patch normalization"
    )
    model_group.add_argument(
        "--use-checkpoint", action="store_true", help="Use checkpointing to save memory"
    )
    model_group.add_argument(
        "--use-fg-mask",
        action="store_true",
        default=True,
        help="Use foreground mask in the model",
    )
    model_group.add_argument(
        "--use-condition-branch",
        action="store_true",
        default=False,
        help="Use auxiliary condition branch in the model",
    )
    model_group.add_argument(
        "--num-conditions",
        type=int,
        default=6,
        help="Number of conditions to predict (for full RSCD: 6 + separate ice)",
    )
    model_group.add_argument(
        "--condition-hidden-dim",
        type=int,
        default=None,
        help="Hidden dimension for the condition branch (default: num_features // 4)",
    )
    model_group.add_argument(
        "--condition-dropout",
        type=float,
        default=0.3,
        help="Dropout rate for the condition branch",
    )

    # Training configuration
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument(
        "--epochs", type=int, default=300, help="Number of training epochs"
    )
    train_group.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    train_group.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )
    train_group.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay"
    )
    train_group.add_argument(
        "--patience", type=int, default=20, help="Early stopping patience"
    )
    train_group.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping",
    )
    train_group.add_argument(
        "--parallel", action="store_true", help="Enable distributed training"
    )
    train_group.add_argument(
        "--stop-on-error",
        action="store_true",
        default=True,
        help="Stop training on error",
    )
    train_group.add_argument(
        "--stop-on-batch-error",
        action="store_true",
        default=True,
        help="Stop training on batch error",
    )
    train_group.add_argument(
        "--better-init", action="store_true", help="Use better initialization"
    )
    train_group.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="Enable Automatic Mixed Precision (bfloat16) training",
    )

    # Scheduler configuration
    sched_group = parser.add_argument_group("Scheduler Configuration")
    sched_group.add_argument(
        "--force-new-min",
        action="store_true",
        help="Force new minimum learning rate in scheduler",
    )
    sched_group.add_argument(
        "--force-new-lr",
        action="store_true",
        help="Force new learning rate in scheduler and optimizer",
    )
    sched_group.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "plateau"],
        help="Scheduler type",
    )
    sched_group.add_argument(
        "--warmup-epochs", type=int, default=10, help="Number of warmup epochs"
    )
    sched_group.add_argument(
        "--warmup-lr", type=float, default=1e-6, help="Warmup learning rate"
    )
    sched_group.add_argument(
        "--min-lr", type=float, default=1e-5, help="Minimum learning rate"
    )
    sched_group.add_argument(
        "--cycle-mul",
        type=float,
        default=1.0,
        help="Cycle multiplier for cosine scheduler",
    )
    sched_group.add_argument(
        "--cycle-decay",
        type=float,
        default=0.1,
        help="Cycle decay for cosine scheduler",
    )
    sched_group.add_argument(
        "--cycle-limit", type=int, default=1, help="Cycle limit for cosine scheduler"
    )
    sched_group.add_argument(
        "--scheduler-mode",
        type=str,
        default="min",
        choices=["min", "max"],
        help="Mode for plateau scheduler",
    )
    sched_group.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.1,
        help="Factor by which to reduce learning rate",
    )
    sched_group.add_argument(
        "--scheduler-patience", type=int, default=1, help="Scheduler patience"
    )
    sched_group.add_argument(
        "--cooldown-epochs", type=int, default=10, help="Cooldown epochs"
    )
    sched_group.add_argument(
        "--noise-pct", type=float, default=0.67, help="Percentage of noise in scheduler"
    )
    sched_group.add_argument(
        "--noise-std", type=float, default=1.0, help="Standard deviation of noise"
    )

    # Dataset configuration
    parser.add_argument(
        "--data-path",
        type=str,
        default="../../../RSCD_dataset_1million_reorganized/",
        help="Path to dataset",
    )
    parser.add_argument(
        "--data-set",
        type=str,
        choices=["CIFAR", "IMNET", "INAT", "INAT19", "RSCD"],
        default="RSCD",
        help="Type of dataset",
    )
    parser.add_argument(
        "--eval-crop-ratio", default=0.875, type=float, help="Crop ratio for evaluation"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loading workers"
    )

    # Augmentation parameters
    parser.add_argument(
        "--color-jitter",
        type=float,
        default=0.3,
        metavar="PCT",
        help="Color jitter factor (default: 0.3)",
    )
    (
        parser.add_argument(
            "--aa",
            type=str,
            default="rand-m9-mstd0.5-inc1",
            metavar="NAME",
            help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)',
        ),
    )
    parser.add_argument(
        "--train-interpolation",
        type=str,
        default="bicubic",
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")',
    )

    # * Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )

    # Experiment configuration
    exp_group = parser.add_argument_group("Experiment Configuration")
    exp_group.add_argument(
        "--name", type=str, default="rscd_mamba", help="Experiment name"
    )
    exp_group.add_argument(
        "--output-dir",
        type=str,
        default="experiments",
        help="Output directory for experiments",
    )
    exp_group.add_argument(
        "--loss",
        type=str,
        default="crossentropy",
        choices=["bce", "crossentropy", "focal", "mse"],
        help="Loss function type",
    )
    exp_group.add_argument("--no-cuda", action="store_true", help="Disable CUDA")

    # Configuration file
    parser.add_argument("--config", type=str, help="Path to JSON configuration file")
    parser.add_argument(
        "--save-config", type=str, help="Save current configuration to JSON file"
    )

    # Distributed training
    dist_group = parser.add_argument_group("Distributed Training")
    dist_group.add_argument(
        "--backend", type=str, default="nccl", help="Distributed backend"
    )
    dist_group.add_argument(
        "--find-unused-parameters",
        action="store_true",
        default=True,
        help="Find unused parameters in DDP",
    )
    dist_group.add_argument(
        "--gradient-as-bucket-view",
        action="store_true",
        default=True,
        help="Use gradient as bucket view",
    )

    # Wandb configuration
    wandb_group = parser.add_argument_group("Wandb Configuration")
    wandb_group.add_argument(
        "--wandb-enabled", action="store_true", help="Enable Wandb logging"
    )
    wandb_group.add_argument("--wandb-project", type=str, help="Wandb project name")
    wandb_group.add_argument("--wandb-entity", type=str, help="Wandb entity/team name")
    wandb_group.add_argument(
        "--wandb-name",
        type=str,
        help="Wandb run name (if not provided, uses experiment name)",
    )
    wandb_group.add_argument(
        "--wandb-tags", type=str, nargs="*", help="Wandb tags for the run"
    )
    wandb_group.add_argument("--wandb-notes", type=str, help="Notes for the Wandb run")
    wandb_group.add_argument(
        "--wandb-save-code", action="store_true", help="Save code to Wandb"
    )
    wandb_group.add_argument(
        "--wandb-upload-checkpoints",
        action="store_true",
        help="Upload checkpoints as Wandb artifacts",
    )
    wandb_group.add_argument(
        "--wandb-log-gradients", action="store_true", help="Log gradients to Wandb"
    )
    wandb_group.add_argument(
        "--wandb-log-parameters",
        action="store_true",
        help="Log model parameters to Wandb",
    )
    wandb_group.add_argument(
        "--wandb-log-freq",
        type=int,
        default=1,
        help="Frequency for Wandb logging (epochs)",
    )
    wandb_group.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="Wandb mode (online, offline, disabled)",
    )
    wandb_group.add_argument(
        "--wandb-artifact-type",
        type=str,
        default="model",
        help="Artifact type for Wandb uploads",
    )

    # return parser.parse_args()
    args, unknown = parser.parse_known_args()

    # Opzionale: stampa argomenti sconosciuti per debug
    if unknown:
        print(f"Argomenti sconosciuti ignorati: {unknown}")

    return args


def load_config_from_file(config_path: str) -> dict:
    """Load configuration from JSON file"""
    with open(config_path, "r") as f:
        return json.load(f)


def save_config_to_file(args: argparse.Namespace, config_path: str):
    """Save configuration to JSON file"""
    config = vars(args)
    # Convert non-serializable objects
    config_serializable = {}
    for key, value in config.items():
        if value is not None:
            config_serializable[key] = value

    with open(config_path, "w") as f:
        json.dump(config_serializable, f, indent=2)


def get_scheduler_type(scheduler_name: str) -> str:
    """Convert scheduler name to SchedulerType enum"""
    scheduler_map = {"cosine": SchedulerType.COSINE, "plateau": SchedulerType.PLATEAU}

    return scheduler_map.get(scheduler_name.lower(), SchedulerType.COSINE)


def get_loss_type(loss_name: str) -> LossType:
    """Convert loss name to LossType enum"""
    loss_map = {
        "mse": LossType.MSE,
        "bce": LossType.BCE,
        "focal": LossType.FOCAL,
        "crossentropy": LossType.CROSSENTROPY,
    }
    return loss_map.get(loss_name.lower(), LossType.CROSSENTROPY)


def setup_distributed_training():
    """Setup distributed training environment"""
    # Check if running with torchrun
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return True
    return False


def main():
    """Main training function"""
    args = parse_arguments()

    # Load configuration from file if provided
    if args.config:
        file_config = load_config_from_file(args.config)
        # Update args with file config (command line args take precedence)
        for key, value in file_config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)

    # Save configuration if requested
    if args.save_config:
        save_config_to_file(args, args.save_config)
        print(f"Configuration saved to {args.save_config}")

    # Setup distributed training
    is_distributed = setup_distributed_training() or args.parallel

    print("=" * 60)
    print("RSCD_MAMBA TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Model Type: {args.model_type}")
    print(f"Distributed Training: {is_distributed}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 60)

    # Create datasets
    dataset_train, args.num_classes = build_dataset(
        is_train=True, is_test=False, args=args
    )
    sampler = build_weighted_sampler(dataset_train)
    dataset_val, _ = build_dataset(is_train=False, is_test=False, args=args)
    dataset_test, _ = build_dataset(is_train=False, is_test=True, args=args)

    print(f"Training samples: {len(dataset_train)}")
    print(f"Validation samples: {len(dataset_val)}")
    print(f"Test samples: {len(dataset_test)}")

    # Create model configuration
    model_config = ModelConfigFactory.create_config(
        model_type=args.model_type,
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=args.in_chans,
        num_classes=args.num_classes,
        depths=args.depths,
        dims=args.dims,
        d_state=args.d_state,
        dt_init=args.dt_init,
        mlp_ratio=args.mlp_ratio,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate,
        norm_layer=args.norm_layer,
        patch_norm=args.patch_norm,
        use_checkpoint=args.use_checkpoint,
        use_fg_mask=args.use_fg_mask,
        use_condition_branch=args.use_condition_branch,
        num_conditions=args.num_conditions,  # Numero di condizioni da predire (per RSCD completo: 6 + ice separato)
        condition_hidden_dim=args.condition_hidden_dim,  # Dimensione layer nascosto del branch condizione (default: num_features // 4)
        condition_dropout=args.condition_dropout,  # Dropout nel branch condizione
    )

    print(f"Model configuration: {model_config}")

    # Create distributed configuration
    distributed_config = DistributedConfig(
        backend=args.backend,
        find_unused_parameters=args.find_unused_parameters,
        gradient_as_bucket_view=args.gradient_as_bucket_view,
    ).auto_detect_from_env()

    # Create training configuration
    training_config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        max_grad_norm=args.max_grad_norm,
        use_distributed=is_distributed,
        stop_on_error=args.stop_on_error,
        stop_on_batch_error=args.stop_on_batch_error,
        better_init=args.better_init,
        use_amp=args.amp,
    )

    # Create scheduler configuration
    scheduler_config = SchedulerConfig(
        scheduler_type=get_scheduler_type(args.scheduler),
        num_epochs=args.epochs,
        min_lr=args.min_lr,
        warmup_lr=args.warmup_lr,
        warmup_epochs=args.warmup_epochs,
        noise_pct=args.noise_pct,
        noise_std=args.noise_std,
        cooldown_epochs=args.cooldown_epochs,
        mode=args.scheduler_mode,
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        cycle_mul=args.cycle_mul,
        cycle_decay=args.cycle_decay,
        cycle_limit=args.cycle_limit,
        force_new_lr=args.force_new_lr,
        force_new_min=args.force_new_min,
        verbose=True,
        optimizer_lr=args.learning_rate,
    )

    # Create Wandb configuration
    wandb_config = WandbConfig(
        enabled=args.wandb_enabled,
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        tags=args.wandb_tags or [],
        notes=args.wandb_notes,
        save_code=args.wandb_save_code,
        upload_checkpoints=args.wandb_upload_checkpoints,
        log_gradients=args.wandb_log_gradients,
        log_parameters=args.wandb_log_parameters,
        log_freq=args.wandb_log_freq,
        mode=args.wandb_mode,
        artifact_type=args.wandb_artifact_type,
    )

    # Build experiment
    experiment = (
        ExperimentBuilder()
        .set_name(args.name)
        .set_output_dir(args.output_dir)
        .set_datasets(
            train_dataset=dataset_train,
            sampler=sampler,
            val_dataset=dataset_val,
            test_dataset=dataset_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        .set_model_config(model_config)
        .set_training_params(training_config)
        .set_scheduler(scheduler_config)
        .set_distributed_config(distributed_config)
        .set_wandb_config(wandb_config)
        .set_loss_type(get_loss_type(args.loss))
        .set_use_cuda(not args.no_cuda)
        .enable_distributed(is_distributed)
        .build()
    )

    # Run experiment
    runner = ExperimentRunner(logger=print, verbose=True)
    experiment_results = runner.run_experiment(experiment)

    # Save experiment configuration
    experiment.save_config()

    return experiment_results


if __name__ == "__main__":
    final_results = main()
