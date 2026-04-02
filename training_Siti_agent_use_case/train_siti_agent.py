"""
Training script per l'agente di Siti con VERL algoritmo.
Il training avverà su un unico modello attraverso LoRA modules. Anche se gli agenti in verità sono 2.
Example usage:
```bash
python train_siti_agent.py --train-file data/train.parquet --val-file data/test.parquet --llm-proxy --lora
```

Per utilizzare uno store esterno:
in un terminale 
```bash
agl store --port 7474 
```
e poi:
```bash
python train_siti_agent.py --train-file data/train.parquet --val-file data/test.parquet --llm-proxy --lora --external-store-address http://localhost:7474
```

Per rendere più semplice la obervability e il monitoring si può separare anche l'esecuzione del runner e dell'algoritmo:
In due terminali diversi:
```bash
AGL_MANAGED_STORE=0 AGL_CURRENT_ROLE=algorithm python train_siti_agent.py --train-file data/train.parquet --val-file data/test.parquet --llm-proxy --lora --external-store-address http://localhost:7474
AGL_MANAGED_STORE=0 AGL_CURRENT_ROLE=runner python train_siti_agent.py --train-file data/train.parquet --val-file data/test.parquet --llm-proxy --lora --external-store-address http://localhost:7474
```

Nel nostro caso d'uso possiamo utilizzare un aggregazione degli span che provengono dal tracer chiamata trajectory level aggregation. Maggiori informazioni in: https://agent-lightning.github.io/posts/trajectory_level_aggregation/

Per attivarla basta:
```bash
python train_siti_agent.py --train-file data/train.parquet --val-file data/test.parquet --llm-proxy --lora --trajectory-level
```
"""

import argparse
import pyarrow.parquet as pq
from datetime import datetime
import uuid
from typing import Any, Dict, Optional, cast, List
from siti_agent import LitSitiAgent
import agentlightning as agl
from agentlightning.env_var import LightningEnvVar, resolve_bool_env_var, resolve_str_env_var
from utils.logger import Logger
import os


logger = Logger(save=False, consoleLevel="DEBUG").getLogger()

def verl_default_config() -> Dict[str,Any]:
    config = {
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False
        },
        "data": {
            "train_batch_size": 16,
            "max_prompt_length": 4096,
            "max_response_length": 2048
        },
        "actor_rollout_ref": {
            "rollout": {
                "tensor_model_parallel_size": 1,
                "n": 4,
                "log_prob_micro_batch_size_per_gpu": 4,
                "multi_turn": {"format": "hermes"},
                "name": "vllm",
                "gpu_memory_utilization": 0.6,
                "engine_kwargs": {
                    "vllm": {
                        "enable_auto_tool_choice": True,
                        "tool_call_parser": "hermes"
                    }
                },
            },
            "actor": {
                "ppo_mini_batch_size":16,
                "ppo_micro_batch_size_per_gpu": 2,
                "optim": {"lr": 1e-6},
                "use_kl_loss": False,
                "kl_loss_coef": 0,
                "entropy_coeff": 0,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.3,
                "fsdp_config": {
                    "param_offload": True,
                    "optimizer_offload": True
                }
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 4,
                "fsdp_config": {"param_offload": True}
            },
            "model": {
                "path": "unsloth/granite-4.0-micro-unsloth-bnb-4bit",
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True
            }
        },
        "trainer": {
            "n_gpus_per_node": 1,
            "val_before_train": True,
            "critic_warmup": 0,
            "logger": ["console","wandb"],
            "project_name": "SitiBTAgent",
            "experiment_name": "whatsapp_agent",
            "nnodes": 1,
            "save_freq": 32,
            "test_freq": 16,
            "total_epochs": 2
        }
    }
    return config

def train(
        *,
        train_file: str,
        val_file: str,
        model: Optional[str],
        llm_proxy:bool,
        ci:bool,
        ci_fast:bool,
        n_runners:int,
        external_store_address:str,
        lora:bool,
        lora_rank:int,
        lora_adapter_path:Optional[str],
        trajectory_level:bool=True,
        weave:bool,
        mongo_uri:Optional[str]
):
    """The training entrypoint function for Siti agent with VERL algorithm.

    Args:
        train_file: The path to the training parquet file.
        val_file: The path to the validation parquet file.
        model: The HF model id or path to override the default model.
        llm_proxy: Whether to enable LLM Proxy tracing/adapter.
        ci: Whether to run a minimal CI-style training loop.
        n_runners: The number of runners for the Trainer.
        ci_fast: Whether to cap the training loop at a single step (implies CI toggles).
        external_store_address: Connects to an external store instead of creating a new one in memory.
        lora: Whether to enable LoRA training.
        lora_rank: LoRA rank to use when LoRA is enabled.
        lora_adapter_path: Optional path to a pre-trained LoRA adapter to load.
        trajectory_level: Whether to enable trajectory level in trace aggregator.
        weave: Whether to enable Weave tracing.
        mongo_uri: MongoDB URI to use for the store.
    """

    # load dataset
    train_table = pq.read_table(train_file)
    train_dataset = cast(List[Dict[str, Any]], train_table.to_pylist()) 
    val_table = pq.read_table(val_file)
    val_dataset = cast(List[Dict[str, Any]], val_table.to_pylist())
    logger.info(f"first 1 row of train dataset: {train_dataset[0]}")
    logger.info(f"first 1 row of validation dataset: {val_dataset[0]}")

    # add verl config
    config = verl_default_config()

    if model:
        config["actor_rollout_ref"]["model"]["path"] = model
    
    # enable lora
    if lora:
        config["actor_rollout_ref"]["model"]["lora_rank"] = lora_rank
        logger.info(f"LoRA enable: lora_rank={lora_rank}")

        if lora_adapter_path:
            config["actor_rollout_ref"]["model"]["lora_adapter_path"] = lora_adapter_path
            logger.info(f"Loading LoRA adapter from: {lora_adapter_path}")
        logger.info("LoRA configuration will trigger verl to set ref_in_actor=True (LoRA mode)")

    # add trajectory level aggregation
    if trajectory_level:
        config["agentlightning"] = {
            "trace_aggregator": {
                "level": "trajectory",
                "trajectory_max_prompt_length": 2048,
                "trajectory_max_response_length": 8192
            }
        }
        logger.info("Trajectory level enabled in trace aggregator")

    if ci or ci_fast:
       # Config the experiment name and project name so that they are available to CI
       timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
       random_suffix = uuid.uuid4().hex[:8]
       EXPERIMENT_NAME = f"siti_agent_{timestamp}_{random_suffix}"
       PROJECT_NAME = "AgentLightningCI"
       # Skip this step if AGL_CURRENT_ROLE is runner
       agl_current_role = resolve_str_env_var(LightningEnvVar.AGL_CURRENT_ROLE)
       if agl_current_role != "runner":
           # Simulate writing to $GITHUB_OUTPUT if it’s set
           github_output = os.getenv("GITHUB_OUTPUT")
           if github_output:
               with open(github_output, "a") as f:
                   f.write(f"project_name={PROJECT_NAME}\n")
                   f.write(f"run_name={EXPERIMENT_NAME}\n")
           print("Set environment variables:")
           print(f"PROJECT_NAME={PROJECT_NAME}")
           print(f"EXPERIMENT_NAME={EXPERIMENT_NAME}")
       # Keep it tiny/light without adding new knobs
       config["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.8
       config["trainer"]["total_epochs"] = 1
       config["trainer"]["total_training_steps"] = 20
       config["trainer"]["test_freq"] = 20
       config["trainer"]["experiment_name"] = EXPERIMENT_NAME
       config["trainer"]["project_name"] = PROJECT_NAME
       config["trainer"].pop("save_freq", None)
       if ci_fast:
           # Extra fast CI toggle for testing purposes.
           config["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.6
           config["trainer"]["total_training_steps"] = 1
           config["trainer"]["test_freq"] = 1
    
    # add algo
    algo = agl.VERL(config)

    # add external store
    if external_store_address:
        store: Optional[agl.LightningStore] = agl.LightningStoreClient(external_store_address)
        logger.info(f"Connect to external Ligthning store at: {external_store_address}")
    elif mongo_uri:
        from agentlightning.store.mongo import MongoLightningStore

        store = MongoLightningStore(mongo_uri=mongo_uri)
    else:
        store = None

    # add proxy LLM
    if llm_proxy:
        tracer = agl.OtelTracer() # dummy tracer
        adapter = agl.LlmProxyTraceToTriplet()
        trainer = agl.Trainer(algorithm=algo, n_runners=n_runners, store=store, tracer=tracer, adapter=adapter)
    elif weave:
        from agentlightning.tracer.weave import WeaveTracer
        tracer = WeaveTracer()
        trainer = agl.Trainer(algorithm=algo, n_runners=n_runners, store=store, tracer=tracer)
    else:
        trainer = agl.Trainer(algorithm=algo, n_runners=n_runners, store=store)

    # start the training
    trainer.fit(LitSitiAgent, train_dataset, val_dataset=val_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a tool/calling agent with Agent-lightning + VERL.")
    parser.add_argument("--train-file", type=str, default="data/train.parquet", help="Path to train parquet file")
    parser.add_argument("--val-file", type=str, default="data/test.parquet", help="Path to val parquet file")
    parser.add_argument("--model", type=str, default=None, help="HF model id or path (optional)")
    parser.add_argument("--llm-proxy", action="store_true", help="Enable LLM Proxy tracing/adapter")
    parser.add_argument("--weave", action="store_true", help="Enable Weave tracing")
    parser.add_argument("--ci", action="store_true", help="Run a minimal CI-style training loop")
    parser.add_argument(
        "--ci-fast", action="store_true", help="Limit the training loop to a single step (implies --ci)"
    )
    parser.add_argument("--n-runners", type=int, default=10, help="Number of runners for Trainer")
    parser.add_argument(
        "--external-store-address",
        type=str,
        default="",
        help="Connect to an external store instead of creating a new one in memory",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Enable LoRA training. When enabled, the reference policy is computed by the actor rollout worker.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA rank to use when --lora is enabled (default: 32)",
    )
    parser.add_argument(
        "--lora-adapter-path",
        type=str,
        default=None,
        help="Optional path to a pre-trained LoRA adapter to load when --lora is enabled",
    )
    parser.add_argument(
        "--trajectory-level",
        action="store_true",
        help="Enable trajectory level in trace aggregator.",
    )
    parser.add_argument(
        "--mongo-uri",
        type=str,
        default=None,
        help="MongoDB URI to use for the store.",
    )

    args = parser.parse_args()

    if args.external_store_address:
        print(f"Connecting to external store at: {args.external_store_address}")
        if resolve_bool_env_var(LightningEnvVar.AGL_MANAGED_STORE, fallback=True):
            raise ValueError(
                "When using an external store, please set the environment variable AGL_MANAGED_STORE=0. "
                "Otherwise the trainer will still try to manage the store lifecycle for you!"
            )

    if args.ci_fast:
        args.ci = True

    



    train(
        train_file=args.train_file,
        val_file=args.val_file,
        model=args.model,
        llm_proxy=args.llm_proxy,
        ci=args.ci,
        ci_fast=args.ci_fast,
        n_runners=args.n_runners,
        external_store_address=args.external_store_address,
        lora=args.lora,
        lora_rank=args.lora_rank,
        lora_adapter_path=args.lora_adapter_path,
        trajectory_level=args.trajectory_level,
        weave=args.weave,
        mongo_uri=args.mongo_uri,
    )

    agl.setup_logging("DEBUG" if args.debug else "INFO") 