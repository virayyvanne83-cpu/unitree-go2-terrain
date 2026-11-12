"""Export a standard 48->12 Go2 PPO checkpoint without starting Isaac Gym."""
import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).parent / "configs/go2.yaml"


def export_policy(checkpoint, output, activation="elu"):
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    state = payload["model_state_dict"]
    weights = {key[len("actor."):]: value for key, value in state.items() if key.startswith("actor.")}
    indices = sorted(int(key.split(".")[0]) for key in weights if key.endswith(".weight"))
    if not indices or indices != list(range(0, 2 * len(indices), 2)):
        raise ValueError("Expected a standard PPO actor with alternating Linear/activation layers")
    if weights["0.weight"].shape[1] != 48 or weights[f"{indices[-1]}.weight"].shape[0] != 12:
        raise ValueError("Expected Go2's 48-input/12-output actor; HIM and height-scan policies are incompatible")
    activations = {"elu": torch.nn.ELU, "relu": torch.nn.ReLU,
                   "tanh": torch.nn.Tanh, "selu": torch.nn.SELU,
                   "lrelu": torch.nn.LeakyReLU, "sigmoid": torch.nn.Sigmoid}
    if activation not in activations:
        raise ValueError(f"Unsupported activation: {activation}")
    layers = OrderedDict()
    for index in indices:
        weight = weights[f"{index}.weight"]
        layers[str(index)] = torch.nn.Linear(weight.shape[1], weight.shape[0])
        if index != indices[-1]:
            layers[str(index + 1)] = activations[activation]()
    actor = torch.nn.Sequential(layers).eval()
    actor.load_state_dict(weights, strict=True)
    if not all(torch.isfinite(tensor).all() for tensor in actor.state_dict().values()):
        raise ValueError("Checkpoint contains non-finite actor weights")
    output.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(actor)
    scripted.save(str(output))
    loaded = torch.jit.load(str(output)).eval()
    probe = torch.linspace(-1, 1, 4 * 48).reshape(4, 48)
    with torch.inference_mode():
        expected, actual = actor(probe), loaded(probe)
        torch.testing.assert_close(actual, expected)
        if actual.shape != (4, 12) or not torch.isfinite(actual).all():
            raise ValueError("Exported policy failed validation")
    metadata = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    iteration=payload.get("iter"), activation=activation, num_obs=48, num_actions=12,
                    hidden_dims=[weights[f"{i}.weight"].shape[0] for i in indices[:-1]])
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[export] {output} (48 -> 12, {activation}, iteration={metadata['iteration']})", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Full path to model_N.pt")
    parser.add_argument("--output", type=Path, help="Output policy_1.pt; defaults to go2.yaml policy_path")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--activation", help="Must match training; weights alone cannot identify it")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    output = args.output or Path(cfg["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", str(ROOT)))
    torch.set_num_threads(1)
    export_policy(args.checkpoint, output, args.activation or cfg["activation"])


if __name__ == "__main__":
    main()
