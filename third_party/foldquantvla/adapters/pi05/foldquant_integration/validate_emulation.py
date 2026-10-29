"""Does the cascade emulation reproduce the deployed W4A4 LLM? Compare kv_stack, PyTorch-emulated vs engine.

Usage: python -m foldquant_integration.validate_emulation --checkpoint ... --dataset-path ... \
           --engine-dir exports/w4a4_cascade/engines [--num 8]
"""
from __future__ import annotations
import argparse, json, tempfile
from pathlib import Path
import torch
from foldquant.export import export_llm, install_llm_emulation
from foldquant.runtime.engine import TensorRTEngine
from foldquant.runtime.plugins import load_plugins
from . import calibration
from .runtime import llm_module, model_of, plugin_libs_of, stack_cache

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint-dir", required=True); ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--engine-dir", required=True); ap.add_argument("--num", type=int, default=8); ap.add_argument("--num-calib", type=int, default=128)
    a = ap.parse_args()
    deployed = calibration.load_policy(a.checkpoint_dir, device="cuda", compile=False)
    model, llm = model_of(deployed), llm_module(deployed)
    ds = calibration.load_dataset(a.dataset_path)
    samples, obs = calibration.sample_observations(ds, a.num_calib, seed=0)
    loop = calibration.make_forward_loop(deployed, obs, seed=0)
    eng_dir = Path(a.engine_dir); manifest = json.loads((eng_dir / "foldquant_export.json").read_text())
    scheme = manifest["schemes"]["llm"]
    # 1) engine kv_stack on held-out prefixes
    load_plugins(plugin_libs_of(eng_dir)); engine = TensorRTEngine(eng_dir / "llm_bf16.engine")
    _, held = calibration.sample_observations(ds, a.num, seed=42)
    prefixes = []
    vwe = model.paligemma_with_expert; orig = vwe.forward
    def spy(*args, **kw):
        if kw.get("inputs_embeds") is not None and kw["inputs_embeds"][1] is None:
            prefixes.append({k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in kw.items()})
        return orig(*args, **kw)
    vwe.forward = spy
    held_loop = calibration.make_forward_loop(deployed, held, seed=42)
    with torch.inference_mode():
        held_loop(vwe)
    vwe.__dict__.pop("forward", None)
    def eng_stack(p):
        am = p["attention_mask"]; am = am[None] if am.dim() == 2 else am
        return engine(prefix_embs=p["inputs_embeds"][0].to(torch.bfloat16), attention_mask=(am.to(torch.bfloat16) if am.is_floating_point() else am.to(torch.bool)),
                      position_ids=p["position_ids"].to(torch.int64))["kv_stack"].clone()
    with torch.inference_mode():
        eng = [eng_stack(p) for p in prefixes]
    # 2) emulated PyTorch kv_stack: re-derive the fold from the same calibration, install, run the prefix
    with tempfile.TemporaryDirectory() as td:
        res = export_llm(llm, Path(td) / "llm.onnx", scheme=scheme, forward_loop=loop, params=(manifest.get("params") or {}).get("llm") or None)
    def flt_stack(p):
        kw = {k: v for k, v in p.items() if k not in ("use_cache", "past_key_values")}
        _h, cache = orig(**kw, past_key_values=None, use_cache=True)
        return stack_cache(cache)
    with torch.inference_mode():
        flt = [flt_stack(p) for p in prefixes]          # control: float PyTorch vs engine
    handle = install_llm_emulation(llm, res)
    try:
        def emu_stack(p):
            # the original prefix seam, exactly as the runtime's PyTorch branch calls it
            kw = {k: v for k, v in p.items() if k not in ("use_cache", "past_key_values")}
            _hidden, cache = orig(**kw, past_key_values=None, use_cache=True)
            return stack_cache(cache)
        with torch.inference_mode():
            emu = [emu_stack(p) for p in prefixes]
    finally:
        handle.remove()
    # 3) compare per layer, K and V
    cos = torch.nn.functional.cosine_similarity
    L = eng[0].shape[0]; rows = []
    for li in range(L):
        for kv, name in ((0, "K"), (1, "V")):
            c = torch.stack([cos(e[li, kv].float().flatten(), m[li, kv].float().flatten(), dim=0) for e, m in zip(eng, emu)])
            rows.append((li, name, c.mean().item(), c.min().item()))
    worst = min(rows, key=lambda r: r[3])
    ctrl = []
    for li in range(L):
        for kv in (0, 1):
            c = torch.stack([cos(e[li, kv].float().flatten(), f[li, kv].float().flatten(), dim=0) for e, f in zip(eng, flt)])
            ctrl.append(c.mean().item())
    emu_mean = sum(r[2] for r in rows) / len(rows); flt_mean = sum(ctrl) / len(ctrl)
    print(f"layers={L} n={len(eng)} scheme={scheme}")
    print(f"emulated-vs-engine mean cosine {emu_mean:.5f} (worst layer {worst[0]} {worst[1]} min {worst[3]:.5f})")
    print(f"float-vs-engine    mean cosine {flt_mean:.5f}   [control: what no emulation gives]")
    print(f"emulation closes {(emu_mean - flt_mean) / max(1e-9, 1 - flt_mean) * 100:.1f}% of the float->engine gap")
    for r in rows[:6] + rows[-2:]: print(f"  layer {r[0]:2d} {r[1]}  mean {r[2]:.5f}  min {r[3]:.5f}")
    (eng_dir / "emulation_validation.json").write_text(json.dumps({"scheme": scheme, "n": len(eng), "layers": L,
        "emulated_vs_engine_mean": emu_mean, "float_vs_engine_mean": flt_mean,
        "per_layer": [{"layer": r[0], "kv": r[1], "mean": r[2], "min": r[3]} for r in rows]}, indent=1))

if __name__ == "__main__":
    main()
