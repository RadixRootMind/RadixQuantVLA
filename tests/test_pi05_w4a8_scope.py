"""CPU regressions for Pi0.5 layer scope and ablation configuration."""
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from gr00t.quantization.duquant_layers import DuQuantConfig, DuQuantLinear, select_targets, wrap_duquant
from tools.quantvla.audit_pi05_w4a8 import CURRENT_INCLUDE, CONSERVATIVE_INCLUDE, EXCLUDED

ROOT = Path(__file__).resolve().parents[1]


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def toy_model():
    model = nn.Module()
    for side in ('paligemma.model.language_model', 'gemma_expert.model'):
        root = 'paligemma_with_expert.' + side + '.layers.0'
        for group, names in [('mlp', ('gate_proj', 'up_proj', 'down_proj')),
                             ('self_attn', ('q_proj', 'k_proj', 'v_proj', 'o_proj'))]:
            for leaf in names:
                parent = model
                parts = (root + '.' + group + '.' + leaf).split('.')
                for part in parts[:-1]:
                    if part not in parent._modules:
                        parent.add_module(part, nn.Module())
                    parent = parent._modules[part]
                parent.add_module(parts[-1], nn.Linear(8, 8, bias=False))
    model.add_module('action_out_proj', nn.Linear(8, 8))
    model.add_module('vision', nn.Linear(8, 8))
    return model


class ScopeRegression(unittest.TestCase):
    def test_actual_wrapper_preserves_attention_and_auxiliary_layers(self):
        with patch.dict(os.environ, {}, clear=True), tempfile.TemporaryDirectory() as directory:
            model = toy_model()
            originals = dict(model.named_modules())
            names = [n for n, _ in select_targets(model, include_regex=CONSERVATIVE_INCLUDE.pattern,
                                                exclude_regex=EXCLUDED.pattern)]
            self.assertEqual(len(names), 6)
            self.assertTrue(all('.mlp.' in n for n in names))
            cfg = DuQuantConfig(weight_bits=4, act_bits=8, block_size=8, block_out_size=8,
                                perm_score='weight', pack_dir=directory)
            wrap_duquant(model, names, cfg)
            for name, original in originals.items():
                if not isinstance(original, nn.Linear):
                    continue
                actual = model.get_submodule(name)
                if name in names:
                    self.assertIsInstance(actual, DuQuantLinear)
                    self.assertEqual(actual.weight_bits, 4)
                    self.assertEqual(actual.cfg.act_bits, 8)
                    output = actual(torch.randn(2, 8))
                    self.assertEqual(tuple(output.shape), (2, 8))
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertEqual(output.dtype, torch.float32)
                else:
                    self.assertIs(actual, original)

    def test_audit_matches_launcher_and_legacy_scope(self):
        source = (ROOT / 'scripts/run_awesome_quant_vla.sh').read_text()
        for var, regex in [('PI05_QVLA_INCLUDE_DEFAULT', CURRENT_INCLUDE),
                           ('PI05_MLP_W4A8_INCLUDE', CONSERVATIVE_INCLUDE),
                           ('PI05_MLP_W4A8_EXCLUDE', EXCLUDED)]:
            self.assertEqual(re.search(var + r"='([^']+)'", source).group(1), regex.pattern)
        targets = select_targets(toy_model(), include_regex=CURRENT_INCLUDE.pattern,
                                 exclude_regex=EXCLUDED.pattern)
        self.assertEqual(len(targets), 10)  # language MLP+attention, expert MLP
        self.assertFalse(any('gemma_expert' in n and 'self_attn' in n for n, _ in targets))

    def test_launcher_plan_defaults_and_four_ablations(self):
        clean = {k: v for k, v in os.environ.items()
                 if not k.startswith(('GR00T_', 'OPENPI_', 'QUANTVLA_', 'AWESOME_QVLA_'))}
        clean.update(QUANTVLA_ROOT=str(ROOT), AWESOME_QVLA_ROOT=str(ROOT))
        def plan(extra):
            return subprocess.check_output(['bash', str(ROOT / 'scripts/run_awesome_quant_vla.sh'),
                                            'pi05_w4a8_mlp_duquant', '--action', 'plan'],
                                           env={**clean, **extra}, text=True, cwd=ROOT)
        neutral = plan({})
        self.assertIn('wbits / abits    : 4 / 8', neutral)
        self.assertIn('pi05 atm / ohb   : 0 / 0', neutral)
        self.assertIn(CONSERVATIVE_INCLUDE.pattern, neutral)
        for atm, ohb in [('0', '0'), ('1', '0'), ('0', '1'), ('1', '1')]:
            result = plan(dict(GR00T_ATM_ALPHA_PATH='/example/candidate.json',
                               GR00T_ATM_ENABLE=atm, GR00T_OHB_ENABLE=ohb))
            self.assertIn(f'pi05 atm / ohb   : {atm} / {ohb}', result)
            self.assertIn('/example/candidate.json', result)

    def test_scheduler_preserves_pi05_ablation_flags(self):
        module = load_script('scope_scheduler', 'run_libero_duquant_benchmark_multi_gpu.py')
        for atm, ohb in [('0', '0'), ('1', '0'), ('0', '1'), ('1', '1')]:
            env = dict(METHOD='duquant', GR00T_ATM_SCOPE='pi05', GR00T_ATM_ENABLE=atm,
                       GR00T_OHB_ENABLE=ohb, GR00T_ATM_ALPHA_PATH='candidate.json')
            module.configure_duquant_env(env, 4, 8, '/tmp/unused-pack')
            self.assertEqual((env['GR00T_ATM_ENABLE'], env['GR00T_OHB_ENABLE']), (atm, ohb))
            self.assertEqual(env['GR00T_ATM_ALPHA_PATH'], 'candidate.json')
        env = dict(METHOD='duquant', GR00T_ATM_ENABLE='1', GR00T_OHB_ENABLE='1')
        module.configure_duquant_env(env, 4, 8, '/tmp/unused-pack')
        self.assertEqual((env['GR00T_ATM_ENABLE'], env['GR00T_OHB_ENABLE']), ('0', '0'))

    def test_service_forwards_denoising_steps(self):
        module = load_script('scope_service', 'openpi_inference_service.py')
        factory = Mock(return_value=types.SimpleNamespace(metadata={}))
        modules = {
            'openpi': types.ModuleType('openpi'),
            'openpi.policies': types.SimpleNamespace(policy=Mock(), policy_config=types.SimpleNamespace(create_trained_policy=factory)),
            'openpi.serving': types.SimpleNamespace(websocket_policy_server=Mock()),
            'openpi.training': types.SimpleNamespace(config=Mock()),
        }
        with patch.dict(sys.modules, modules), \
                patch.object(module, '_apply_quantization_if_requested'), \
                patch.object(module, '_apply_pi05_atm_if_requested'), \
                patch.object(module.socket, 'gethostbyname', return_value='127.0.0.1'):
            for steps, expected in [(0, None), (7, {'num_steps': 7})]:
                module.main(module.ArgsConfig(denoising_steps=steps))
                self.assertEqual(factory.call_args.kwargs['sample_kwargs'], expected)
            with self.assertRaises(ValueError):
                module.main(module.ArgsConfig(denoising_steps=-1))


if __name__ == '__main__':
    unittest.main()
