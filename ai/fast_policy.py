import types

import numpy as np
import torch


def _cache_weights(policy):
    def lin(layer):
        return layer.weight.detach().cpu().numpy().T.astype(np.float32), layer.bias.detach().cpu().numpy().astype(np.float32)

    def stack(seq):
        return [lin(m) for m in seq if isinstance(m, torch.nn.Linear)]

    ex = policy.mlp_extractor
    policy._np = {
        "pi": stack(ex.policy_net), "vf": stack(ex.value_net),
        "act": lin(policy.action_net), "val": lin(policy.value_net),
    }


def install_fast_inference(model, seed=0):
    policy = model.policy
    if type(policy.action_dist).__name__ != "MultiCategoricalDistribution" or \
            list(policy.action_dist.action_dims) != [2, 2]:
        return False
    if not isinstance(policy.mlp_extractor.policy_net[1], torch.nn.Tanh):
        return False
    rng = np.random.default_rng(seed)
    orig_set_mode = policy.set_training_mode

    def set_training_mode(self, mode):
        orig_set_mode(mode)
        if not mode:
            _cache_weights(self)

    def mlp(x, layers):
        for w, b in layers:
            x = np.tanh(x @ w + b)
        return x

    def forward(self, obs, deterministic=False):
        if self.training or not hasattr(self, "_np"):
            return type(self).forward(self, obs, deterministic)
        x = obs.detach().cpu().numpy().astype(np.float32, copy=False)
        c = self._np
        logits = mlp(x, c["pi"]) @ c["act"][0] + c["act"][1]
        values = mlp(x, c["vf"]) @ c["val"][0] + c["val"][1]
        logits = logits.reshape(len(x), 2, 2)
        logits = logits - logits.max(axis=2, keepdims=True)
        logp_all = logits - np.log(np.exp(logits).sum(axis=2, keepdims=True))
        p_on = np.exp(logp_all[:, :, 1])
        actions = (p_on > 0.5) if deterministic else (rng.random(p_on.shape) < p_on)
        actions = actions.astype(np.int64)
        logp = np.take_along_axis(logp_all, actions[:, :, None], axis=2)[:, :, 0].sum(axis=1)
        return (torch.from_numpy(actions), torch.from_numpy(values.astype(np.float32)),
                torch.from_numpy(logp.astype(np.float32)))

    policy.set_training_mode = types.MethodType(set_training_mode, policy)
    policy.forward = types.MethodType(forward, policy)
    return True
