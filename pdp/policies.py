"""Explicit public per-node preferences with a default on the whole ID domain.

The policy must be exogenous and fixed across neighboring graph inputs.
Its preferences are public inputs; this mechanism does not protect preferences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Any, Dict, Mapping

from .personalization import HashBudgetPolicy, NodeBudget


@dataclass(frozen=True, init=False)
class FixedBudgetPolicy(HashBudgetPolicy):
    """Arbitrary fixed ID budgets; absent or unknown IDs retain a public default.

    Do not construct overrides from a current private member list or topology.
    The same object/policy must be used on G and G-v.
    """
    _group_by_id: Mapping[int, int] = field(repr=False, compare=False)

    def __init__(self, default_budget: NodeBudget,
                 overrides: Mapping[int, NodeBudget]):
        def frozen_budget(budget: NodeBudget) -> NodeBudget:
            if not isinstance(budget, NodeBudget):
                raise ValueError("all fixed budgets must be NodeBudget values")
            return NodeBudget(str(budget.epsilon), str(budget.delta))
        default = frozen_budget(default_budget)
        copied = {}
        for identifier, budget in overrides.items():
            if isinstance(identifier, bool) or not isinstance(identifier, int):
                raise ValueError("fixed preference IDs must be integers")
            copied[identifier] = frozen_budget(budget)
        groups = [default]
        group_by_id = {}
        for identifier, budget in sorted(copied.items()):
            if budget not in groups:
                groups.append(budget)
            group_by_id[identifier] = groups.index(budget)
        public_mapping = {
            "default": {"epsilon": default.epsilon, "delta": default.delta},
            "overrides": {
                str(identifier): {"epsilon": budget.epsilon, "delta": budget.delta}
                for identifier, budget in sorted(copied.items())
            },
        }
        digest = hashlib.sha256(json.dumps(public_mapping, sort_keys=True).encode()).hexdigest()
        super().__init__(groups=tuple(groups), salt=digest)
        object.__setattr__(self, "_group_by_id", MappingProxyType(group_by_id))

    def group_for(self, stable_id: int) -> int:
        if isinstance(stable_id, bool) or not isinstance(stable_id, int):
            raise ValueError("stable node IDs must be integers")
        return self._group_by_id.get(stable_id, 0)

    def public_description(self) -> Dict[str, Any]:
        return {
            "policy": "public_fixed_budget_mapping_with_default",
            "public_policy_sha256": self.salt, "default_group": 0,
        }
