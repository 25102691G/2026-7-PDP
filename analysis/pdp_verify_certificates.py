"""Independently recompute public budget inequalities for every saved model."""

from decimal import Decimal, localcontext
from fractions import Fraction
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUNS = ["integer-pdp-pilot-v1", "stable-slots-pilot-v2",
        "positive-head-pilot-v3", "positive-head-confirmation-v1"]


def exact_conversion_reference(rho, delta):
    with localcontext() as context:
        context.prec = 180
        value = Decimal(rho.numerator)/Decimal(rho.denominator)
        return value+2*(value*(-Decimal(delta).ln())).sqrt()


def verify():
    checks = []
    for run_id in RUNS:
        root = ROOT/"outputs/pdp_runs"/run_id
        manifest = json.loads((root/"manifest.json").read_text())
        protocol = manifest["registered_protocol"]
        for cell in manifest["cells"]:
            cert = json.loads((root/"trusted/models"/cell["id"]/"certificate.json").read_text())
            schema = cert["schema"]
            if schema not in {"b0-integer-range-v1", "stable-slot-node-pdp-v1"}:
                continue
            expected_steps = protocol["training"]["private_steps"]
            radius = Fraction(protocol["training"]["clip_norm"])*(1 << protocol["training"]["grid_bits"])
            radius = radius.numerator//radius.denominator
            assert cert["optimizer_steps"] == expected_steps
            assert cert["clip_radius_lattice"] == radius
            scale = cert["sigma_lattice"]
            if schema == "b0-integer-range-v1":
                sensitivity = 2*protocol["training"]["batch_size"]*radius
                rho = Fraction(expected_steps*sensitivity**2, 2*scale**2)
                assert rho == Fraction(cert["rho_numerator"], cert["rho_denominator"])
                reference = exact_conversion_reference(rho, cert["delta"])
                assert reference <= Decimal(cert["epsilon_upper"]) <= Decimal(cert["epsilon_target"])
                if "personalization" in cert:
                    base = Decimal(cert["epsilon_target"])
                    for target, group in zip(protocol["node_budgets"], cert["personalization"]["groups"]):
                        probability = Fraction(group["probability_numerator"], group["probability_denominator"])
                        delta = probability*Fraction(cert["delta"])
                        assert delta == Fraction(group["delta_upper_numerator"], group["delta_upper_denominator"])
                        assert delta <= Fraction(target["delta"])
                        with localcontext() as context:
                            context.prec = 180
                            p = Decimal(probability.numerator)/Decimal(probability.denominator)
                            epsilon = (1+p*(base.exp()-1)).ln()
                        assert epsilon <= Decimal(group["epsilon_upper"]) <= Decimal(target["epsilon"])
                    assert cert["personalization"]["outer_sampling_count"] == 1
            else:
                description = protocol["stable_slots"]
                limits = description["appearance_limits_by_method"][cell["method"]]
                uniform = cell["method"] in description["uniform_methods"]
                for target, limit, group in zip(protocol["node_budgets"], limits, cert["groups"]):
                    assert group["appearance_limit"] == limit
                    sensitivity = 2*limit*radius
                    assert group["sensitivity_sum_lattice"] == sensitivity
                    rho = Fraction(expected_steps*sensitivity**2, 2*scale**2)
                    assert rho == Fraction(group["rho_numerator"], group["rho_denominator"])
                    epsilon_target = protocol["base_epsilon"][cell["method"]] if uniform else target["epsilon"]
                    delta_target = protocol["delta_0"] if uniform else target["delta"]
                    reference = exact_conversion_reference(rho, group["delta_upper"])
                    assert reference <= Decimal(group["epsilon_upper"]) <= Decimal(epsilon_target)
                    assert Decimal(group["delta_upper"]) <= Decimal(delta_target)
                assert cert["outer_node_sampling"] is False
                assert cert["batch_privacy_amplification"] is False
            checks.append({"run_id": run_id, "cell": cell["id"], "schema": schema,
                           "budget_inequalities_verified": True})
    path = ROOT/"outputs/pdp_reports/final-v1/privacy_verification.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps({
        "private_models_checked": len(checks), "checks": checks,
        "meaning": "Arithmetic and artifact conformance check; the referenced mathematical proofs, not these checks alone, establish privacy.",
    }, indent=2)+"\n")
    print(f"Verified public budget inequalities for {len(checks)} private models")


if __name__ == "__main__":
    verify()
