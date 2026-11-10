"""Controller guards tested without subprocesses or data access."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class ControlTests(unittest.TestCase):
    def api(self):
        path=ROOT/'docs/pdp_research/new_matched_pdp_development/workflow.py'
        spec=importlib.util.spec_from_file_location('matched_workflow',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_verification_evaluation_budget_is_separate_and_conservative(self):
        module=self.api();plan={}
        module.charge_verification(plan,4*3600)
        self.assertEqual(plan['verification_wall_seconds'],4*3600)
        with self.assertRaises(RuntimeError):module.charge_verification(plan,3601)
        self.assertGreater(plan['verification_wall_seconds'],5*3600)

    def test_no_retry_after_any_failure_or_interruption(self):
        module=self.api()
        with tempfile.TemporaryDirectory() as d:
            tasks=[dict(status='NOT_ATTEMPTED',attempts=0,output=str(Path(d)/str(i))) for i in range(25)]
            module.require_fresh(tasks,0)
            for status in ('RUNNING','FAILED'):
                tasks[0].update(status=status,attempts=1)
                with self.assertRaises(RuntimeError):module.require_fresh(tasks,0)
                with self.assertRaises(RuntimeError):module.require_fresh(tasks,1)

    def test_frozen_order_existing_output_and_attempt_limit(self):
        module=self.api()
        with tempfile.TemporaryDirectory() as d:
            tasks=[dict(status='NOT_ATTEMPTED',attempts=0,output=str(Path(d)/str(i))) for i in range(25)]
            with self.assertRaises(RuntimeError):module.require_fresh(tasks,1)
            Path(tasks[0]['output']).mkdir()
            with self.assertRaises(FileExistsError):module.require_fresh(tasks,0)
            with self.assertRaises(ValueError):module.require_fresh(tasks+[tasks[0]],0)
            with self.assertRaises(ValueError):module.require_fresh(tasks,25)


if __name__=='__main__':unittest.main()
