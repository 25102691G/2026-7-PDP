"""Pure statistics: sample SD, strict tail threshold and n0/n1 behavior."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


class ReportingTests(unittest.TestCase):
    def api(self):
        p=Path(__file__).resolve().parents[1]/'docs/pdp_research/new_matched_pdp_development/finalize.py'
        spec=importlib.util.spec_from_file_location('matched_report_test',p)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_frozen_task_identity_cannot_use_mutated_source_or_inputs(self):
        module=self.api()
        original=[dict(task_id=str(i),source_sha256={'source':'hash'},input_sha256={'graph':'hash'},status='NOT_ATTEMPTED',attempts=0) for i in range(25)]
        current=[dict(t,status='COMPLETED',attempts=1) for t in original]
        module.validate_frozen_tasks(current,original)
        current[0]['source_sha256']={}
        with self.assertRaises(RuntimeError):module.validate_frozen_tasks(current,original)

    def test_final_resource_gate_precedes_completion(self):
        module=self.api()
        with patch.object(module,'guard_disk',return_value=1) as disk:
            module.gate_final_resources({'verification_wall_seconds':100},1,100)
            disk.assert_called_once_with(100)
            with self.assertRaises(RuntimeError):module.gate_final_resources({'verification_wall_seconds':5*3600},1)

    def test_sample_sd_and_strictly_below_fifty_keep_bad_seed(self):
        rows=[dict(dataset='email',method='New Matched PDP',status='COMPLETED',coverage=x,spread=x/10) for x in (0,50,100)]
        r=next(r for r in self.api().summarize(rows) if r['dataset']=='email' and r['method']=='New Matched PDP')
        self.assertEqual(r['n'],3);self.assertEqual(r['coverage_mean'],50)
        self.assertEqual(r['coverage_sample_sd'],50);self.assertEqual(r['spread_sample_sd'],5)
        self.assertEqual(r['coverage_min'],0);self.assertEqual(r['coverage_range'],100)
        self.assertEqual(r['below50_count'],1);self.assertEqual(r['below50_fraction'],1/3)

    def test_failures_missing_and_single_sample_are_not_fabricated_results(self):
        rows=[dict(dataset='email',method='New Matched PDP',status='COMPLETED',coverage=40,spread=4),
              dict(dataset='email',method='New Matched PDP',status='FAILED'),
              dict(dataset='email',method='New Matched PDP',status='NOT_ATTEMPTED')]
        all_rows=self.api().summarize(rows)
        r=next(r for r in all_rows if r['dataset']=='email' and r['method']=='New Matched PDP')
        self.assertEqual(r['n'],1);self.assertIsNone(r['coverage_sample_sd'])
        self.assertEqual(r['failed'],1);self.assertEqual(r['not_attempted'],1)
        empty=next(r for r in all_rows if r['dataset']=='bitcoin' and r['method']=='New Matched PDP')
        self.assertEqual(empty['n'],0);self.assertIsNone(empty['coverage_mean']);self.assertIsNone(empty['below50_fraction'])


if __name__=='__main__':unittest.main()
