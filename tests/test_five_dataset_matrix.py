"""Task-scope and seed checks; no real graph or model training."""
import importlib.util
from pathlib import Path
import unittest


class MatrixTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.five_dataset_matrix'))
        from experiments import five_dataset_matrix
        return five_dataset_matrix

    def test_exactly_one_hundred_cells_with_four_configurations(self):
        module=self.api();tasks=module.matrix(Path('/tmp/artificial-study'))
        self.assertEqual(len(tasks),100)
        self.assertEqual(len({t['id'] for t in tasks}),100)
        self.assertEqual({t['dataset'] for t in tasks},{'email','bitcoin','lastfm','hepph','facebook'})
        self.assertEqual({(t['algorithm'],t['epsilon']) for t in tasks},
                         {('PrivIM*',1),('PrivIM*',4),('node_pdp_v1',None),('node_pdp_v2_projected',None)})
        self.assertTrue(all(t['model_seed'] in range(5) and t['partition_seed']==0 for t in tasks))

    def test_pdp_command_uses_explicit_model_seed_and_no_noise_seed(self):
        module=self.api();task=next(t for t in module.matrix(Path('/tmp/artificial-study'))
                                  if t['dataset']=='email' and t['algorithm']=='node_pdp_v1' and t['model_seed']==3)
        command=module.pdp_command(task,Path('/tmp/public-registry.json'))
        self.assertEqual(command[command.index('--model-seed')+1],'3')
        self.assertNotIn('--noise-seed',command)
        self.assertNotIn('--sampling-seed',command)

    def test_completed_or_failed_tasks_are_not_launched_again(self):
        module=self.api()
        self.assertTrue(module.should_execute(dict(status='NEED_RERUN')))
        for status in ['REUSE','COMPLETED','FAILED','RUNNING','INCOMPATIBLE']:
            self.assertFalse(module.should_execute(dict(status=status)))

    def test_privim_command_preserves_full_original_configuration(self):
        module=self.api();task=module.matrix(Path('/tmp/artificial-study'))[3]
        command=module.privim_command(task)
        for flag,value in {'--settings':'PrivIM+','--target_epsilon':'1','--n_iteration':'100',
                           '--loss-penalty':'0.5','--batchsize':'16','--seed':'3',
                           '--model-seed-mode':'fixed','--optimizer-momentum':'0.0'}.items():
            self.assertEqual(command[command.index(flag)+1],value)
        self.assertIn('--dp-noise-entropy',command)
        self.assertIn('--auto-parameters',command)
        self.assertNotIn('--dp-noise-seed',command)

    def test_statistics_keep_bad_results_and_use_sample_standard_deviation(self):
        module=self.api()
        tasks=[dict(dataset='email',algorithm='node_pdp_v1',epsilon=None,status='COMPLETED',
                    model_seed=s,metrics={'coverage':value,'spread':value,'training_seconds':1})
               for s,value in enumerate([0,1,2,3,4])]
        result=module.summarize_tasks(tasks)[0]
        self.assertEqual(result['n'],5)
        self.assertEqual(result['coverage_mean'],2)
        self.assertAlmostEqual(result['coverage_sample_sd'],(2.5)**0.5)

    def test_one_result_and_failed_attempt_are_not_fabricated_repeats(self):
        module=self.api()
        tasks=[dict(dataset='email',algorithm='node_pdp_v1',epsilon=None,status='REUSE',
                    model_seed=0,metrics={'coverage':5,'spread':1,'training_seconds':None}),
               dict(dataset='email',algorithm='node_pdp_v1',epsilon=None,status='FAILED',
                    model_seed=1,failure='synthetic failure')]
        result=module.summarize_tasks(tasks)[0]
        self.assertEqual(result['n'],1)
        self.assertEqual(result['failed_n'],1)
        self.assertIsNone(result['coverage_sample_sd'])


if __name__=='__main__':unittest.main()
