"""Check submission script orchestration with a mock interpreter, without Slurm."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class SubmissionValidation(unittest.TestCase):
    def test_vista_script_runs_all_stages_and_preserves_arguments(self):
        script = Path('scripts/vista_pipeline.slurm').resolve()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            interpreter = root/'mock python'
            interpreter.write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
                'with open(os.environ["SS_TEST_LOG"], "a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n')
            interpreter.chmod(0o755)
            env = dict(os.environ, SLURM_SUBMIT_DIR=str(root), SLURM_JOB_ID='test',
                       SLURM_CPUS_PER_TASK='72', SS_WORKERS='3', SS_TEST_LOG=str(root/'calls.jsonl'))
            inputs = '/input directory/*.h5'
            command = ['bash', str(script), str(interpreter), inputs, 'output directory']
            subprocess.run(command, env=env, check=True, capture_output=True, text=True)
            calls = [json.loads(line) for line in (root/'calls.jsonl').read_text().splitlines()]
            self.assertEqual([args[3] for args in calls],
                             ['plan-events', 'prepare-events', 'plan-bins', 'stack-bins'])
            self.assertEqual(calls[0][4], inputs)
            self.assertEqual(calls[1][-2:], ['--workers', '3'])
            self.assertEqual(calls[3][-3:], ['--workers', '3', '--plots'])
            env['SS_WORKERS'] = '73'
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('allocated CPU count', result.stderr)


if __name__ == '__main__':
    unittest.main()
