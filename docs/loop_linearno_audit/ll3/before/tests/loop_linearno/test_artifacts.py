import json
import unittest

from linearno_loop.contracts import CLI_CONTRACT, CLASS_PATHS
from linearno_loop.matrix import configuration_matrix
from linearno_loop.schema import read_metadata, restore_config
from loop_linearno.support import ROOT


class ArtifactTests(unittest.TestCase):
    def test_saved_configuration_matrix_is_reproducible(self):
        path=ROOT/'docs/loop_linearno_audit/ll1/configuration-matrix.json'
        self.assertEqual(json.loads(path.read_text()),configuration_matrix())

    def test_contract_catalog_and_metadata_fixture_are_not_models(self):
        directory=ROOT/'docs/loop_linearno_audit/ll1'
        catalog=json.loads((directory/'contract-catalog.json').read_text())
        self.assertEqual(catalog['cli'],CLI_CONTRACT)
        self.assertEqual(catalog['class_paths'],CLASS_PATHS)
        self.assertEqual(catalog['class_status'],'PLANNED_NOT_IMPLEMENTED_NO_PLACEHOLDERS')
        m=read_metadata(directory/'synthetic-metadata.json')
        self.assertEqual(m['data_spec']['scope'],'synthetic')
        self.assertTrue(restore_config(m)['load_policy']['strict'])
        # LL2 adds primitives; the LL1 planned full-model classes remain absent.
        for name in ('standard.py', 'airfrans.py', 'shapenet.py', 'core.py'):
            self.assertFalse((ROOT/'cdlno/linearno_loop'/name).exists())
        self.assertFalse((ROOT/'tran_evaluate/linearno_loop').exists())


if __name__=='__main__':unittest.main()
