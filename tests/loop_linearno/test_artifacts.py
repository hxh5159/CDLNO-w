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
        # This saved LL1 fixture is metadata only; LL3 now supplies synthetic
        # SR classes, without turning the historical fixture into trained data.
        self.assertEqual(m['provenance_spec']['code_version'],'LL1-contract-only')
        # LL8 adds actual launchers; the frozen LL1 metadata remains a fixture.
        for task in ('airfoil','darcy','elasticity','pipe','ns','plasticity','airfrans','car'):
            self.assertTrue((ROOT/f'tran_evaluate/linearno_loop/{task}.sh').is_file())


if __name__=='__main__':unittest.main()
