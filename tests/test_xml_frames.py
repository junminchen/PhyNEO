"""Independent checks of the native multipole frame contract."""
from decimal import Decimal
from pathlib import Path
import subprocess
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def xmls():
    for name in subprocess.check_output(['git', 'ls-files', '*.xml'], cwd=ROOT, text=True).splitlines():
        path = ROOT/name
        if path.is_symlink():
            continue
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        if root.find('ADMPPmeForce') is not None:
            yield name, root


class FrameConventions(unittest.TestCase):
    def test_native_axis_identifiers(self):
        for name, root in xmls():
            types = {a.get('name') for a in root.findall('AtomTypes/Type')}
            for atom in root.findall('ADMPPmeForce/Atom'):
                for key in ('kz', 'kx', 'ky'):
                    axis = atom.get(key, '').lstrip('-')
                    with self.subTest(xml=name, type=atom.get('type'), axis=key):
                        self.assertTrue(not axis or axis in types)

    def test_axial_multipole_components(self):
        for name, root in xmls():
            for atom in root.findall('ADMPPmeForce/Atom'):
                z, x = atom.get('kz', ''), atom.get('kx', '')
                if not z or not x:
                    with self.subTest(xml=name, type=atom.get('type')):
                        for key in ('dX', 'dY', 'qXY', 'qXZ', 'qYZ'):
                            self.assertEqual(Decimal(atom.get(key, '0')), 0)
                        if z:
                            self.assertEqual(Decimal(atom.get('qXX')), Decimal(atom.get('qYY')))
                            self.assertEqual(2*Decimal(atom.get('qXX'))+Decimal(atom.get('qZZ')), 0)
                        else:
                            for key in ('dZ', 'qXX', 'qYY', 'qZZ'):
                                self.assertEqual(Decimal(atom.get(key, '0')), 0)


if __name__ == '__main__':
    unittest.main()
