#!/usr/bin/env python3
"""Check native axis encodings and multipole symmetry in the example XMLs."""
import argparse
from decimal import Decimal
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
MOMENTS = ('dX', 'dY', 'dZ', 'qXX', 'qYY', 'qZZ', 'qXY', 'qXZ', 'qYZ')
TRANSVERSE = ('dX', 'dY', 'qXY', 'qXZ', 'qYZ')


def number(value):
    text = format(value, 'f')
    return (text.rstrip('0').rstrip('.') if '.' in text else text) if value else '0'


def update_text(text):
    root = ET.fromstring(text)
    atom_types = {t.get('name') for t in root.findall('AtomTypes/Type')}
    rows, changes = [], []
    for atom in root.findall('ADMPPmeForce/Atom'):
        old = dict(atom.attrib)
        new = dict(old)
        for key in ('kz', 'kx', 'ky'):
            if new.get(key) in ('0', '-0') and '0' not in atom_types:
                new[key] = ''
        z, x = new.get('kz', ''), new.get('kx', '')
        if not z:
            for key in ('kx', 'ky'):
                if key in new:
                    new[key] = ''
            for key in MOMENTS:
                if key in new:
                    new[key] = '0'
        elif not x and not z.startswith('-'):
            for key in TRANSVERSE:
                if key in new:
                    new[key] = '0'
            axial = -Decimal(new['qZZ'])/2
            new['qXX'] = new['qYY'] = number(axial)
        delta = {}
        for key, value in new.items():
            same = old.get(key) == value
            if not same and key in MOMENTS:
                same = Decimal(old[key]) == Decimal(value)
            if not same:
                delta[key] = value
        rows.append(delta)
        if delta:
            changes.append({'type': old['type'], 'before': {k: old[k] for k in delta},
                            'after': delta, 'dZ_preserved': old.get('dZ') == new.get('dZ'),
                            'qZZ_preserved': old.get('qZZ') == new.get('qZZ')})
    index = 0

    def section(match):
        def atom(match_atom):
            nonlocal index
            value = match_atom.group()
            for key, replacement in rows[index].items():
                value, count = re.subn(r'(\b' + key + r'=")[^"]*(")',
                                      lambda m: m[1]+replacement+m[2], value)
                if count != 1:
                    raise ValueError('Missing or duplicate attribute '+key)
            index += 1
            return value
        return match[1] + re.sub(r'<Atom\b[^>]*?/>', atom, match[2]) + match[3]
    updated = re.sub(r'(<ADMPPmeForce\b[^>]*>)(.*?)(</ADMPPmeForce>)', section, text, flags=re.DOTALL)
    before, after = text.splitlines(keepends=True), updated.splitlines(keepends=True)
    for tag, _, _, first, last in SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
        if tag != 'equal':
            for i in range(first, last):
                line = after[i]
                after[i] = line.rstrip(' \t\r\n') + ('\n' if line.endswith('\n') else '')
    return ''.join(after), changes


def maintained_files():
    for name in subprocess.check_output(['git', 'ls-files', '*.xml'], cwd=ROOT, text=True).splitlines():
        path = ROOT/name
        if path.is_symlink():
            continue
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        if root.find('ADMPPmeForce/Atom') is not None:
            yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = []
    for path in maintained_files():
        original = path.read_text()
        updated, changes = update_text(original)
        if args.write and updated != original:
            path.write_text(updated)
        report.append({'path': str(path.relative_to(ROOT)), 'changed': original != updated,
                       'multipole_updates': changes})
    if args.report:
        args.report.write_text(json.dumps(report, indent=2)+'\n')
    count = sum(row['changed'] for row in report)
    print(f"{len(report)} XMLs checked; {count} {'updated' if args.write else 'need updating'}")
    return 0 if args.write or not count else 1


if __name__ == '__main__':
    raise SystemExit(main())
