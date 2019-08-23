# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
""" Test/example for the Cp2kMultistageWorkChain"""

from __future__ import print_function
from __future__ import absolute_import

import click
import ase.build

from aiida.engine import run
<<<<<<< HEAD
from aiida.orm import Code, Dict, StructureData, Float, Str
=======
from aiida.orm import (Code, Dict, StructureData)
from aiida.common import NotExistent
>>>>>>> aiida1_multistage
from aiida_cp2k.workchains import Cp2kMultistageWorkChain

@click.command('cli')
@click.argument('cp2k_code_string')
def main(cp2k_code_string):
    """Example usage: verdi run cp2k-5.1@localhost"""

    print("Testing CP2K multistage workchain on H2O (RKS, no need for smearing)...")

<<<<<<< HEAD
    code = Code.get_from_string(cp2k_code_string)
=======
print("Testing CP2K multistage workchain on H2O- (UKS, no need for smearing)...")
>>>>>>> aiida1_multistage

    atoms = ase.build.molecule('H2O')
    atoms.center(vacuum=2.0)
    structure = StructureData(ase=atoms)

<<<<<<< HEAD
    options = {
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        },
        "max_wallclock_seconds": 1 * 3 * 60,
=======
# lowering the settings for acheaper calculation
parameters = Dict(dict={
        'FORCE_EVAL': {
          'DFT': {
            'UKS': True,
            'MULTIPLICITY': 2,
            'CHARGE': -1,
            'MGRID': {
              'CUTOFF': 280,
              'REL_CUTOFF': 30,
}}}})
options = {
    "resources": {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 1,
    },
    "max_wallclock_seconds": 1 * 3 * 60,
}
inputs = {
    'protocol_tag': Str('sp'),
    'base': {
        'cp2k': {
            'structure': structure,
            'parameters': parameters,
            'code': code,
            'metadata': {
                'options': options,
            }
        }
>>>>>>> aiida1_multistage
    }
    inputs = {
        #'min_cell_size': Float(4.1),
        'protocol_tag': Str('test'),
        'cp2k_base': {
            'cp2k': {
                'structure': structure,
                'code': code,
                'metadata': {
                    'options': options,
    }}}}

    run(Cp2kMultistageWorkChain, **inputs)

if __name__ == '__main__':
    main()
