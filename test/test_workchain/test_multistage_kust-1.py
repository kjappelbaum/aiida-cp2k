# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
###############################################################################
# Copyright (c), The AiiDA-CP2K authors.                                      #
# SPDX-License-Identifier: MIT                                                #
# AiiDA-CP2K is hosted on GitHub at https://github.com/aiidateam/aiida-cp2k   #
# For further information on the license, see the LICENSE.txt file.           #
###############################################################################

from __future__ import print_function
from __future__ import absolute_import
import os
import click

import sys
import ase.build

from aiida.engine import submit
from aiida.orm import (Code, Dict, StructureData)
from aiida.common import NotExistent
from aiida_cp2k.workchains import Cp2kMultistageWorkChain
from aiida.plugins import DataFactory
# =============================================================================
CifData = DataFactory('cif')


# structure
cif = CifData(file=os.path.abspath("../data/hkust-1.cif"))
structure = cif.get_structure()
structure.store()


@click.command("cli")
@click.argument("codelabel")
@click.option("--run", is_flag=True, help="Actually submit calculation")
def main(codelabel, run):
    try:
        code = Code.get_from_string(codelabel)
    except NotExistent:
        print("The code '{}' does not exist".format(codelabel))
        sys.exit(1)
    # lowering the settings for a cheaper calculation
    parameters = Dict(dict={
            'FORCE_EVAL': {
              'DFT': {
                'MGRID': {
                  'CUTOFF': 280,
                  'REL_CUTOFF': 30,
    }}}})
    options = {"resources": {"num_machines": 2}, "max_wallclock_seconds": 15 * 60 * 60}
    inputs = {
        'protocol_tag': Str('standard'),
        'base': {
            'cp2k': {
                'structure': structure,
                'parameters': parameters,
                'code': code,
                'metadata': {
                    'options': options,
                }
            }
        }
    }


    print("Testing CP2K multistage workchain on HKUST-1")

    if run:
        submit(Cp2kMultistageWorkChain,
                **inputs)
    else:
        print("Generating test input ...")
        inputs["base"]["cp2k"]["metadata"]["dry_run"] = True
        inputs["base"]["cp2k"]["metadata"]["store_provenance"] = False
        run(Cp2kMultistageWorkChain, **inputs)
        print("Submission test successful")
        print("In order to actually submit, add '--run'")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
