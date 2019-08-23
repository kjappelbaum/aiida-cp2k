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

import click
import sys
import ase.build

from aiida.engine import run, submit
from aiida.orm import Code, Dict, StructureData
from aiida.common import NotExistent
from aiida_cp2k.workchains import Cp2kMultistageWorkChain

# This is an expensive test, used as a proof-of-robustness


@click.command("cli")
@click.argument("codelabel")
@click.option("--run", is_flag=True, help="Actually submit calculation")
def main(codelabel, run):
    try:
        code = Code.get_from_string(codelabel)
    except NotExistent:
        print("The code '{}' does not exist".format(codelabel))
        sys.exit(1)

    print("Testing CP2K multistage workchain on Eu-MIL-153")

    # structure
    structure = StructureData(ase=ase.io.read("../data/eu-mil-103.cif"))
    parameters = Dict(dict={})

    options = {"resources": {"num_machines": 3}, "max_wallclock_seconds": 3 * 60 * 60}
    inputs = {
        "protocol_tag": Str("large_system"),
        "starting_settings_idx": Int(0),
        "base": {
            "cp2k": {
                "structure": structure,
                "code": code,
                "parameters": parameters,
                "metadata": {"options": options},
            }
        },
    }

    if run:
        submit(Cp2kMultistageWorkChain, **inputs)
    else:
        print("Generating test input ...")
        inputs["base"]["cp2k"]["metadata"]["dry_run"] = True
        inputs["base"]["cp2k"]["metadata"]["store_provenance"] = False
        run(Cp2kMultistageWorkChain, **inputs)
        print("Submission test successful")
        print("In order to actually submit, add '--submit'")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
