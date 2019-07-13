# -*- coding: utf-8 -*-
"""Multistage workchain"""
from __future__ import absolute_import
import ruamel.yaml as yaml #does not convert OFF to False
import os
from copy import deepcopy

from aiida.common import AttributeDict
from aiida.engine import append_, while_, WorkChain, ToContext
from aiida.engine import workfunction as wf
from aiida.orm import Dict, Int, Float, SinglefileData, Str, RemoteData
from aiida.plugins import CalculationFactory

from aiida_cp2k.workchains import Cp2kBaseWorkChain

Cp2kCalculation = CalculationFactory('cp2k')  # pylint: disable=invalid-name



def merge_dict(dct, merge_dct):
    """ Taken from https://gist.github.com/angstwad/bf22d1822c38a92ec0a9
    Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, merge_dict recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct (overwrites dct data if in both)
    :return: None
    """
    import collections
    for k, v in merge_dct.items(): # it was .iteritems() in python2
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.Mapping)):
            merge_dict(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]

@wf
def merge_Dict(p1, p2):
    p1_dict = p1.get_dict()
    p2_dict = p2.get_dict()
    merge_dict(p1_dict, p2_dict)
    return Dict(dict=p1_dict).store()

def get_kinds_section(structure, multistage_settings):
    """ Write the &KIND sections given the structure and the settings_dict"""
    kinds = []
    all_atoms = set(structure.get_ase().get_chemical_symbols())
    for a in all_atoms:
        kinds.append({
            '_': a,
            'BASIS_SET': multistage_settings['basis_set'][a],
            'POTENTIAL': multistage_settings['pseudopotential'][a],
            'MAGNETIZATION': multistage_settings['initial_magnetization'][a],
            })
    return { 'FORCE_EVAL': { 'SUBSYS': { 'KIND': kinds }}}

def get_input_multiplicity(structure, multistage_settings):
    multiplicity = 1
    all_atoms = structure.get_ase().get_chemical_symbols()
    for key, value in multistage_settings['initial_magnetization'].items():
        multiplicity += all_atoms.count(key) * value
    multiplicity = int(round(multiplicity))
    multiplicity_dict = {'FORCE_EVAL': {'DFT': {'MULTIPLICITY' :multiplicity}}}
    if multiplicity != 1:
        multiplicity_dict['FORCE_EVAL']['DFT']['UKS'] = True
    return multiplicity_dict

class Cp2kMultistageWorkChain(WorkChain):
    """Workchain to submit GEO_OPT/CELL_OPT/MD workchains with iterative fashion"""

    _calculation_class = Cp2kCalculation

    @classmethod
    def define(cls, spec):
        # yapf: disable
        super(Cp2kMultistageWorkChain, cls).define(spec)
        spec.expose_inputs(Cp2kBaseWorkChain, namespace='base')
        spec.input('protocol_tag', valid_type=Str, default=Str('standard'), required=False,
            help='The tag of the protocol: to be read from the multistage_{tag}.yaml')
        spec.input('protocol_yaml', valid_type=SinglefileData, required=False,
            help='Specify a custom yaml file with the multistage settings')
        spec.input('protocol_modify', valid_type=Dict, default=Dict(dict={}), required=False,
            help='Specify custom settings that overvrite the yaml settings')
        spec.input('starting_settings_idx', valid_type=Int, default=Int(0), required=False,
            help='If idx>0 is chosen, the calculation jumps directly to overwriting settings_0 with settings_1')
        spec.input('parent_calc_folder', valid_type=RemoteData, required=False,
            help='Provide an initial parent folder that contains the wavefunction, to which restart')

        spec.outline(
            cls.setup_multistage,
            while_(cls.should_run_stage0)(
              cls.update_settings,
              cls.run_stage,
              cls.inspect_stage0_settings,
            ),
            cls.inspect_stage,
            while_(cls.should_run_stage)(
                cls.update_stage,
                cls.run_stage,
                cls.inspect_stage,
            ),
            cls.results,
        )
        #spec.expose_outputs(Cp2kBaseWorkChain)

    def setup_multistage(self):

        # Store the workchain inputs in context (to be modified later). ctx.base_inp = { base_settings, cp2k: { calculation_settings}}
        self.ctx.base_inp = AttributeDict(self.exposed_inputs(Cp2kBaseWorkChain, 'base'))

        #check if an input parent_calc_folder is provided
        try:
            self.ctx.parent_calc_folder = self.inputs.parent_calc_folder
        except:
            self.ctx.parent_calc_folder = None

        # Read yaml file chosen with the tag and overwrite with custom modifications
        dir = os.path.dirname(os.path.abspath(__file__))
        yamlfullpath = '{}/multistage_{}.yaml'.format(dir, self.inputs.protocol_tag.value)
        with open(yamlfullpath, 'r') as stream:
            self.ctx.parameters_yaml = yaml.safe_load(stream)
        merge_dict(self.ctx.parameters_yaml,self.inputs.protocol_modify.get_dict())

        # Generate input parameters and store them
        self.ctx.parameters = deepcopy(self.ctx.parameters_yaml['settings_0'])
        kinds = get_kinds_section(self.ctx.base_inp['cp2k']['structure'], self.ctx.parameters_yaml)
        merge_dict(self.ctx.parameters,kinds)
        multiplicity = get_input_multiplicity(self.ctx.base_inp['cp2k']['structure'], self.ctx.parameters_yaml)
        merge_dict(self.ctx.parameters,multiplicity)
        merge_dict(self.ctx.parameters,self.ctx.parameters_yaml['stage_0'])

        # Initialize
        self.ctx.iteration_stage0 = 0
        self.ctx.stage_tag = 'stage_0'
        self.ctx.stage_idx = 0
        self.ctx.settings_tag = 'settings_0'
        self.ctx.structure_new = None

    def should_run_stage0(self):
        self.ctx.iteration_stage0 += 1
        if self.ctx.iteration_stage0 == 1 or not self.ctx.settings_ok:
            return True
        else:
            return False

    def update_settings(self):
        # Update the settings idx and check if this is available
        self.ctx.settings_idx = self.inputs.starting_settings_idx.value + (self.ctx.iteration_stage0-1)
        if self.ctx.settings_idx>0:
            self.ctx.settings_tag = 'settings_{}'.format(self.ctx.settings_idx)
            if  self.ctx.settings_tag in self.ctx.parameters_yaml.keys():
                merge_dict(self.ctx.parameters,self.ctx.parameters_yaml[self.ctx.settings_tag])
            else:
                return self.exit_codes(901,'ERROR_NO_MORE_SETTINGS','Stage0 did not converge but there are no more robust settings to try')

    def run_stage(self):
        """Check for restart, prepare input, submit and direct output to context"""

        # Check if it is needed to restart the calculation and provide the parent folder and new structure
        if self.ctx.parent_calc_folder:
            self.ctx.base_inp['cp2k']['parent_calc_folder'] = self.ctx.parent_calc_folder
            self.ctx.parameters['FORCE_EVAL']['DFT']['SCF']['SCF_GUESS'] = 'RESTART'
            self.ctx.parameters['FORCE_EVAL']['DFT']['WFN_RESTART_FILE_NAME'] = './parent_calc/aiida-RESTART.wfn'
            if self.ctx.structure_new:
                self.ctx.base_inp['cp2k']['structure'] = self.ctx.structure_new
        else:
            self.ctx.parameters['FORCE_EVAL']['DFT']['SCF']['SCF_GUESS'] = 'ATOMIC'



        # Overwrite the generated input with the custom cp2k/parameters and give a label to BaseWC and Cp2kCalc
        merge_dict(self.ctx.parameters, AttributeDict(self.exposed_inputs(Cp2kBaseWorkChain, 'base')['cp2k']['parameters'].get_dict()))
        self.ctx.base_inp['cp2k']['parameters'] = Dict(dict = self.ctx.parameters).store()
        self.ctx.base_inp['metadata']['label'] = 'base({}/{})'.format(self.ctx.stage_tag,self.ctx.settings_tag,)
        self.ctx.base_inp['cp2k']['metadata']['label'] = self.ctx.base_inp['cp2k']['parameters'].get_dict()['GLOBAL']['RUN_TYPE']

        running_base = self.submit(Cp2kBaseWorkChain, **self.ctx.base_inp)
        self.report("submitted Cp2kBaseWorkChain<{}> for {}/{}".format(running_base.pk,self.ctx.stage_tag,self.ctx.settings_tag))
        return ToContext(stages=append_(running_base))


    def inspect_stage0_settings(self):
        # bandgap < 0.01ev
        # base not converging
        self.ctx.settings_ok = True

    def inspect_stage(self):
        """ Update geometry and parent folder """
        stage = self.ctx.stages[-1]
        self.ctx.structure_new = stage.outputs.output_structure
        self.ctx.parent_calc_folder = stage.outputs.remote_folder
        return

    def should_run_stage(self):
        """Update the stage idx and tage and check if the new stage is in the protocol,
        if not the WorkChain is finished
        """
        self.ctx.stage_idx += 1
        self.ctx.stage_tag = 'stage_{}'.format(self.ctx.stage_idx)
        return self.ctx.stage_tag in self.ctx.parameters_yaml.keys()

    def update_stage(self):
        """ Update the (&MOTION) settings for the new stage """
        merge_dict(self.ctx.parameters,self.ctx.parameters_yaml[self.ctx.stage_tag])


    def results(self):
        """ Add final info """
        return