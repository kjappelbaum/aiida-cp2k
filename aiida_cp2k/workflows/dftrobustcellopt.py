from aiida.orm.code import Code
from aiida.orm.data.base import Float
from aiida.orm.utils import CalculationFactory, DataFactory
from aiida.work import workfunction as wf
from aiida.work.workchain import WorkChain, ToContext, Outputs, while_
from aiida.work.run import submit
from .dftutilities import default_options, dict_merge, disable_printing_charges_dict, empty_pd, merge_ParameterData
from copy import deepcopy

# data objects
StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')
RemoteData = DataFactory('remote')

# workchains
from aiida_cp2k.workflows import Cp2kDftBaseWorkChain
from aiida_cp2k.workflows import Cp2kGeoOptWorkChain
from aiida_cp2k.workflows import Cp2kCellOptWorkChain
from aiida_cp2k.workflows import Cp2kMdWorkChain


default_geo_dict = ParameterData(dict=disable_printing_charges_dict).store()

class Cp2kRobustCellOptWorkChain(WorkChain):
    """Robust workflow that tries to optimize geometry combining molecular dynamics, cell optimization, and standard
    geometry optimization. It is under development!"""
    @classmethod
    def define(cls, spec):
        super(Cp2kRobustCellOptWorkChain, cls).define(spec)

        # specify the inputs of the workchain
        spec.input('code', valid_type=Code)
        spec.input('structure', valid_type=StructureData)
        spec.input('parameters', valid_type=ParameterData, default=empty_pd)
        spec.input('_options', valid_type=dict, default=deepcopy(default_options))
        spec.input('parent_folder', valid_type=RemoteData, default=None, required=False)

        # specify the chain of calculations
        spec.outline(
            cls.setup,
            cls.validate_inputs,
            cls.run_energy,
            cls.parse_calculation,
            cls.run_cellopt_init,
            cls.parse_calculation,
            cls.run_md,
            cls.parse_calculation,
            cls.run_cellopt,
            cls.parse_calculation,
            cls.return_results,
        )

        # specify the outputs of the workchain
        spec.output('output_structure', valid_type=StructureData)
        spec.output('output_parameters', valid_type=ParameterData)
        spec.output('remote_folder', valid_type=RemoteData)

    def setup(self):
        """Setup initial values of all the parameters."""
        self.ctx.parameters = merge_ParameterData(default_geo_dict, self.inputs.parameters)
        try:
            self.ctx.restart_calc = self.inputs.parent_folder
        except:
            self.ctx.restart_calc = None

    def validate_inputs(self):
        #TODO: implement some validations, if necessary
        pass

    def parse_calculation(self):
        """Get the output structure and remember the parent folder."""
        self.ctx.structure = self.ctx.cp2k['output_structure']
        self.ctx.restart_calc = self.ctx.cp2k['remote_folder']
        self.ctx.output_parameters = self.ctx.cp2k['output_parameters'] #from DftBase
        self.ctx.last_dft_dict = {'FORCE_EVAL': {'DFT': None}}
        self.ctx.last_dft_dict['FORCE_EVAL']['DFT'] = \
                deepcopy(self.ctx.cp2k['input_parameters'].get_dict()['FORCE_EVAL']['DFT'])


    def run_energy(self):
        """Run ENERGY calculation."""
        inputs = {
            'code'                : self.inputs.code,
            'structure'           : self.inputs.structure,
            'parameters'          : self.ctx.parameters,
            '_options'            : self.inputs._options,
            '_label'              : 'Stage0_Energy',
            }

        # restart wavefunctions if they are provided
        if self.ctx.restart_calc:
            inputs['parent_folder'] = self.ctx.restart_calc

        # run the calculation
        running  = submit(Cp2kDftBaseWorkChain, **inputs)
        self.report("pk: {} | Stage 0: starting ENERGY".format(running.pid))
        return ToContext(cp2k=Outputs(running))

    def run_cellopt_init(self):
        """Run CELL_OPT calculation."""

        # For the first time we do wery rough cell optimization with only 20 steps max and keeping angles fixed.
        motion_hardcoded = {
                'MOTION':{
                    'CELL_OPT': {
                        'OPTIMIZER': 'BFGS',
                        'MAX_ITER': 20,
                        'KEEP_ANGLES' : True,
                    },
                },
        }
        dict_merge(motion_hardcoded, self.ctx.last_dft_dict)
        inputs = {
            'code'                : self.inputs.code,
            'structure'           : self.ctx.structure,
            'parameters'          : merge_ParameterData(self.ctx.parameters, ParameterData(dict=motion_hardcoded)),
            '_options'            : self.inputs._options,
            '_label'              : 'Stage1_CellOpt',
            }

        # restart wavefunctions if they are provided
        if self.ctx.restart_calc:
            inputs['parent_folder'] = self.ctx.restart_calc

        # run the calculation
        running  = submit(Cp2kCellOptWorkChain, **inputs)
        self.report("pk: {} | Stage 1: preliminary CELL_OPT".format(running.pid))
        return ToContext(cp2k=Outputs(running))

    def run_md(self):
        """Run MD calculation."""
        inputs = {
            'code'                : self.inputs.code,
            'structure'           : self.ctx.structure,
            'parameters'          : merge_ParameterData(ParameterData(dict=self.ctx.last_dft_dict), self.ctx.parameters),
            '_options'            : self.inputs._options,
            '_label'              : 'Stage2_MD',
            }

        # restart wavefunctions if they are provided
        if self.ctx.restart_calc:
            inputs['parent_folder'] = self.ctx.restart_calc

        # run the calculation
        running  = submit(Cp2kMdWorkChain, **inputs)
        self.report("pk: {} | Stage2: shake with MD".format(running.pid))
        return ToContext(cp2k=Outputs(running))

    def run_cellopt(self):
        """Run CELL_OPT calculation."""
        inputs = {
            'code'                : self.inputs.code,
            'structure'           : self.ctx.structure,
            'parameters'          : merge_ParameterData(ParameterData(dict=self.ctx.last_dft_dict), self.ctx.parameters),
            '_options'            : self.inputs._options,
            '_label'              : 'Stage3_CellOpt',
            }

        # restart wavefunctions if they are provided
        if self.ctx.restart_calc:
            inputs['parent_folder'] = self.ctx.restart_calc

        # run the calculation
        running  = submit(Cp2kCellOptWorkChain, **inputs)
        self.report("pk: {} | Stage 3: final CELL_OPT".format(running.pid))
        return ToContext(cp2k=Outputs(running))

    def return_results(self):
        """Return results of the workchain"""
        self.out('output_structure', self.ctx.structure)
        self.out('output_parameters', self.ctx.output_parameters)
        self.out('remote_folder', self.ctx.restart_calc)
