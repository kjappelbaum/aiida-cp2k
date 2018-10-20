from aiida.orm import Code
from aiida.orm.data.base import Str
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.remote import RemoteData
from aiida.orm.data.structure import StructureData
from aiida.orm.utils import CalculationFactory, DataFactory
from aiida.work.run import submit
from aiida.work.workchain import WorkChain, Outputs, ToContext, if_, while_

from .dftutilities import dict_merge, get_multiplicity, get_atom_kinds, default_options_dict

# calculation objects
Cp2kCalculation = CalculationFactory('cp2k')

cp2k_default_parameters = {
    'GLOBAL':{
            'RUN_TYPE': 'ENERGY',
            'PRINT_LEVEL': 'MEDIUM',
            'EXTENDED_FFT_LENGTHS': True,   # Needed for large systems
    },
    'FORCE_EVAL': {
        'METHOD': 'QUICKSTEP',              #default: QS
        'STRESS_TENSOR': 'ANALYTICAL' ,     #default: NONE
        'DFT': {
            'MULTIPLICITY': 1,
            'UKS': False,
            'CHARGE': 0,
            'BASIS_SET_FILE_NAME': [
               'BASIS_MOLOPT',
               'BASIS_MOLOPT_UCL',
            ],
            'POTENTIAL_FILE_NAME': 'GTH_POTENTIALS',
            'RESTART_FILE_NAME'  : './parent_calc/aiida-RESTART.wfn',
            'QS': {
                'METHOD':'GPW',
            },
            'POISSON': {
                'PERIODIC': 'XYZ',
            },
            'MGRID': {
                'CUTOFF':     600,
                'NGRIDS':       4,
                'REL_CUTOFF':  50,
            },
            'SCF':{
                'SCF_GUESS': 'ATOMIC',
                'EPS_SCF': 1.0e-6,
                'MAX_SCF': 50,
                'MAX_ITER_LUMO': 10000, #needed for the bandgap
                'OT':{
                    'MINIMIZER': 'DIIS',
                    'PRECONDITIONER': 'FULL_ALL',
                    },
                'OUTER_SCF':{
                    'EPS_SCF': 1.0e-6,
                    'MAX_SCF': 10,
                    },
                'PRINT':{
                    'RESTART':{
                        'BACKUP_COPIES': 0,
                        'EACH' :{
                            'QS_SCF': 20,
                        },
                    },
                    'RESTART_HISTORY':{
                        '_': 'OFF'
                    },
                },
            },
            'XC': {
                'XC_FUNCTIONAL': {
                    '_': 'PBE',
                },
                'VDW_POTENTIAL': {
                   'POTENTIAL_TYPE': 'PAIR_POTENTIAL',
                   'PAIR_POTENTIAL': {
                      'PARAMETER_FILE_NAME': 'dftd3.dat',
                      'TYPE': 'DFTD3(BJ)',
                      'REFERENCE_FUNCTIONAL': 'PBE',
                   },
                },
            },
            'PRINT': {
                'E_DENSITY_CUBE': {
                    '_': 'OFF',
                    'STRIDE': '1 1 1',
                },
                'MO_CUBES': {
                    '_': 'ON', # this is to print the band gap
                    'WRITE_CUBE': 'F',
                    'STRIDE': '1 1 1',
                    'NLUMO': 1,
                    'NHOMO': 1,
                },
                'MULLIKEN': {
                    '_': 'ON',  #default: ON
                },
                'LOWDIN': {
                    '_': 'OFF',  #default: OFF
                },
                'HIRSHFELD': {
                    '_': 'OFF',  #default: OFF
                },
            },
        },
        'SUBSYS': {
        },
        'PRINT': {
            'FORCES':{
                '_': 'OFF', #if you want: compute forces with RUN_TYPE ENERGY_FORCE and print them
            },
        },
    },
}

def last_scf_loop(fpath):
    """Simple function that extracts the part of the output starting from the last SCF loop."""
    with open(fpath) as f:
        content = f.readlines()

    # find the last scf loop in the cp2k output file
    for n, line in enumerate(reversed(content)):
        if "SCF WAVEFUNCTION OPTIMIZATION" in line:
            break
    return content[-n-1:]

def scf_converged(fpath):
    """Take the last SCF cycle and check whether it converged or not."""
    content = last_scf_loop(fpath)
    for line in content:
        if "SCF run converged in" in line:
            return True
    return False

def scf_was_diverging(fpath):
    """A function that detects diverging SCF: always diverging if not converged!"""
    return True
#    content = last_scf_loop(fpath)
#    for line in content:
#        if "Minimizer" in line and "CG" in line:
#            grep_string = "OT CG"
#            break
#
#        elif "Minimizer" in line and "DIIS" in line:
#            grep_string = "OT DIIS"
#            break
#
#    n_change = 7
#    difference = []
#    n_positive = 0
#    for line in content:
#        if grep_string in line:
#            difference.append(line.split()[n_change])
#    for number in difference[-12:]:
#        if float(number) > 0:
#            n_positive +=1
#
#    if n_positive>5:
#        return True
#    return False

def scf_getting_weird(fpath):
    """A function that detects weird things are happening."""
    #TODO: True for the moment so that it always switch to CG
    return True

class Cp2kDftBaseWorkChain(WorkChain):
    """A base workchain to be used for DFT calculations with CP2K."""
    @classmethod
    def define(cls, spec):
        super(Cp2kDftBaseWorkChain, cls).define(spec)

        # specify the inputs of the workchain
        spec.input('code', valid_type=Code)
        spec.input('structure', valid_type=StructureData)
        spec.input('parameters', valid_type=ParameterData, default=ParameterData(dict={}))
        spec.input('options', valid_type=ParameterData, default=ParameterData(dict=default_options_dict))
        spec.input('parent_folder', valid_type=RemoteData, default=None, required=False)
        spec.input('_guess_multiplicity', valid_type=bool, default=False)

        # specify the chain of calculations to be performed
        spec.outline(
            cls.setup,
            while_(cls.should_run_calculation)(
                cls.prepare_calculation,
                cls.run_calculation,
                cls.inspect_calculation,
            ),
            cls.return_results,
        )
        
        # specify the outputs of the workchain
        spec.output('output_structure', valid_type=StructureData, required=False)
        spec.output('output_parameters', valid_type=ParameterData)
        spec.output('remote_folder', valid_type=RemoteData)

    def setup(self):
        """Perform initial setup."""
        self.ctx.done = False
        self.ctx.nruns = 0
        self.ctx.structure = self.inputs.structure
        try:
            self.ctx.restart_calc = self.inputs.parent_folder
        except:
            self.ctx.restart_calc = None
        self.ctx.parameters = cp2k_default_parameters
        user_params = self.inputs.parameters.get_dict()

        # As it should be possible to redefine the default atom kinds by user I
        # put the default values prior to merging self.ctx.parameters with
        # user_params
        kinds = get_atom_kinds(self.inputs.structure)
        self.ctx.parameters['FORCE_EVAL']['SUBSYS']['KIND'] = kinds

        dict_merge(self.ctx.parameters, user_params)

        self.ctx.options = self.inputs.options.get_dict()

        # Trying to guess the multiplicity of the system
        if self.inputs._guess_multiplicity:
            self.report("Guessing multiplicity")
            multiplicity = get_multiplicity(self.inputs.structure)
            self.ctx.parameters['FORCE_EVAL']['DFT']['MULTIPLICITY'] = multiplicity
            self.report("Obtained multiplicity: {}".format(multiplicity))
            if multiplicity != 1:
                self.ctx.parameters['FORCE_EVAL']['DFT']['UKS'] = True
                self.report("Switching to UKS calculation")
            else:
                self.report("As multiplicity is 1, I do NOT switch on UKS.")
            # Otherwise take the default

    def should_run_calculation(self):
        return not self.ctx.done

    def prepare_calculation(self):
        """Prepare all the neccessary input links to run the calculation."""
        self.ctx.inputs = {
            'code'      : self.inputs.code,
            'structure' : self.ctx.structure,
            '_options'  : self.ctx.options,
            }

        # restart from the previous calculation only if the necessary data are provided
        # TODO: it should be inlinde or work function that creates a new AiiDA data object
        if self.ctx.restart_calc:
            self.ctx.inputs['parent_folder'] = self.ctx.restart_calc
            self.ctx.parameters['FORCE_EVAL']['DFT']['SCF']['SCF_GUESS'] = 'RESTART'
        else:
            self.ctx.parameters['FORCE_EVAL']['DFT']['SCF']['SCF_GUESS'] = 'ATOMIC'

        # TODO: add geometry restart if it is possible to do so

        # use the new parameters
        p = ParameterData(dict=self.ctx.parameters)
        p.store()
        self.ctx.inputs['parameters'] = p

    def run_calculation(self):
        """Run cp2k calculation."""
        # Create the calculation process and launch it
        process = Cp2kCalculation.process()
        running  = submit(process, **self.ctx.inputs)
        self.report("pk: {} | Running DFT calculation with cp2k".format(running.pid))
        self.ctx.nruns += 1
        return ToContext(calculation=Outputs(running))

    def inspect_calculation(self):
        """Analyse the results of CP2K calculation and decide weather there is a need to restart it. If yes, then
        decide exactly how to restart thea calculation."""
        # TODO: check whether CP2K did not stop the execution because of an
        # error that it detected. In that case the calculation will most
        # probably be in the status "PARSINGFAILED"

        # I will try to disprove those statements. I will not succeed in doing
        # so - the calculation will be considered as completed
        converged_geometry = True
        converged_scf = True
        exceeded_time = False

        # File to analyze
        outfile = self.ctx.calculation['retrieved'].get_abs_path() + '/path/aiida.out'

        # TODO: make a try here, can be that those outputs do not exist
        self.ctx.restart_calc = self.ctx.calculation['remote_folder']
        self.ctx.output_parameters = self.ctx.calculation['output_parameters']

        #TODO: parse and analyse the bandgap

        # First (and the simplest) check is whether the runtime was exceeded
        exceeded_time = self.ctx.output_parameters.dict['exceeded_walltime']
        if exceeded_time:
            self.report("The time of the cp2k calculation has been exceeded")
        else:
            self.report("The time of the cp2k calculation has NOT been exceeded")

        # return converged geometry
        try:
            self.ctx.structure = self.ctx.calculation['output_structure']
        except:
            self.report("Cp2k calculation did not provide any output structure")

        # Second check is whether the last SCF did converge
        converged_scf = scf_converged(outfile)
        if not converged_scf and scf_getting_weird(outfile):
            # If, however, scf was even diverging I should go for more robust
            # minimizer.
            self.ctx.parameters['FORCE_EVAL']['DFT']['SCF']['OT']['MINIMIZER'] = 'CG'
            self.report("Going for more robust (but slow) SCF minimizer")
            # Also, to avoid being trapped in the wrong minimum I restart
            # from atomic wavefunctions.
            self.ctx.restart_calc = None
            self.report("Not going to restart from the previous wavefunctions")
            # I will disable outer_scf steps to enforce convergence
            self.ctx.parameters['FORCE_EVAL']['DFT']['SCF']['MAX_SCF'] = 2000
            self.ctx.parameters['FORCE_EVAL']['DFT']['SCF']['OUTER_SCF']['MAX_SCF'] = 0

            # TODO: I may also look for the forces here. For example a very
            # strong force may cause convergence problems, needs to be
            # implemented
            # UPDATE: from now forces are be printed by default

       # Third check:
       # TODO: check for the geometry convergence/divergence problems
       # useful for geo/cell-opt restart
       # if aiida-1.restart in retrieved (folder):
       #    self.ctx.parameters['EXT_RESTART'] = {'RESTART_FILE_NAME': './parent_calc/aiida-1.restart'}

        # if all is fine, we are done.
        if converged_geometry and converged_scf and not exceeded_time:
            self.report("Calculation converged, terminating the workflow")
            self.ctx.done = True

    def return_results(self):
        self.out('output_structure', self.ctx.structure)
        self.out('output_parameters', self.ctx.output_parameters)
        self.out('remote_folder', self.ctx.restart_calc)
