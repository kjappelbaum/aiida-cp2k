from aiida.orm import Code
from aiida.orm.data.base import Str
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.remote import RemoteData
from aiida.orm.data.structure import StructureData
from aiida.orm.utils import CalculationFactory, DataFactory
from aiida.work.run import submit
from aiida.work.workchain import WorkChain, Outputs, ToContext, if_, while_
from copy import deepcopy

from .dftutilities import dict_merge, get_atom_kinds, default_options, empty_pd

# calculation objects
Cp2kCalculation = CalculationFactory('cp2k')

# This is a general input that runs PBE-D3(BJ) for ENERGY, MD-NPT_F, GEO_OPT, CELL_OPT
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
                'EPS_SCF': 1.0e-7,
                'MAX_SCF': 50,
                'MAX_ITER_LUMO': 10000, #needed for the bandgap
                'OT':{
                    'MINIMIZER': 'DIIS',
                    'PRECONDITIONER': 'FULL_ALL',
                    },
                'OUTER_SCF':{
                    'EPS_SCF': 1.0e-7,
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
                    '_': 'ON', # this is to print the band gap but only at the end of CELL_OPT/MD/GEO_OPT
                    'WRITE_CUBE': 'F',
                    'STRIDE': '1 1 1',
                    'NLUMO': 1,
                    'NHOMO': 1,
                    'ADD_LAST': 'SYMBOLIC',
                    'EACH': {
                     'CELL_OPT' : 0,
                     'GEO_OPT'  : 0,
                     'MD' : 0,
                    }
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
    'MOTION': {
        'GEO_OPT': {
            'TYPE': 'MINIMIZATION',                     #default: MINIMIZATION
            'OPTIMIZER': 'BFGS',                        #default: BFGS
            'MAX_ITER': 50,                             #default: 200
            'MAX_DR':    '[bohr] 0.0030',               #default: [bohr] 0.0030
            'RMS_DR':    '[bohr] 0.0015',               #default: [bohr] 0.0015
            'MAX_FORCE': '[bohr^-1*hartree] 0.00045',   #default: [bohr^-1*hartree] 0.00045
            'RMS_FORCE': '[bohr^-1*hartree] 0.00030',   #default: [bohr^-1*hartree] 0.00030
            'BFGS' : {
                'TRUST_RADIUS': '[angstrom] 0.25',      #default: [angstrom] 0.25
            },
        },
        'CELL_OPT': {
            'TYPE': 'DIRECT_CELL_OPT',                 #default: DIRECT_CELL_OPT
            'KEEP_ANGLES' : False,                     #default: False
            'KEEP_SYMMETRY': False,                    #default: False (works only if symm is specified in the &CELL)
            'OPTIMIZER': 'BFGS',                       #default: BFGS
            'MAX_ITER': 100,                           #default: 200
            'EXTERNAL_PRESSURE': '[bar] 1.0',          #default: [bar] 100 0 0 0 100 0 0 0 100
            'PRESSURE_TOLERANCE': '[bar] 100',         #default: [bar] 100
            'MAX_DR':    '[bohr] 0.030',               #default: [bohr] 0.0030
            'RMS_DR':    '[bohr] 0.015',               #default: [bohr] 0.0015
            'MAX_FORCE': '[bohr^-1*hartree] 0.0010',   #default: [bohr^-1*hartree] 0.00045
            'RMS_FORCE': '[bohr^-1*hartree] 0.0007',   #default: [bohr^-1*hartree] 0.00030
            'BFGS' : {
                'TRUST_RADIUS': '[angstrom] 0.25',     #default: [angstrom] 0.25
            },
        },
        'MD': {
            'ENSEMBLE': 'NPT_F',                    #main options: NVT, NPT_F
            'STEPS': 20,                            #default: 3
            'TIMESTEP': '[fs] 0.5',                 #default: [fs] 0.5
            'TEMPERATURE': '[K] 300',               #default: [K] 300
            'DISPLACEMENT_TOL': '[angstrom] 1.0',   #default: [bohr] 100
            'THERMOSTAT' : {
                'TYPE': 'CSVR',
                'CSVR': {
                    'TIMECON': 0.1,                 #default: 1000, use: 0.1 for equilibration, 50~100 for production
                },
            },
            'BAROSTAT': {                           #by default the barosthat uses the same thermo as the partricles
                'PRESSURE': '[bar] 1.0',            #default: 0.0
                'TIMECON': '[fs] 1000',             #default: 1000, good for crystals
            },
            'PRINT': {
                'ENERGY': {
                    '_': 'OFF',                     #default: LOW (print .ener file)
                },
            },
        },
        'PRINT': {
            'TRAJECTORY': {
                'FORMAT': 'DCD_ALIGNED_CELL',
                'EACH': {
                    'CELL_OPT' : 1,
                    'GEO_OPT'  : 1,
                    'MD' : 1,
                },
            },
            'RESTART':{
                'BACKUP_COPIES': 0,
                'EACH': {
                    'CELL_OPT' : 1,
                    'GEO_OPT'  : 1,
                    'MD' : 1,
                },
            },
            'RESTART_HISTORY':{
                '_': 'OFF'
                },
            },
            'CELL': {
                '_': 'OFF',
            },
            'FORCES': {
                '_': 'OFF',
            },
            'STRESS': {
                '_': 'OFF',
            },
            'VELOCITIES': {
                '_': 'OFF',
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

def scf_getting_weird(fpath):
    """A function that detects weird things are happening."""
    with open(fpath) as f:
        content = f.readlines()
    # find the last appearance of "Total charge density on r-space grids:" in the file
    # this should compare the expected number of electrons and the number of electrons
    # that was obtainted integrating the charge-density
    for n, line in enumerate(reversed(content)):
        if "Total charge density on r-space grids:" in line:
            break
    if abs(float(content[-n-1].split()[6])) > 1e-4:
        return True
    else:
        return False

class Cp2kDftBaseWorkChain(WorkChain):
    """A base workchain to be used for DFT calculations with CP2K."""
    @classmethod
    def define(cls, spec):
        super(Cp2kDftBaseWorkChain, cls).define(spec)

        # specify the inputs of the workchain
        spec.input('code', valid_type=Code)
        spec.input('structure', valid_type=StructureData)
        spec.input('parameters', valid_type=ParameterData, default=empty_pd)
        spec.input('_options', valid_type=dict, default=deepcopy(default_options))
        spec.input('parent_folder', valid_type=RemoteData, default=None, required=False)
        spec.input('run_type', valid_type=Str, default=Str('energy'), required=False)

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
        spec.output('input_parameters', valid_type=ParameterData)
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

        # deepcopy(cp2k_default_parameters) - should be deepcopy, otherwise the source dictionary is changing as well, and if
        # subworkflow is called several times, the changed value remains in cp2k_default_parameters
        self.ctx.parameters = deepcopy(cp2k_default_parameters)

#        self.report("Cp2kDftBaseWorkchain, self.ctx.parameters getting defaults:\n{}".format(str(self.ctx.parameters)))
        user_params = self.inputs.parameters.get_dict()
#        self.report("Cp2kDftBaseWorkchain, user_params:\n{}".format(str(user_params)))

        # As it should be possible to redefine the default atom kinds by user I
        # put the default values prior to merging self.ctx.parameters with
        # user_params
        kinds = get_atom_kinds(self.inputs.structure)
        self.ctx.parameters['FORCE_EVAL']['SUBSYS']['KIND'] = kinds
#        self.report("Cp2kDftBaseWorkchain, self.ctx.parameters after adding kinds:\n{}".format(str(self.ctx.parameters)))

        dict_merge(self.ctx.parameters, user_params)
#        self.report("Cp2kDftBaseWorkchain, self.ctx.parameters after adding user_params:\n{}".format(str(self.ctx.parameters)))

        self.ctx._options = self.inputs._options

#        self.report("Cp2kDftBaseWorkchain, self.ctx.parameters, final:\n{}".format(str(self.ctx.parameters)))

    def should_run_calculation(self):
        return not self.ctx.done

    def prepare_calculation(self):
        """Prepare all the neccessary input links to run the calculation."""
        self.ctx.inputs = {
            'code'      : self.inputs.code,
            'structure' : self.ctx.structure,
            '_options'  : self.ctx._options,
            '_label'    : 'Cp2kCalculation',
            }

        # restart from the previous calculation only if the necessary data are provided
        # TODO: it should be inlinde or work function that creates a new AiiDA data object
        if self.ctx.restart_calc:
            self.ctx.inputs['parent_folder'] = self.ctx.restart_calc
            self.ctx.parameters['FORCE_EVAL']['DFT']['SCF']['SCF_GUESS'] = 'RESTART'
        else:
            self.ctx.parameters['FORCE_EVAL']['DFT']['SCF']['SCF_GUESS'] = 'ATOMIC'

        if self.inputs.run_type=='energy':
            self.ctx.parameters.pop('MOTION')
        elif self.inputs.run_type=='cell_opt':
            self.ctx.parameters['GLOBAL']['RUN_TYPE'] = 'CELL_OPT'
            dict_merge(self.ctx.parameters, deepcopy(disable_printing_charges_dict))
            dict_merge(self.ctx.parameters, {'FORCE_EVAL':{'PRINT':{'FORCES':{'_': 'OFF'}}}}) # TODO: needed?
        elif self.inputs.run_type=='geo_opt':
            self.ctx.parameters['GLOBAL']['RUN_TYPE'] = 'GEO_OPT'
            dict_merge(self.ctx.parameters, deepcopy(disable_printing_charges_dict))
            dict_merge(self.ctx.parameters, {'FORCE_EVAL':{'PRINT':{'FORCES':{'_': 'OFF'}}}}) # TODO: needed?
        elif self.inputs.run_type=='md':
            self.ctx.parameters['GLOBAL']['RUN_TYPE'] = 'MD'
            dict_merge(self.ctx.parameters, deepcopy(disable_printing_charges_dict))
            dict_merge(self.ctx.parameters, {'FORCE_EVAL':{'PRINT':{'FORCES':{'_': 'OFF'}}}}) # TODO: needed?

        # TODO: add geometry restart if it is possible to do so

        # use the new parameters
        parameters = ParameterData(dict=self.ctx.parameters).store()
        self.ctx.inputs['parameters'] = parameters

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
        exceeded_time = False

        # File to analyze
        outfile = self.ctx.calculation['retrieved'].get_abs_path() + '/path/aiida.out'

        # TODO: make a try here, can be that those outputs do not exist
        self.ctx.restart_calc = self.ctx.calculation['remote_folder']
        self.ctx.output_parameters = self.ctx.calculation['output_parameters']
        self.ctx.retrieved = self.ctx.calculation['retrieved']

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
        olddict = deepcopy(self.ctx.parameters)
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

            if olddict == self.ctx.parameters:
                raise RuntimeError("Cp2kDftBaseWorkChain: Sorry, I no longer know what can be improved.")

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
        self.out('input_parameters', self.ctx.inputs['parameters'])
        self.out('output_structure', self.ctx.structure)
        self.out('output_parameters', self.ctx.output_parameters)
        self.out('remote_folder', self.ctx.restart_calc)
        self.out('retrieved', self.ctx.retrieved)
