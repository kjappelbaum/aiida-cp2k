#!/usr/bin/env python
import os
import json
from aiida import load_dbenv
load_dbenv()

from aiida.orm import Code, Computer
from aiida.orm import DataFactory


# Read computer and code from file (local_config.json)
with open('local_config.json') as f:
    in_dict = json.load(f)
    computer = Computer.get(in_dict['computer'])
    code = Code.get(in_dict['code'])



#Let's define a simple cubic structure, e.g BaTO3
StructureData = DataFactory('structure')
alat = 9.8528 # angstrom
cell = [[alat, 0., 0.,],
        [0., alat, 0.,],
        [0., 0., alat,],
       ]
s = StructureData(cell=cell)
coords = """O       2.280398       9.146539       5.088696
   O       1.251703       2.406261       7.769908
   O       1.596302       6.920128       0.656695
   O       2.957518       3.771868       1.877387
   O       0.228972       5.884026       6.532308
   O       9.023431       6.119654       0.092451
   O       7.256289       8.493641       5.772041
   O       5.090422       9.467016       0.743177
   O       6.330888       7.363471       3.747750
   O       7.763819       8.349367       9.279457
   O       8.280798       3.837153       5.799282
   O       8.878250       2.025797       1.664102
   O       9.160372       0.285100       6.871004
   O       4.962043       4.134437       0.173376
   O       2.802896       8.690383       2.435952
   O       9.123223       3.549232       8.876721
   O       1.453702       1.402538       2.358278
   O       6.536550       1.146790       7.609732
   O       2.766709       0.881503       9.544263
   O       0.856426       2.075964       5.010625
   O       6.386036       1.918950       0.242690
   O       2.733023       4.452756       5.850203
   O       4.600039       9.254314       6.575944
   O       3.665373       6.210561       3.158420
   O       3.371648       6.925594       7.476036
   O       5.287920       3.270653       6.155080
   O       5.225237       6.959594       9.582991
   O       0.846293       5.595877       3.820630
   O       9.785620       8.164617       3.657879
   O       8.509982       4.430362       2.679946
   O       1.337625       8.580920       8.272484
   O       8.054437       9.221335       1.991376
   H       1.762019       9.820429       5.528454
   H       3.095987       9.107088       5.588186
   H       0.554129       2.982634       8.082024
   H       1.771257       2.954779       7.182181
   H       2.112148       6.126321       0.798136
   H       1.776389       7.463264       1.424030
   H       3.754249       3.824017       1.349436
   H       3.010580       4.524142       2.466878
   H       0.939475       5.243834       6.571945
   H       0.515723       6.520548       5.877445
   H       9.852960       6.490366       0.393593
   H       8.556008       6.860063      -0.294256
   H       7.886607       7.941321       6.234506
   H       7.793855       9.141028       5.315813
   H       4.467366       9.971162       0.219851
   H       5.758685      10.102795       0.998994
   H       6.652693       7.917443       3.036562
   H       6.711966       7.743594       4.539279
   H       7.751955       8.745180      10.150905
   H       7.829208       9.092212       8.679343
   H       8.312540       3.218330       6.528858
   H       8.508855       4.680699       6.189990
   H       9.742249       1.704975       1.922581
   H       8.799060       2.876412       2.095861
   H       9.505360       1.161677       6.701213
   H       9.920117      -0.219794       7.161006
   H       4.749903       4.186003      -0.758595
   H       5.248010       5.018415       0.403676
   H       3.576065       9.078451       2.026264
   H       2.720238       9.146974       3.273164
   H       9.085561       4.493058       9.031660
   H       9.215391       3.166305       9.749133
   H       1.999705       2.060411       1.927796
   H       1.824184       0.564565       2.081195
   H       7.430334       0.849764       7.438978
   H       6.576029       1.537017       8.482885
   H       2.415851       1.576460       8.987338
   H       2.276957       0.099537       9.289499
   H       1.160987       1.818023       4.140602
   H       0.350256       2.874437       4.860741
   H       5.768804       2.638450       0.375264
   H       7.221823       2.257514       0.563730
   H       3.260797       5.243390       5.962382
   H       3.347848       3.732214       5.988196
   H       5.328688       9.073059       5.982269
   H       5.007063       9.672150       7.334875
   H       4.566850       6.413356       3.408312
   H       3.273115       7.061666       2.963521
   H       3.878372       7.435003       6.843607
   H       3.884673       6.966316       8.283117
   H       5.918240       3.116802       5.451335
   H       5.355924       2.495093       6.711958
   H       5.071858       7.687254      10.185667
   H       6.106394       7.112302       9.241707
   H       1.637363       5.184910       4.169264
   H       0.427645       4.908936       3.301903
   H       9.971698       7.227076       3.709104
   H      10.647901       8.579244       3.629806
   H       8.046808       5.126383       2.213838
   H       7.995317       4.290074       3.474723
   H       1.872601       7.864672       7.930401
   H       0.837635       8.186808       8.987268
   H       8.314696      10.115534       2.212519
   H       8.687134       8.667252       2.448452
"""

for line in coords.splitlines():
    symbol, x, y, z = tuple(line.split())
    s.append_atom(position=(x,y,z), symbols=symbol)


#Let's define some parameters
ParameterData = DataFactory('parameter')

parameters = ParameterData(dict={
          'global': {
              'print_level': 'medium',
              'run_type': 'MD',
              'timings': {
                  'threshold': 0.001,
              },
          },
          'motion': {
                'MD': {
                    'ensemble': 'NVE',
                    'steps': 10,
                    'timestep': 0.1,
                    'temperature':300,
                },
                'print': {
                    'velocities': {
                        'each': {
                            'md':1
                        }
                    },
                    'trajectory': {
                        'each': {
                            'md':1
                        }
                    },
                    'forces': {
                        'each': {
                            'md':1
                        }
                    }
                }
            },
          'force_eval': {
              'method': 'quickstep',
              'dft': {
                  'qs': {
                      'eps_default': 1.0e-12,
                      'wf_interpolation': 'ps',
                      'extrapolation_order': 3,
                  },
                  'mgrid': {
                      'ngrids': 4,
                      'cutoff':280,
                      'rel_cutoff': 30,
                  },
                  'xc': {
                      'xc_functional': {
                          '_': 'PADE',
                      },
                  },
                  'scf': {
                      'SCF_GUESS': 'ATOMIC',
                      'OT': {
                          '_': 'ON',
                          'MINIMIZER': 'DIIS',
                      },
                      'MAX_SCF': 20,
                      'EPS_SCF': 1.0E-07,
                      'OUTER_SCF': {
                          'MAX_SCF': 10,
                          'EPS_SCF': 1.0E-7,
                      },
                      'PRINT': {
                          'RESTART': {
                              '_': 'OFF',
                          },
                      },
                  },
              },
          },
})


#what about k-points?
#~ KpointsData = DataFactory('array.kpoints')
#~ kpoints = KpointsData()
#~ kpoints.set_kpoints_mesh([5,5,5])
#~ kpoints.set_kpoints_mesh([5,5,5],offset=(0.5,0.5,0.5))


#Set up the calculation:
calc = code.new_calc()
calc.set_computer(computer)
calc.set_max_wallclock_seconds(30*60) # 30 min
calc.set_resources({"num_machines": 1, "num_mpiprocs_per_machine": 12})

calc.use_structure(s)
calc.use_code(code)
calc.use_parameters(parameters)
#~ calc.use_kpoints(kpoints)
#~ calc.use_pseudos_from_family('all_uspp')


calc.label = "MD"
print calc.store_all()
calc.submit()

