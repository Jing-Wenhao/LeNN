from ase.db import connect
from ase.db.row import AtomsRow
from ase.io import read, Trajectory
from ase.constraints import FixAtoms
from ase import Atoms
from ase.calculators.vasp import Vasp
from pymatgen.io import ase
from pymatgen.core.structure import Structure, IStructure, Molecule
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.adsorption import *
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from matplotlib import pyplot as plt
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.surface import Slab, SlabGenerator, generate_all_slabs, Structure, Lattice
import os
import math
from ase.io import write
from hads import element_information

def caled_db():
    db = connect(r'H.db')
    caled = []
    for atom in db.select():
        specie = atom.get('species')
        id = atom.get('id')
        terminal = atom.get('terminal')
        ads = atom.get('ads')
        ads_e_single = atom.get('ads_e_single')
        caled.append([specie, terminal, ads])
    return caled

def to_cal(db):
    to_cal_db = connect(r'to-cal.db')
    caled = caled_db()
    for atom in db.select():
        specie = atom.get('species')
        structure = atom.toatoms()
        id = atom.get('id')
        terminal = atom.get('terminal')
        ads = atom.get('ads')
        if [specie, terminal, ads] not in caled and len(structure)<=130 and exclude(specie)>=1:
            to_cal_db.write(structure, species=specie, terminal=terminal, ads=ads, miller_index=111)
    return to_cal_db

def exclude(species):
    count = 0
    for i in species:
        if i.isalpha() and i not in element_information.back:
            count += 1
    return count

def cal(rmse):
    db = connect('h_structure1.db')
    db_cal = connect('H.db')

    for i in range(db.count()):
        specie = db[i+1].get('species')
        ter = db[i+1].get('terminal')
        ads = db[i+1].get('ads')
        atom = db.get(i+1).toatoms()
        if len(atom) <= 150:
            if not os.path.isdir(r'/home/jingwh/work/semi-H/%s-%d' % (specie, ter)):
                os.mkdir(r'/home/jingwh/work/semi-H/%s-%d' % (specie, ter))
            if not os.path.isdir(r'/home/jingwh/work/semi-H/%s-%d/cal/%d' % (specie, ter, ads)):
                try:
                    os.mkdir(r'/home/jingwh/work/semi-H/%s-%d/cal/%d' % (specie, ter, ads))
                except FileNotFoundError:
                    os.mkdir(r'/home/jingwh/work/semi-H/%s-%d/cal' % (specie, ter))
                    os.mkdir(r'/home/jingwh/work/semi-H/%s-%d/cal/%d' % (specie, ter, ads))
            write(r'/home/jingwh/work/semi-H/%s-%d/POSCAR-%d' % (specie, ter, ads), atom)
        #    os.system('pwd')
            a, b, c = atom.get_cell()
            k1, k2, k3 = 30 // math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2), \
                         30 // math.sqrt(b[0] ** 2 + b[1] ** 2 + b[2] ** 2), 1

            zmax = 0
            zmin = 100
            for z in atom:
                if z.position[2] > zmax:
                    zmax = z.position[2]
                if z.position[2] < zmin:
                    zmin = z.position[2]
            constraint_line = (zmax - zmin) / 2 + zmin
            constraints = FixAtoms(mask=atom.positions[:, 2] < constraint_line)
            atom.set_constraint(constraints)

            os.chdir(r'/home/jingwh/work/semi-H/%s-%d/cal/%d' % (specie, ter, ads))
            os.system('pwd')
            if ads == 0:
                tmp = os.popen("grep 'reached required' OUTCAR").read()
                if tmp == '':
                    calc = Vasp(
                        xc='pbe',
                        encut=500,
                        kpts=(k1, k2, k3),
                        ediff=1e-5,
                        sigma=0.1,
                        gamma=True,
                        ismear=0,
                        nelm=350,
                        npar=4,
                        directory=r'/home/jingwh/work/semi-H/%s-%d/cal/%d' % (specie, ter, ads),
                        # idipol=3,
                        # ldipol=".TRUE.",
                        algo='fast',
                        prec='normal',
                        # ivdw=11,
                        ispin=1,
                        ibrion=2,
                        potim=0.1,
                        nsw=500,
                        isif=2,
                        ediffg=-0.05,
                        pstress=0,
                        lwave=False,
                        icharg=2,
                        lreal='A',
                        lcharg=False,
                        isym=0,
                        symprec=1e-11,
                        addgrid=True
                    )
                    atom.calc = calc
                    calc.calculate(atom)
                    db_cal.write(atom, species=specie, ads=0, miller_index=111, terminal=ter, ads_e=0, ads_e_single=0)

            else:
                tmp = os.popen("grep 'reached required' OUTCAR").read()
                if tmp == '':
                    traj_results = Trajectory(r'/home/jingwh/work/semi-H/%s-%d/cal/%d/relaxation.traj' % (specie, ter, ads), 'w')
                    calc = Vasp(xc='pbe',
                                encut=500,
                                kpts=(k1, k2, k3),
                                ediff=1e-5,
                                sigma=0.1,
                                gamma=True,
                                ismear=0,
                                nelm=350,
                                npar=4,
                                directory=r'/home/jingwh/work/semi-H/%s-%d/cal/%d' % (specie, ter, ads),
                                # idipol=3,
                                # ldipol=".TRUE.",
                                algo='fast',
                                prec='normal',
                                # ivdw=11,
                                ispin=1,
                                ibrion=2,
                                potim=0.1,
                                nsw=500,
                                isif=2,
                                ediffg=-0.05,
                                pstress=0,
                                lwave=False,
                                icharg=2,
                                lreal='A',
                                lcharg=False,
                                isym=0,
                                symprec=1e-11,
                                addgrid=True
                                )
                    atom.calc = calc
                    calc.calculate(atom)
                    traj_results.write(atom)
                    e_ads = atom.get_potential_energy()
                    e_bare = db_cal.get(species=specie, terminal=ter, ads=0).energy
                    db_cal.write(atom, species=specie, ads=ads, miller_index=111, terminal=ter,ads_e=e_ads - e_bare + ads * 3.402,
                               ads_e_single=e_ads - db_cal.get(species=specie, terminal=ter, ads=ads - 1).energy + 3.402)
