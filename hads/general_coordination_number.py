from ase.io import read
from ase.data import covalent_radii as CR
import sys

from datetime import datetime
import numpy as np

def get_coordination_numbers(atoms, covalent_percent=1.25):
    """Returns an array of coordination numbers and an array of existing bonds determined by
    distance and covalent radii.  By default a bond is defined as 120% of the combined radii
    or less. This can be changed by setting 'covalent_percent' to a float representing a 
    factor to multiple by (default = 1.2).

    If 'exclude' is set to an array,  these atomic numbers with be unable to form bonds.
    This only excludes them from being counted from other atoms,  the coordination
    numbers for these atoms will still be calculated,  but will be unable to form
    bonds to other excluded atomic numbers.
    """

    # Get all the distances
    distances = np.divide(atoms.get_all_distances(mic=True), covalent_percent)
    
    # Atomic Numbers
    numbers = atoms.numbers
    # Coordination Numbers for each atom
    cn = []
    cr = np.take(CR, numbers)
    # Array of indices of bonded atoms.  len(bonded[x]) == cn[x]
    bonded = []
    indices = list(range(len(atoms)))
    for i in indices:
        bondedi = []
        for ii in indices:
            # Skip if measuring the same atom
            if i == ii:
                continue
            if (cr[i] + cr[ii]) >= distances[i,ii]:
                bondedi.append(ii)
        # Add this atoms bonds to the bonded list
        bonded.append(bondedi)
    for i in bonded:
        cn.append(len(i))
    return cn, bonded


def get_gcn(site, cn, bond, surface_type="fcc"):
    """Returns the generalized coordination number of the site given.  To define
    A site,  just give a list containing the indices of the atoms forming the site.
    The calculated coordination numbers and bonding needs to be supplied, and the
    surface type also needs to be changed if the surface is not represented by bulk fcc.
    """
    # Define the types of bulk accepted
    gcn_bulk = {"fcc": [12., 18., 22., 26.], "bcc": [14., 22., 28., 32.]}
    gcn = 0.
    if len(site) == 0:
        return 0
    sur = site
    counted = []
    for i in site:
        counted.append(i)
    for i in sur:
        for ii in bond[i]:
            if ii in counted:
                continue
            counted.append(ii)
            gcn += cn[ii]
    return gcn / gcn_bulk[surface_type][len(site) - 1]


if __name__ == "__main__":
    # These are the tests used to confirm the Pt201 surface generated with a wulff construction of 201 atoms
    # cn, bonds = get_coordination_numbers(read("POSCAR"), exclude=[1, 6, 7, 8])
    # print "Generalized Coordination Number of BULK: ", get_gcn([121], cn, bonds)
    # print "Generalized Coordination Number of FCC111 TOP CENTER: ", get_gcn([190], cn, bonds)
    # print "Generalized Coordination Number of FCC111 TOP MIDDLE: ", get_gcn([144], cn, bonds)
    # print "Generalized Coordination Number of FCC100 TOP: ", get_gcn([197], cn, bonds)
    # print "Generalized Coordination Number of FCC100 EDGE: ", get_gcn([198], cn, bonds)
    # print "Generalized Coordination Number of FCC111 EDGE: ", get_gcn([180], cn, bonds)
    # print "Generalized Coordination Number of KINK: ", get_gcn([145], cn, bonds)
    a = read(sys.argv[1])
    cn, bonds = get_coordination_numbers(a)
    cset = []
    args = sys.argv[3:]
    for i in range(len(args)):
        args[i] = int(args[i])
    for i in range(len(args)):
        for ii in bonds[args[i]]:
            if not ii in args:        
                cset.append(ii)
                a[ii].number = 23
            else:
                a[ii].number = 8
                
    from ase.visualize import view
    view(a)    

    print("Unique bonds to site: ", len(set(cset)))
    print("Coordination Number: ", np.sum(np.take(cn, args)) / len(args))
    print("Generalized Coordination Number: ", get_gcn(args, cn, bonds, surface_type=sys.argv[2]))
