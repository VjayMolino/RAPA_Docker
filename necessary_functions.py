# ===============================================================================
#          FILE: necessary_functions.py
#
#         USAGE:
#
#   DESCRIPTION: This file contains all the functions needed by the script along with the imports line of the packages
#
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#    COPYRIGHTS: Mossa Ghattas, Anthony Cruz-Balberdy, Thomas Kurtzman, Daniel Mckay
#        AUTHOR: Mossa Ghattas, mossa.ghattas@gmail.com
#  ORGANIZATION: Lehman College, Chemistry Department
#     INITIATED: 11/29/2019
#     FINALIZED: 12/20/2022
#      REVISION:  ---
# ===============================================================================


import copy
import sys
import numpy as np
import mdtraj as md
import operator
from collections import defaultdict
from collections import Counter
from math import pi, sin, cos
import itertools
import math
import subprocess as subp
import shlex
import os
import glob
import psutil

# heavy to heavy atom distance cutoff and Cys disulfide bridge cutoff
h_bond_heavy_atm_cutoff = 0.38  # mdtraj uses nm instead of Angstroms. We will be using 3.8 Angstroms (0.38 nm) for heavy-to-heavy atom cutoff.
cys_disulfide_bond_cutoff = 0.23  # mdtraj uses nm instead of Angstroms. We will be using 2.3 Angstroms (0.23 nm) for cys-s---s-cys cutoff.
degenerate_states_e_cutoff = 2.0

##############################################################
# implemented from https://stackoverflow.com/questions/27930038/how-to-define-global-function-in-python
def create_global_counter():
    """Function to create a global function.
    """
    global branch_counter

    def branch_counter(max_branch=2000):
        """Create a global counter used to enumerate the result files.

        Args:
            max_branch (int, optional): Maximum number of results files to enumerate. Defaults to 2000.

        Yields:
            (int): Int number to enumerate the results files
        """
        bcounter = 1
        while bcounter < max_branch:
            yield bcounter
            bcounter += 1

        sys.stderr.write("\n\n\033[1;31;48mStopIteration\033[1;37;0m: Maximum number of branches reached ({}).\nIf you think that this is could be right and you need more branches then increase\033[1;36;48m max_branch\033[1;37;0m on line 50 in\033[1;33;48m necessary_functions.py\033[1;37;0m.\n\n".format(max_branch))
        sys.exit()


# Start the function
create_global_counter()
global_counter = branch_counter()


##############################################################


##############################################################
# implemented from https://stackoverflow.com/questions/12420779/simplest-way-to-get-the-equivalent-of-find-in-python
def findfiles(folder, local_out_prefix):
    """Function to find all files that match the prefix of the given directory tree.

    Args:
        folder (string): name or path to the folder to search.
        local_out_prefix (string): Prefix to filter the results.

    Returns:
        (list): List with the relative path to the results files.
    """
    list_files = []
    for root, folders, files in os.walk(folder):
        for filename in folders + files:
            full_file_path = os.path.join(root, filename)
            if local_out_prefix in full_file_path:
                list_files.append(full_file_path)

    return list_files


##############################################################


##############################################################
# implemented from https://rosettacode.org/wiki/Flatten_a_list
def flatten(list_of_lists):
    """Function to extract all items within list of lists (nested lists) into just one list.

    Args:
        list_of_lists (list): the list to which we will pull all items and put in one dimension list.

    Returns:
        (list): List with all items.
    """
    return sum(([x] if not isinstance(x, list) else flatten(x) for x in list_of_lists), [])


##############################################################


##############################################################
# implemented from https://stackoverflow.com/questions/9553638/find-the-index-of-an-item-in-a-list-of-lists
def find_target(target, list_of_lists):
    """Function to extract the list in a list of lists that contain the "target".

    Args:
        target: the item we are locating in a list of list.
        list_of_lists (list): the list to which we will pull all items and put in one dimension list.

    Returns:
        (list): list that contain the item and its associated "key" - sort of like how dictionary works but with using lists.
    """
    for index_lst, lst in enumerate(list_of_lists):
        for index_lst_son, lst_son in enumerate(lst):
            if lst_son == target:
                return index_lst, index_lst_son


##############################################################


##############################################################
# look up table: use OO dist and OHO angle to get corresponding E in kcal/mol
# https://stackoverflow.com/questions/38953748/python-best-way-to-lookup-values-in-a-table-like-scheme
def PES_lookup_table(branch_name, OO_distance_in_question_in_nm, OHO_angle_in_question):
    """Function to look-up the E of a H-bond.

    Args:
        branch_name (object): the object in which we are evaluating the H-bond.
        OO_distance_in_question_in_nm (float): heavy-to-heavy atom distance in nm.
        OHO_angle_in_question (float): heavyatom-H-heavyatom angle in degrees.

    Returns:
        (float): the E of the H-bond in kcal/mol from the data table
    """

    OO_distance_in_question_in_A = OO_distance_in_question_in_nm * 10
    if 180 >= round(OHO_angle_in_question, 1) >= 90 and 5.0 >= round(OO_distance_in_question_in_A, 1) >= 2.5:

        OO_distances = branch_name.OO_distances
        OHO_angles = branch_name.OHO_angles
        Energies_table = [branch_name.PES_energies[x:x + 901] for x in range(0, len(branch_name.PES_energies), 901)]

        row = OO_distances.index(str(round(OO_distance_in_question_in_A, 1)))
        col = OHO_angles.index(str(round(OHO_angle_in_question, 1)))

        try:
            return round(float(Energies_table[row][col]), 3)
        except IndexError:
            raise ValueError(OO_distance_in_question_in_nm, OHO_angle_in_question)
    else:
        print("       Is the hv-H-hv angle beyond the 90-180 range? Or is the hv-hv distance beyond the 2.5-5 A range?")
        print("       hv-hv distance is {} A and hv-H-hv angle is {}".format(round(OO_distance_in_question_in_A, 3), round(OHO_angle_in_question, 2)))
        return 0
##############################################################


##############################################################
# https://azevedolab.net/resources/dihedral_angle.pdf
# https://azevedolab.net/python.php
def calculate_dihedral(point1, point2, point3, point4):
    """Function to calculate the dihedral angle between 4 atoms.

    Args:
        point1 (list): The x, y, and z coordinates of atom 1.
        point2 (list): The x, y, and z coordinates of atom 2.
        point3 (list): The x, y, and z coordinates of atom 3.
        point4 (list): The x, y, and z coordinates of atom 4.

    Returns:
        (float): the dihedral angle in radians.
    """
    # Calculate coordinates for vectors q1, q2 and q3
    q1 = np.subtract(point2, point1)  # b - a
    q2 = np.subtract(point3, point2)  # c - b
    q3 = np.subtract(point4, point3)  # d - c

    # Calculate cross vectors
    q1_x_q2 = np.cross(q1, q2)
    q2_x_q3 = np.cross(q2, q3)

    # Calculate normal vectors
    n1 = q1_x_q2 / np.sqrt(np.dot(q1_x_q2, q1_x_q2))
    n2 = q2_x_q3 / np.sqrt(np.dot(q2_x_q3, q2_x_q3))

    # Calculate unit vectors
    u1 = n2
    u3 = q2 / (np.sqrt(np.dot(q2, q2)))
    u2 = np.cross(u3, u1)

    # Calculate cosine and sine
    cos_theta = np.dot(n1, u1)
    sin_theta = np.dot(n1, u2)

    # Calculate theta
    dihedral_angle_rad = -math.atan2(sin_theta, cos_theta)  # it is different from atan2 from fortran math.atan2(y,x)

    return dihedral_angle_rad


##############################################################


##############################################################
# https://stackoverflow.com/questions/17763655/rotation-of-a-point-in-3d-about-an-arbitrary-axis-using-python?answertab=votes#tab-top -------------> The script in this link isn't fully correct - I fixed it and used here
# http://scipp.ucsc.edu/~haber/ph216/rotation_12.pdf
def r_matrix_3_by_3(theta, u):
    """Function to construct a rotation matrix.

    Args:
        theta (float): The rotation angle in degrees.
        u (list): The x, y, and z of the vector u of rotation.

    Returns:
        (list): The rotation matrix in a list format.
    """
    return [[cos(theta) + u[0] ** 2 * (1 - cos(theta)), u[0] * u[1] * (1 - cos(theta)) - u[2] * sin(theta), u[0] * u[2] * (1 - cos(theta)) + u[1] * sin(theta)], [u[0] * u[1] * (1 - cos(theta)) + u[2] * sin(theta), cos(theta) + u[1] ** 2 * (1 - cos(theta)), u[1] * u[2] * (1 - cos(theta)) - u[0] * sin(theta)], [u[0] * u[2] * (1 - cos(theta)) - u[1] * sin(theta), u[1] * u[2] * (1 - cos(theta)) + u[0] * sin(theta), cos(theta) + u[2] ** 2 * (1 - cos(theta))]]


##############################################################


##############################################################
# https://stackoverflow.com/questions/17763655/rotation-of-a-point-in-3d-about-an-arbitrary-axis-using-python?answertab=votes#tab-top -------------> The script in this link isnt fully correct - I fixed it and used here
# http://scipp.ucsc.edu/~haber/ph216/rotation_12.pdf
def rotate(point_to_rotate, axis_of_rotation_point1, axis_of_rotation_point2, theta):
    """Function to rotate a point about a vector.

    Args:
        point_to_rotate (list): list that contains the x, y, and z coordinates of the point to be rotated.
        axis_of_rotation_point1 (list): the x, y, and z of point 1 on the the axis of rotation.
        axis_of_rotation_point2 (list): the x, y, and z of point 2 on the the axis of rotation.
        theta (float): the angle of rotation in radians

    Returns:
        (list): the result of the rotation process of 'point_to_rotate'.
    """
    point_translated_to_origin = [point_to_rotate[0] - axis_of_rotation_point1[0], point_to_rotate[1] - axis_of_rotation_point1[1], point_to_rotate[2] - axis_of_rotation_point1[2]]
    p1_translated_to_origin = [axis_of_rotation_point1[0] - axis_of_rotation_point1[0], axis_of_rotation_point1[1] - axis_of_rotation_point1[1], axis_of_rotation_point1[2] - axis_of_rotation_point1[2]]
    p2_translated_to_origin = [axis_of_rotation_point2[0] - axis_of_rotation_point1[0], axis_of_rotation_point2[1] - axis_of_rotation_point1[1], axis_of_rotation_point2[2] - axis_of_rotation_point1[2]]

    u = []
    squaredsum = 0
    for o, f in zip(p1_translated_to_origin, p2_translated_to_origin):
        u.append(f - o)
        squaredsum += (f - o) ** 2
        sqrt_of_squaredsum = math.sqrt(squaredsum)

    u = [o / sqrt_of_squaredsum for o in u]

    r = r_matrix_3_by_3(theta, u)
    rotated = []

    rotated.append((r[0][0] * point_translated_to_origin[0]) + (r[1][0] * point_translated_to_origin[1]) + (r[2][0] * point_translated_to_origin[2]) + axis_of_rotation_point1[0])
    rotated.append((r[0][1] * point_translated_to_origin[0]) + (r[1][1] * point_translated_to_origin[1]) + (r[2][1] * point_translated_to_origin[2]) + axis_of_rotation_point1[1])
    rotated.append((r[0][2] * point_translated_to_origin[0]) + (r[1][2] * point_translated_to_origin[1]) + (r[2][2] * point_translated_to_origin[2]) + axis_of_rotation_point1[2])

    return rotated


##############################################################


##############################################################
def initialize_empty_lists(branch_name):
    """Function to initialize empty lists (attributes) of the object 'branch_name'.

    Args:
        branch_name (object): The object to which we are initializing the empty lists (attributes).

    Returns:
        The empty lists as attributes of the object (branch_name).
    """
    branch_name.switched_atm_name_uncharged = []
    branch_name.atm_name_change_to_N = []
    branch_name.atm_name_change_to_O = []
    branch_name.atm_name_change_to_C = []  # For Histidine flips
    branch_name.directly_set_donor_of_oxi_SER_THR_TYR = []
    branch_name.indirectly_set_donor_of_oxi_SER_THR_TYR = []
    branch_name.his_with_just_one_rotamer = []
    branch_name.his_with_splits = defaultdict(dict)
    branch_name.asngln_with_splits = []

##############################################################


##############################################################
def file_outputting(branch_name, out_prefix):
    """Function to output the branched structure. This produces a file that ends with (*before_changes.pdb).
            We then open this file and change the atom names of the atoms to which they were flipped to like seen in ASN/GLN/HIS if a flip was observed in the analysis.
            Here, we also get rid of all H atoms and we only add H atoms of SER/THR/TYR to which an accepting neighbor is involved in an H-bond and we output the new structure with the observed changes.

    Args:
        branch_name (object): The object to which we are outputting.
        out_prefix (string): The name of the output file we are saving.

    Returns:
        Two .pdb format files of the branched structure (*before_changes.pdb (temporary) and *___*.pdb)
    """
    # This block is for addition of H for O of SER/THR/TYR and for fixing element symbol for flipped ASN/GLN and flipped HIS and it also gets rid of the H atom(s) of: 1) OH group of SER/THR/TYR when it's calculated based on nearest acceptor 2) NH2 side chain of ASN/GLN when a flip has taken place 3) H of CE and CD of HIS/HIE/HID when a flip has taken place

    O_SER_TYR_THR_to_be_protonated_manually = []
    H_coordinates_O_SER_TYR_THR_to_be_added_manually = []

    # for i in branch_name.donor_of_oxi_SER_THR_TYR_acc_neighbor_set:
    #     O_SER_TYR_THR_to_be_protonated_manually.append(i[0][0])
    #
    #     datoms = np.asarray([i[0][0], i[1][0]], dtype="int").reshape(1, 2)
    #     dist = md.compute_distances(branch_name.structure, datoms)[0][0]
    #
    #     x3 = ((0.095 * (i[1][2][0] - i[0][2][0])) / dist) + i[0][2][0]  # 0.095 is the 0.095 nm (0.95 Angstrom) which is length of side chain O-H bond of SER/THR/TYR that spruce uses. We use nm because dist was calculated in nm by mdtraj.
    #     y3 = ((0.095 * (i[1][2][1] - i[0][2][1])) / dist) + i[0][2][1]
    #     z3 = ((0.095 * (i[1][2][2] - i[0][2][2])) / dist) + i[0][2][2]
    #
    #     H_coordinates_O_SER_TYR_THR_to_be_added_manually.append([truncate(x3, 3), truncate(y3, 3), truncate(z3, 3)])

    # This is better (than the previous commented block) because it preserves the C-O-H angle to ~109.5 degrees rather than just directing the H to where the acceptor is and having a bad C-O-H angle
    for i in branch_name.donor_of_oxi_SER_THR_TYR_acc_neighbor_set:
        O_SER_TYR_THR_to_be_protonated_manually.append(i[0][0])

        SER_TYR_THR_O_index = i[0][0]
        acc_neighbor_from_SER_TYR_THR_OH = i[1][0]

        SER_TYR_THR_O_h_coords = coords_of_don_neighbor_atom(branch_name, SER_TYR_THR_O_index, acc_neighbor_from_SER_TYR_THR_OH, 0)  # that 0, means to not print the atom PDB line. Default is 1, which will print the lines.

        H_coordinates_O_SER_TYR_THR_to_be_added_manually.append([round(SER_TYR_THR_O_h_coords[0]*10, 3), round(SER_TYR_THR_O_h_coords[1]*10, 3), round(SER_TYR_THR_O_h_coords[2]*10, 3)])  # I multiplied by 10 cuz original coords were divided by 10 (mdtraj uses nm while pdb strcuture has coordinates in A) and saved in xyz array

    tree_branch_output_file_name_fix = "{}_{}".format(out_prefix, next(global_counter))
    tree_branch_output_file_name_before_changes_name_fix = tree_branch_output_file_name_fix + '_before_changes.pdb'

    try:
        branch_name.structure.save_pdb(tree_branch_output_file_name_before_changes_name_fix)
    except KeyError:
        pass

    with open(tree_branch_output_file_name_before_changes_name_fix, "r") as file:
        in_file = file.readlines()

        tree_branch_output_file_name = '___'.join(branch_name.path)

        with open("{}_file_reference.txt".format(branch_name.path[0]), "a") as outfile_ref:
            # outfile_ref.write("{}.pdb ==> {}.pdb\n".format(tree_branch_output_file_name, tree_branch_output_file_name_fix))
            outfile_ref.write("{} ==> {}.pdb\n".format(tree_branch_output_file_name, tree_branch_output_file_name_fix))

        with open(tree_branch_output_file_name_fix + '.pdb', "w") as outfile:
            index = 0
            atm_nmbr_in_pdb = 1
            for line in in_file:
                if line[0:6] == 'CONECT':
                    continue
                elif line[0:4] == 'ATOM' or line[0:6] == 'HETATM':
                    L1 = line[:6]
                    L2 = atm_nmbr_in_pdb
                    L3 = line[16:76]

                    if index in branch_name.atm_name_change_to_N:
                        atom_name = ' {}'.format(branch_name.topology.atom(index).name)
                        L4 = ' N  '
                    elif index in branch_name.atm_name_change_to_O:
                        atom_name = ' {}'.format(branch_name.topology.atom(index).name)
                        L4 = ' O  '
                    elif index in branch_name.atm_name_change_to_C:
                        atom_name = ' {}'.format(branch_name.topology.atom(index).name)
                        L4 = ' C  '
                    else:
                        atom_name = line[12:16]
                        L4 = line[76:80]

                    if line[76:78] != ' H':  # We will only write non H atoms in the output files. Not going to write any lines of H atoms except the manually calculated H atoms of SER/THR/TYR only if the O atom index is in O_SER_TYR_THR_to_be_protonated_manually
                        if index not in O_SER_TYR_THR_to_be_protonated_manually:
                            outfile.write("{:6s}{:5d} {:^4s}{}{}\n".format(L1, L2, atom_name, L3, L4))
                            atm_nmbr_in_pdb += 1

                        elif index in O_SER_TYR_THR_to_be_protonated_manually:

                            outfile.write("{:6s}{:5d} {:^4s}{}{}\n".format(L1, L2, atom_name, L3, L4))

                            if branch_name.topology.atom(index).name == 'OG':  # SER
                                atom_name = ' HG '
                            elif branch_name.topology.atom(index).name == 'OG1':  # THR
                                atom_name = ' HG1'
                            elif branch_name.topology.atom(index).name == 'OH':  # TYR
                                atom_name = ' HH '

                            residue_info = line[16:30]

                            OH_x = H_coordinates_O_SER_TYR_THR_to_be_added_manually[O_SER_TYR_THR_to_be_protonated_manually.index(index)][0]
                            OH_y = H_coordinates_O_SER_TYR_THR_to_be_added_manually[O_SER_TYR_THR_to_be_protonated_manually.index(index)][1]
                            OH_z = H_coordinates_O_SER_TYR_THR_to_be_added_manually[O_SER_TYR_THR_to_be_protonated_manually.index(index)][2]

                            rest_of_line = line[54:76]

                            outfile.write("{:6s}{:5d} {:^4s}{}{:8.3f}{:8.3f}{:8.3f}{}{}\n".format("ATOM", atm_nmbr_in_pdb + 1, atom_name, residue_info, OH_x, OH_y, OH_z, rest_of_line, ' H  '))
                            atm_nmbr_in_pdb += 2

                    index += 1

                else:
                    outfile.write(line)

        outfile.close()


    print('#############################################################################################################################################################################################')
    print('END PRODUCT FILE WAS OUTPUT AS:\n----------------> {}'.format(tree_branch_output_file_name_fix + '.pdb'))
    print('#############################################################################################################################################################################################')

##############################################################


##############################################################
def checking_for_acid_dyad_branching(branch_name, motherstructure_acid_branches, motherstructure_acidic_residues_branching_by_atoms, out_prefix):
    print('{} ({}) does not have any more unknown residues.'.format(branch_name.name, branch_name))
    print('The complete branching pathway is : {}'.format(branch_name.path))
    print('Proceeding to ACID DYAD check and file outputting................................................')

    local_out_prefix = out_prefix

    if len(motherstructure_acid_branches) > 0:
        print('ACID DYAD BRANCHING.....')
        neg_branch_numbering = 1
        for neg_branch_index in np.arange(len(motherstructure_acid_branches)):

            temp_branch_name = 'NegBranch' + str(neg_branch_numbering)
            globals()['NegBranch' + str(neg_branch_numbering)] = copy.deepcopy(branch_name)
            globals()['NegBranch' + str(neg_branch_numbering)].name = temp_branch_name
            globals()['NegBranch' + str(neg_branch_numbering)].path.append(temp_branch_name)

            print('Negative branch name: {}'.format(temp_branch_name))

            acid_dyad_residues_to_be_kept_unprotonated = []
            acid_dyad_residues_to_be_changed_to_donor_form = []

            for branch_atom in np.unique(flatten(motherstructure_acidic_residues_branching_by_atoms)):
                if branch_atom in flatten(motherstructure_acid_branches[neg_branch_index]):
                    acid_dyad_residues_to_be_changed_to_donor_form.append(branch_name.topology.atom(branch_atom).residue.index)

            for branch_atom in flatten(motherstructure_acidic_residues_branching_by_atoms):
                if branch_name.topology.atom(branch_atom).residue.index not in acid_dyad_residues_to_be_changed_to_donor_form and branch_name.topology.atom(branch_atom).residue.index not in acid_dyad_residues_to_be_kept_unprotonated:
                    acid_dyad_residues_to_be_kept_unprotonated.append(branch_name.topology.atom(branch_atom).residue.index)

            print('{} is going to have residues {} kept unprotonated (charged form) while residues {} will be protonated (neutral form)'.format(temp_branch_name, acid_dyad_residues_to_be_kept_unprotonated, acid_dyad_residues_to_be_changed_to_donor_form))
            print('########################')

            for w in acid_dyad_residues_to_be_kept_unprotonated:

                print('\nMaking sure the residues that are supposed to be kept unprotonated have the unprotonated residue name format (ASP/ASH) .........\n')

                print('{} before conversion'.format(globals()['NegBranch' + str(neg_branch_numbering)].topology.residue(w)))
                if globals()['NegBranch' + str(neg_branch_numbering)].topology.residue(w).name == 'ASH':
                    globals()['NegBranch' + str(neg_branch_numbering)].topology.residue(w).name = 'ASP'
                    print('{} after conversion to donor form'.format(globals()['NegBranch' + str(neg_branch_numbering)].topology.residue(w)))
                elif globals()['NegBranch' + str(neg_branch_numbering)].topology.residue(w).name == 'GLH':
                    globals()['NegBranch' + str(neg_branch_numbering)].topology.residue(w).name = 'GLU'
                    print('{} after conversion to donor form'.format(globals()['NegBranch' + str(neg_branch_numbering)].topology.residue(w)))

            for w in acid_dyad_residues_to_be_changed_to_donor_form:

                print('\nConverting the negatively charged residues into their neutral donor form (PART 2) .........\n')

                print('{} before conversion'.format(globals()['NegBranch' + str(neg_branch_numbering)].topology.residue(w)))
                if globals()['NegBranch' + str(neg_branch_numbering)].topology.residue(w).name == 'ASP':
                    globals()['NegBranch' + str(neg_branch_numbering)].topology.residue(w).name = 'ASH'
                    print('{} after conversion to donor form'.format(globals()['NegBranch' + str(neg_branch_numbering)].topology.residue(w)))
                elif globals()['NegBranch' + str(neg_branch_numbering)].topology.residue(w).name == 'GLU':
                    globals()['NegBranch' + str(neg_branch_numbering)].topology.residue(w).name = 'GLH'
                    print('{} after conversion to donor form'.format(globals()['NegBranch' + str(neg_branch_numbering)].topology.residue(w)))

            branches_acidic_atom_names_that_were_flipped = []

            for w in motherstructure_acid_branches[neg_branch_index]:
                if branch_name.topology.atom(w).name == 'OD1' or branch_name.topology.atom(w).name == 'OE1':
                    temp_OD1OE1 = globals()['NegBranch' + str(neg_branch_numbering)].topology.select('((name OD1 and resname ASH) or (name OE1 and resname GLH)) and resid {}'.format(branch_name.topology.atom(w).residue.index))[0]
                    temp_OD2OE2 = globals()['NegBranch' + str(neg_branch_numbering)].topology.select('((name OD2 and resname ASH) or (name OE2 and resname GLH)) and resid {}'.format(branch_name.topology.atom(w).residue.index))[0]

                    if globals()['NegBranch' + str(neg_branch_numbering)].topology.atom(temp_OD2OE2).name == 'OD2':
                        globals()['NegBranch' + str(neg_branch_numbering)].topology.atom(temp_OD2OE2).name = 'OD1'
                        globals()['NegBranch' + str(neg_branch_numbering)].topology.atom(temp_OD1OE1).name = 'OD2'
                    elif globals()['NegBranch' + str(neg_branch_numbering)].topology.atom(temp_OD2OE2).name == 'OE2':
                        globals()['NegBranch' + str(neg_branch_numbering)].topology.atom(temp_OD2OE2).name = 'OE1'
                        globals()['NegBranch' + str(neg_branch_numbering)].topology.atom(temp_OD1OE1).name = 'OE2'

                    branches_acidic_atom_names_that_were_flipped.append([temp_OD1OE1, temp_OD2OE2])
                    print('Atoms {} \'s names of the winner were just flipped'.format([temp_OD1OE1, temp_OD2OE2]))
                    print('############')

                elif branch_name.topology.atom(w).name == 'OD1' or branch_name.topology.atom(w).name == 'OE1':
                    print('No atom names of the acidic branching atom need to be flipped!!')

            print('Outputting files......')
            file_outputting(globals()['NegBranch' + str(neg_branch_numbering)], local_out_prefix)
            print('########################')

            neg_branch_numbering += 1

    else:
        print('There are No ACID DYADS')
        print('Outputting files......')
        file_outputting(branch_name, local_out_prefix)


##############################################################


##############################################################
def spX_tetrahedron_center_H_finder(branch_name, atom_1, center_atom_H_distance_in_A, hybridization):
    # This function is used inside coords_of_don_neighbor_atom(). It's needed to locate the H atom based on hybridization of the heavy atom.
    # Here I'm using similar approach as I did in HIS donor atom finder in this function, by making OG atom the center atom of a tetrahedron shape like that of C sp3.
    # The trick is that I need to make new coords for cb atom such that the og-cb bond is .840 A (0.084 nm) so when I rotate that vector 109.5 degrees it gets me a new position for the H atom I'm interested in.
    tetrahedron_center = branch_name.xyz[0][atom_1]  # spX hybrizized atom
    center_atom_H_distance_in_nm = center_atom_H_distance_in_A / 10
    if hybridization == 'sp3':
        hybridization_angle_in_radians = 1.9111  # 1.9111 radians is 109.5 degrees (sp3 hybridized atoms angles)
    elif hybridization == 'sp2':
        hybridization_angle_in_radians = 2.0944  # 2.0944 radians is 120 degrees (sp2 hybridized atoms angles)

    spX_bonded_atom_name = 'NOT SET'
    if branch_name.topology.atom(atom_1).name == 'N':
        if branch_name.topology.atom(atom_1).residue.name != 'NME':
            spX_bonded_atom_name = 'CA'
        elif branch_name.topology.atom(atom_1).residue.name == 'NME':
            spX_bonded_atom_name = 'C'
    else:
        if branch_name.topology.atom(atom_1).residue.name in ['SER', 'THR']:
            spX_bonded_atom_name = 'CB'
        elif branch_name.topology.atom(atom_1).residue.name == 'TYR':
            spX_bonded_atom_name = 'CZ'
        elif branch_name.topology.atom(atom_1).residue.name in ['ASN', 'ASP', 'ASH']:
            spX_bonded_atom_name = 'CG'
        elif branch_name.topology.atom(atom_1).residue.name in ['GLN', 'GLU', 'GLH']:
            spX_bonded_atom_name = 'CD'
        elif branch_name.topology.atom(atom_1).residue.name == 'ARG':
            if branch_name.topology.atom(atom_1).name in ['NH1', 'NH2']:
                spX_bonded_atom_name = 'CZ'
        elif branch_name.topology.atom(atom_1).residue.name in ['LYS', 'LYN']:
            spX_bonded_atom_name = 'CE'

    if spX_bonded_atom_name != 'NOT SET':
        spX_bonded_atom = branch_name.xyz[0][branch_name.topology.select('name "{}" and resid {}'.format(spX_bonded_atom_name, branch_name.topology.atom(atom_1).residue.index))[0]]

        length_spX_atom_to_bonded_atom = math.sqrt((tetrahedron_center[0] - spX_bonded_atom[0]) ** 2 + (tetrahedron_center[1] - spX_bonded_atom[1]) ** 2 + (tetrahedron_center[2] - spX_bonded_atom[2]) ** 2)
        spX_bonded_atom_dx = (tetrahedron_center[0] - spX_bonded_atom[0]) / length_spX_atom_to_bonded_atom
        spX_bonded_atom_dy = (tetrahedron_center[1] - spX_bonded_atom[1]) / length_spX_atom_to_bonded_atom
        spX_bonded_atom_dz = (tetrahedron_center[2] - spX_bonded_atom[2]) / length_spX_atom_to_bonded_atom
        new_spX_bonded_atom_x_coord = tetrahedron_center[0] - center_atom_H_distance_in_nm * spX_bonded_atom_dx
        new_spX_bonded_atom_y_coord = tetrahedron_center[1] - center_atom_H_distance_in_nm * spX_bonded_atom_dy
        new_spX_bonded_atom_z_coord = tetrahedron_center[2] - center_atom_H_distance_in_nm * spX_bonded_atom_dz
        # Now I need to translate the tetrahedron center to origin (0,0,0) and that will also change the coords of the spX bonded atom.
        spX_bonded_atom_x_translated = new_spX_bonded_atom_x_coord - tetrahedron_center[0]
        spX_bonded_atom_y_translated = new_spX_bonded_atom_y_coord - tetrahedron_center[1]
        spX_bonded_atom_z_translated = new_spX_bonded_atom_z_coord - tetrahedron_center[2]

        if spX_bonded_atom_z_translated == 0.0:  # in the rare case where this variable is 0, it will lead to nan results which wont be right.
            spX_bonded_atom_z_translated = 0.001
        # https://sciencing.com/vector-perpendicular-8419773.html
        tetrahedron_center_to_spX_bonded_atom_vector = [spX_bonded_atom_x_translated, spX_bonded_atom_y_translated, spX_bonded_atom_z_translated]
        perpendicular_vector = [1, 1, -(tetrahedron_center_to_spX_bonded_atom_vector[0] + tetrahedron_center_to_spX_bonded_atom_vector[1]) / tetrahedron_center_to_spX_bonded_atom_vector[2]]

        # scaling the vector OG-CB' (' means prime) (vector perpendicular to tetrahedron_center_to_spX_bonded_atom_vector) to same length which is 0.084nm (0.84 A) like in cb-og example.
        perpendicular_vector_length_square = 0
        for dim in perpendicular_vector:
            perpendicular_vector_length_square += dim ** 2
        perpendicular_vector_length = np.sqrt(perpendicular_vector_length_square)

        perpendicular_vector_normalized = []
        for dim in perpendicular_vector:
            perpendicular_vector_normalized.append(dim / (perpendicular_vector_length / center_atom_H_distance_in_nm))

        second_point_of_rotation_axis_x = perpendicular_vector_normalized[0] + tetrahedron_center[0]
        second_point_of_rotation_axis_y = perpendicular_vector_normalized[1] + tetrahedron_center[1]
        second_point_of_rotation_axis_z = perpendicular_vector_normalized[2] + tetrahedron_center[2]

        second_point_of_rotation_axis_coords = [second_point_of_rotation_axis_x, second_point_of_rotation_axis_y, second_point_of_rotation_axis_z]

        p4 = [new_spX_bonded_atom_x_coord, new_spX_bonded_atom_y_coord, new_spX_bonded_atom_z_coord]
        p2 = tetrahedron_center
        p3 = second_point_of_rotation_axis_coords
        spX_heavy_atom_H = rotate(p4, p2, p3, hybridization_angle_in_radians)

        return spX_heavy_atom_H
    else:
        print("spX_bonded_atom_name IS NOT SET. How is this possible? Check again!!")


##############################################################


##############################################################
def coords_of_don_neighbor_atom(branch_name, z, y, print_atom_line=1):
    # this function is for finding the H atom coordinates of all different atom names (of polar heavy atoms) in addition to the CE and CD atoms of HIS residue and its different protonated states. It can also use reduce dictionary in case atom isnt found by our way or if the atom is a ligand atom which we dont have H finder incorporated for
    indexerror_count = 0
    best_possible_H_of_the_donor_neighbor = []
    if branch_name.topology.atom(z).name == 'N':  # Backbone N
        # atom_z_residue_name = branch_name.topology.atom(z).residue.name
        atom_z_residue_pdb_recorded_index = int(str(branch_name.topology.atom(z).residue)[3:])

        if atom_z_residue_pdb_recorded_index == 1:  # This takes care of the first residue in sequence of the chain WHICH its N terminal is not capped with ACE. There will be 3 H atoms on the N terminus (sp3 hybridized N).
            try:
                branch_name.xyz[0][branch_name.topology.select('name C and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
            except IndexError:
                print("IndexError")
                indexerror_count += 1
            else:
                p1 = branch_name.xyz[0][branch_name.topology.select('name C and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

            try:
                branch_name.xyz[0][branch_name.topology.select('name CA and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
            except IndexError:
                print("IndexError")
                indexerror_count += 1
            else:
                p2 = branch_name.xyz[0][branch_name.topology.select('name CA and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

            try:
                branch_name.xyz[0][branch_name.topology.select('name N and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
            except IndexError:
                print("IndexError")
                indexerror_count += 1
            else:
                p3 = branch_name.xyz[0][branch_name.topology.select('name N and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

            try:
                spX_tetrahedron_center_H_finder(branch_name, z, 1.00, 'sp3')
            except IndexError:
                print("IndexError")
                indexerror_count += 1
            else:
                p4 = spX_tetrahedron_center_H_finder(branch_name, z, 1.00, 'sp3')

            if indexerror_count == 0:
                radians_of_the_dihedral_to_rotate = calculate_dihedral(p1, p2, p3, p4)
                eclipsed_first_H_atom = rotate(p4, p2, p3, radians_of_the_dihedral_to_rotate)

                H1 = z + 1
                H2 = z + 2
                H3 = z + 3

                H1_xyz = rotate(eclipsed_first_H_atom, p2, p3, pi)
                H2_xyz = rotate(eclipsed_first_H_atom, p2, p3, pi / 3)
                H3_xyz = rotate(eclipsed_first_H_atom, p2, p3, -pi / 3)

                if H1 not in branch_name.reduced_topology_not_available_donors:
                    best_possible_H_of_the_donor_neighbor = [[H1, H1_xyz]]
                if H2 not in branch_name.reduced_topology_not_available_donors:
                    best_possible_H_of_the_donor_neighbor.append([H2, H2_xyz])
                if H3 not in branch_name.reduced_topology_not_available_donors:
                    best_possible_H_of_the_donor_neighbor.append([H3, H3_xyz])  # else:  # if indexerror_count not equal to 0, that means some atoms couldnt be located which means they are missing of we are dealing with an atom that has atomname of N while it's not at all a standard residue

                if len(flatten(best_possible_H_of_the_donor_neighbor)) == 4:
                    best_possible_H_of_the_donor_neighbor = best_possible_H_of_the_donor_neighbor[0]

        else:  # any amino acid residue that is NOT the very first residue in sequence. This also works with NME N atom.
            try:
                branch_name.xyz[0][branch_name.topology.select('name C and resid {}'.format(branch_name.topology.atom(z).residue.index - 1))[0]]
            except IndexError:
                print("IndexError")
                indexerror_count += 1
            else:
                c_atom_xyz = branch_name.xyz[0][branch_name.topology.select('name C and resid {}'.format(branch_name.topology.atom(z).residue.index - 1))[0]]

            try:
                branch_name.xyz[0][branch_name.topology.select('name N and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
            except IndexError:
                print("IndexError")
                indexerror_count += 1
            else:
                n_atom_xyz = branch_name.xyz[0][branch_name.topology.select('name N and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

            # c_atom_xyz = branch_name.xyz[0][branch_name.topology.select('name C and resid {}'.format(branch_name.topology.atom(z).residue.index - 1))[0]]
            # n_atom_xyz = branch_name.xyz[0][branch_name.topology.select('name N and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

            if branch_name.topology.atom(z).residue.name != 'NME':  # amino acid carbon alpha atom has a atom name of CA
                try:
                    branch_name.xyz[0][branch_name.topology.select('name CA and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
                except IndexError:
                    print("IndexError")
                    indexerror_count += 1
                else:
                    ca_atom_xyz = branch_name.xyz[0][branch_name.topology.select('name CA and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

                # ca_atom_xyz = branch_name.xyz[0][branch_name.topology.select('name CA and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
            elif branch_name.topology.atom(z).residue.name == 'NME':  # NME C atom has a atom name of C (in the pdb file it had it as CH3)
                try:
                    branch_name.xyz[0][branch_name.topology.select('name C and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
                except IndexError:
                    print("IndexError")
                    indexerror_count += 1
                else:
                    ca_atom_xyz = branch_name.xyz[0][branch_name.topology.select('name C and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

                # ca_atom_xyz = branch_name.xyz[0][branch_name.topology.select('name C and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

            if indexerror_count == 0:
                vector_1 = c_atom_xyz - n_atom_xyz
                vector_2 = ca_atom_xyz - n_atom_xyz
                vector1_vector_2_cosine_angle = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
                c_n_ca_angle_in_radians = np.arccos(vector1_vector_2_cosine_angle)
                c_n_ca_angle_in_degrees = np.degrees(c_n_ca_angle_in_radians)
                c_n_ca_angle_bisected_in_degrees = c_n_ca_angle_in_degrees / 2

                vector_1 = n_atom_xyz - c_atom_xyz
                vector_2 = ca_atom_xyz - c_atom_xyz
                vector1_vector_2_cosine_angle = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
                n_c_ca_angle_in_radians = np.arccos(vector1_vector_2_cosine_angle)
                n_c_ca_angle_in_degrees = np.degrees(n_c_ca_angle_in_radians)

                n_midpoint_c_angle_in_degrees = 180 - c_n_ca_angle_bisected_in_degrees - n_c_ca_angle_in_degrees
                n_c_length_in_nm = math.sqrt(((c_atom_xyz[0] - n_atom_xyz[0]) ** 2) + ((c_atom_xyz[1] - n_atom_xyz[1]) ** 2) + ((c_atom_xyz[2] - n_atom_xyz[2]) ** 2))

                # https://www.calculator.net/triangle-calculator.html?vc=60.903&vx=1.320&vy=&va=30.663&vz=&vb=88.434&angleunits=d&x=66&y=27
                midpoint_c_length_in_nm = n_c_length_in_nm * np.sin(np.radians(c_n_ca_angle_bisected_in_degrees)) / np.sin(np.radians(n_midpoint_c_angle_in_degrees))

                length_c_ca = math.sqrt(((c_atom_xyz[0] - ca_atom_xyz[0]) ** 2) + ((c_atom_xyz[1] - ca_atom_xyz[1]) ** 2) + ((c_atom_xyz[2] - ca_atom_xyz[2]) ** 2))
                C_dx = (c_atom_xyz[0] - ca_atom_xyz[0]) / length_c_ca
                C_dy = (c_atom_xyz[1] - ca_atom_xyz[1]) / length_c_ca
                C_dz = (c_atom_xyz[2] - ca_atom_xyz[2]) / length_c_ca
                midpoint_c_ca_that_disects_c_n_ca_angle_x_coord = c_atom_xyz[0] - midpoint_c_length_in_nm * C_dx
                midpoint_c_ca_that_disects_c_n_ca_angle_y_coords = c_atom_xyz[1] - midpoint_c_length_in_nm * C_dy
                midpoint_c_ca_that_disects_c_n_ca_angle_z_coords = c_atom_xyz[2] - midpoint_c_length_in_nm * C_dz
                midpoint_c_ca_xyz = [midpoint_c_ca_that_disects_c_n_ca_angle_x_coord, midpoint_c_ca_that_disects_c_n_ca_angle_y_coords, midpoint_c_ca_that_disects_c_n_ca_angle_z_coords]

                length_midpoint_c_ca_to_n = math.sqrt((n_atom_xyz[0] - midpoint_c_ca_xyz[0]) ** 2 + (n_atom_xyz[1] - midpoint_c_ca_xyz[1]) ** 2 + (n_atom_xyz[2] - midpoint_c_ca_xyz[2]) ** 2)
                N_dx = (n_atom_xyz[0] - midpoint_c_ca_xyz[0]) / length_midpoint_c_ca_to_n
                N_dy = (n_atom_xyz[1] - midpoint_c_ca_xyz[1]) / length_midpoint_c_ca_to_n
                N_dz = (n_atom_xyz[2] - midpoint_c_ca_xyz[2]) / length_midpoint_c_ca_to_n
                N_H_x_coord = n_atom_xyz[0] + 0.100 * N_dx
                N_H_y_coord = n_atom_xyz[1] + 0.100 * N_dy
                N_H_z_coord = n_atom_xyz[2] + 0.100 * N_dz
                backbone_N_H_atom = [N_H_x_coord, N_H_y_coord, N_H_z_coord]
                #best_possible_H_of_the_donor_neighbor = [backbone_N_H_atom]
                best_possible_H_of_the_donor_neighbor = backbone_N_H_atom

    elif branch_name.topology.atom(z).name == 'NE' and branch_name.topology.atom(z).residue.name == 'ARG':  # ARG NE
        try:
            branch_name.xyz[0][branch_name.topology.select('name CZ and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            cz_atom_xyz = branch_name.xyz[0][branch_name.topology.select('name CZ and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

        try:
            branch_name.xyz[0][branch_name.topology.select('name NE and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            ne_atom_xyz = branch_name.xyz[0][branch_name.topology.select('name NE and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

        try:
            branch_name.xyz[0][branch_name.topology.select('name CD and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            cd_atom_xyz = branch_name.xyz[0][branch_name.topology.select('name CD and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

        # cz_atom_xyz = branch_name.xyz[0][branch_name.topology.select('name CZ and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        # ne_atom_xyz = branch_name.xyz[0][branch_name.topology.select('name NE and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        # cd_atom_xyz = branch_name.xyz[0][branch_name.topology.select('name CD and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

        if indexerror_count == 0:
            vector_1 = cz_atom_xyz - ne_atom_xyz
            vector_2 = cd_atom_xyz - ne_atom_xyz
            vector1_vector_2_cosine_angle = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
            cz_ne_cd_angle_in_radians = np.arccos(vector1_vector_2_cosine_angle)
            cz_ne_cd_angle_in_degrees = np.degrees(cz_ne_cd_angle_in_radians)
            cz_ne_cd_angle_bisected_in_degrees = cz_ne_cd_angle_in_degrees / 2

            vector_1 = ne_atom_xyz - cz_atom_xyz
            vector_2 = cd_atom_xyz - cz_atom_xyz
            vector1_vector_2_cosine_angle = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
            ne_cz_cd_angle_in_radians = np.arccos(vector1_vector_2_cosine_angle)
            ne_cz_cd_angle_in_degrees = np.degrees(ne_cz_cd_angle_in_radians)

            ne_midpoint_cz_angle_in_degrees = 180 - cz_ne_cd_angle_bisected_in_degrees - ne_cz_cd_angle_in_degrees
            ne_cz_length_in_nm = math.sqrt(((cz_atom_xyz[0] - ne_atom_xyz[0]) ** 2) + ((cz_atom_xyz[1] - ne_atom_xyz[1]) ** 2) + ((cz_atom_xyz[2] - ne_atom_xyz[2]) ** 2))

            # https://www.calculator.net/triangle-calculator.html?vc=60.903&vx=1.320&vy=&va=30.663&vz=&vb=88.434&angleunits=d&x=66&y=27
            midpoint_cz_length_in_nm = ne_cz_length_in_nm * np.sin(np.radians(cz_ne_cd_angle_bisected_in_degrees)) / np.sin(np.radians(ne_midpoint_cz_angle_in_degrees))

            length_cz_cd = math.sqrt(((cz_atom_xyz[0] - cd_atom_xyz[0]) ** 2) + ((cz_atom_xyz[1] - cd_atom_xyz[1]) ** 2) + ((cz_atom_xyz[2] - cd_atom_xyz[2]) ** 2))
            cz_dx = (cz_atom_xyz[0] - cd_atom_xyz[0]) / length_cz_cd
            cz_dy = (cz_atom_xyz[1] - cd_atom_xyz[1]) / length_cz_cd
            cz_dz = (cz_atom_xyz[2] - cd_atom_xyz[2]) / length_cz_cd
            midpoint_cz_cd_that_disects_cz_ne_cd_angle_x_coord = cz_atom_xyz[0] - midpoint_cz_length_in_nm * cz_dx
            midpoint_cz_cd_that_disects_cz_ne_cd_angle_y_coords = cz_atom_xyz[1] - midpoint_cz_length_in_nm * cz_dy
            midpoint_cz_cd_that_disects_cz_ne_cd_angle_z_coords = cz_atom_xyz[2] - midpoint_cz_length_in_nm * cz_dz
            midpoint_cz_cd_xyz = [midpoint_cz_cd_that_disects_cz_ne_cd_angle_x_coord, midpoint_cz_cd_that_disects_cz_ne_cd_angle_y_coords, midpoint_cz_cd_that_disects_cz_ne_cd_angle_z_coords]

            length_midpoint_cz_cd_to_ne = math.sqrt((ne_atom_xyz[0] - midpoint_cz_cd_xyz[0]) ** 2 + (ne_atom_xyz[1] - midpoint_cz_cd_xyz[1]) ** 2 + (ne_atom_xyz[2] - midpoint_cz_cd_xyz[2]) ** 2)
            ne_dx = (ne_atom_xyz[0] - midpoint_cz_cd_xyz[0]) / length_midpoint_cz_cd_to_ne
            ne_dy = (ne_atom_xyz[1] - midpoint_cz_cd_xyz[1]) / length_midpoint_cz_cd_to_ne
            ne_dz = (ne_atom_xyz[2] - midpoint_cz_cd_xyz[2]) / length_midpoint_cz_cd_to_ne
            ne_H_x_coord = ne_atom_xyz[0] + 0.100 * ne_dx
            ne_H_y_coord = ne_atom_xyz[1] + 0.100 * ne_dy
            ne_H_z_coord = ne_atom_xyz[2] + 0.100 * ne_dz
            ne_H_atom = [ne_H_x_coord, ne_H_y_coord, ne_H_z_coord]
            #best_possible_H_of_the_donor_neighbor = [ne_H_atom]
            best_possible_H_of_the_donor_neighbor = ne_H_atom

    elif branch_name.topology.atom(z).name == 'NE1' and branch_name.topology.atom(z).residue.name == 'TRP':  # TRP NE1
        try:
            branch_name.topology.select('name CG and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            cg_atom_index = branch_name.topology.select('name CG and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]

        try:
            branch_name.topology.select('name CD2 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            cd_atom_index = branch_name.topology.select('name CD2 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]

        try:
            branch_name.topology.select('name NE1 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            ne_atom_index = branch_name.topology.select('name NE1 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]

        # cg_atom_index = branch_name.topology.select('name CG and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]
        # cd_atom_index = branch_name.topology.select('name CD2 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]
        # ne_atom_index = branch_name.topology.select('name NE1 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]

        if indexerror_count == 0:
            midpoint_CG_CD_x_coord = (branch_name.xyz[0][cg_atom_index][0] + branch_name.xyz[0][cd_atom_index][0]) / 2
            midpoint_CG_CD_y_coord = (branch_name.xyz[0][cg_atom_index][1] + branch_name.xyz[0][cd_atom_index][1]) / 2
            midpoint_CG_CD_z_coord = (branch_name.xyz[0][cg_atom_index][2] + branch_name.xyz[0][cd_atom_index][2]) / 2
            length_midpoint_CG_CD_to_NE = math.sqrt(((branch_name.xyz[0][ne_atom_index][0] - midpoint_CG_CD_x_coord) ** 2) + ((branch_name.xyz[0][ne_atom_index][1] - midpoint_CG_CD_y_coord) ** 2) + ((branch_name.xyz[0][ne_atom_index][2] - midpoint_CG_CD_z_coord) ** 2))
            NE_dx = (branch_name.xyz[0][ne_atom_index][0] - midpoint_CG_CD_x_coord) / length_midpoint_CG_CD_to_NE
            NE_dy = (branch_name.xyz[0][ne_atom_index][1] - midpoint_CG_CD_y_coord) / length_midpoint_CG_CD_to_NE
            NE_dz = (branch_name.xyz[0][ne_atom_index][2] - midpoint_CG_CD_z_coord) / length_midpoint_CG_CD_to_NE
            NE_H_x_coord = branch_name.xyz[0][ne_atom_index][0] + 0.100 * NE_dx
            NE_H_y_coord = branch_name.xyz[0][ne_atom_index][1] + 0.100 * NE_dy
            NE_H_z_coord = branch_name.xyz[0][ne_atom_index][2] + 0.100 * NE_dz
            #best_possible_H_of_the_donor_neighbor = [[NE_H_x_coord, NE_H_y_coord, NE_H_z_coord]]
            best_possible_H_of_the_donor_neighbor = [NE_H_x_coord, NE_H_y_coord, NE_H_z_coord]

    elif branch_name.topology.atom(z).name == 'NH1' and branch_name.topology.atom(z).residue.name == 'ARG':  # ARG NH1
        try:
            branch_name.xyz[0][branch_name.topology.select('name NE and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p1 = branch_name.xyz[0][branch_name.topology.select('name NE and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

        try:
            branch_name.xyz[0][branch_name.topology.select('name CZ and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p2 = branch_name.xyz[0][branch_name.topology.select('name CZ and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

        try:
            branch_name.xyz[0][branch_name.topology.select('name NH1 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p3 = branch_name.xyz[0][branch_name.topology.select('name NH1 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

        try:
            spX_tetrahedron_center_H_finder(branch_name, z, 1.00, 'sp2')
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p4 = spX_tetrahedron_center_H_finder(branch_name, z, 1.00, 'sp2')

        # p1 = branch_name.xyz[0][branch_name.topology.select('name NE and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        # p2 = branch_name.xyz[0][branch_name.topology.select('name CZ and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        # p3 = branch_name.xyz[0][branch_name.topology.select('name NH1 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        # p4 = spX_tetrahedron_center_H_finder(branch_name, z, 1.00, 'sp2')

        if indexerror_count == 0:
            HH12 = z + 1
            HH11 = z + 2

            radians_of_the_dihedral_to_rotate = calculate_dihedral(p1, p2, p3, p4)

            HH12_xyz = rotate(p4, p2, p3, radians_of_the_dihedral_to_rotate)
            HH11_xyz = rotate(HH12_xyz, p2, p3, pi)

            if HH12 not in branch_name.reduced_topology_not_available_donors:
                best_possible_H_of_the_donor_neighbor = [[HH12, HH12_xyz]]
            if HH11 not in branch_name.reduced_topology_not_available_donors:
                best_possible_H_of_the_donor_neighbor.append([HH11, HH11_xyz])

            if len(flatten(best_possible_H_of_the_donor_neighbor)) == 4:
                best_possible_H_of_the_donor_neighbor = best_possible_H_of_the_donor_neighbor[0]

    elif branch_name.topology.atom(z).name == 'NH2' and branch_name.topology.atom(z).residue.name == 'ARG':  # ARG NH2
        try:
            branch_name.xyz[0][branch_name.topology.select('name NE and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p1 = branch_name.xyz[0][branch_name.topology.select('name NE and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

        try:
            branch_name.xyz[0][branch_name.topology.select('name CZ and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p2 = branch_name.xyz[0][branch_name.topology.select('name CZ and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

        try:
            branch_name.xyz[0][branch_name.topology.select('name NH2 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p3 = branch_name.xyz[0][branch_name.topology.select('name NH2 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

        try:
            spX_tetrahedron_center_H_finder(branch_name, z, 1.00, 'sp2')
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p4 = spX_tetrahedron_center_H_finder(branch_name, z, 1.00, 'sp2')

        # p1 = branch_name.xyz[0][branch_name.topology.select('name NE and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        # p2 = branch_name.xyz[0][branch_name.topology.select('name CZ and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        # p3 = branch_name.xyz[0][branch_name.topology.select('name NH2 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        # p4 = spX_tetrahedron_center_H_finder(branch_name, z, 1.00, 'sp2')

        if indexerror_count == 0:
            # z+3 and z+4 is used instead of z+1 and z+2.
            # if you use z+1 and z+2, it will append wrong atom. Explanation of an example: if NH1 is 2110 and NH2 is 2111, the 2H of NH1 would be 2111 and 2112 while the 2H of NH2 will be 2112 and 2113.
            # notice that you have duplicate of 2112. so, the solution is to make the Hs of NH1 z+1 and z+2 while the Hs of NH2 as z+3 and z+4
            HH21 = z + 3
            HH22 = z + 4

            radians_of_the_dihedral_to_rotate = calculate_dihedral(p1, p2, p3, p4)

            HH21_xyz = rotate(p4, p2, p3, radians_of_the_dihedral_to_rotate)
            HH22_xyz = rotate(HH21_xyz, p2, p3, pi)

            if HH21 not in branch_name.reduced_topology_not_available_donors:
                best_possible_H_of_the_donor_neighbor = [[HH21, HH21_xyz]]
            if HH22 not in branch_name.reduced_topology_not_available_donors:
                best_possible_H_of_the_donor_neighbor.append([HH22, HH22_xyz])

            if len(flatten(best_possible_H_of_the_donor_neighbor)) == 4:
                best_possible_H_of_the_donor_neighbor = best_possible_H_of_the_donor_neighbor[0]

    elif branch_name.topology.atom(z).name == 'NZ' and branch_name.topology.atom(z).residue.name in ['LYS', 'LYN']:  # LYS NZ
        try:
            branch_name.xyz[0][branch_name.topology.select('name CD and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p1 = branch_name.xyz[0][branch_name.topology.select('name CD and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

        try:
            branch_name.xyz[0][branch_name.topology.select('name CE and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p2 = branch_name.xyz[0][branch_name.topology.select('name CE and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

        try:
            branch_name.xyz[0][branch_name.topology.select('name NZ and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p3 = branch_name.xyz[0][branch_name.topology.select('name NZ and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

        try:
            spX_tetrahedron_center_H_finder(branch_name, z, 1.00, 'sp3')
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p4 = spX_tetrahedron_center_H_finder(branch_name, z, 1.00, 'sp3')

        # p1 = branch_name.xyz[0][branch_name.topology.select('name CD and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        # p2 = branch_name.xyz[0][branch_name.topology.select('name CE and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        # p3 = branch_name.xyz[0][branch_name.topology.select('name NZ and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        # p4 = spX_tetrahedron_center_H_finder(branch_name, z, 1.00, 'sp3')

        if indexerror_count == 0:
            radians_of_the_dihedral_to_rotate = calculate_dihedral(p1, p2, p3, p4)
            eclipsed_first_H_atom = rotate(p4, p2, p3, radians_of_the_dihedral_to_rotate)

            HZ1 = z + 1
            HZ2 = z + 2
            HZ3 = z + 3

            HZ1_xyz = rotate(eclipsed_first_H_atom, p2, p3, pi)
            HZ2_xyz = rotate(eclipsed_first_H_atom, p2, p3, pi / 3)
            HZ3_xyz = rotate(eclipsed_first_H_atom, p2, p3, -pi / 3)

            if HZ1 not in branch_name.reduced_topology_not_available_donors:
                best_possible_H_of_the_donor_neighbor = [[HZ1, HZ1_xyz]]
            if HZ2 not in branch_name.reduced_topology_not_available_donors:
                best_possible_H_of_the_donor_neighbor.append([HZ2, HZ2_xyz])
            if HZ3 not in branch_name.reduced_topology_not_available_donors:
                best_possible_H_of_the_donor_neighbor.append([HZ3, HZ3_xyz])

            if len(flatten(best_possible_H_of_the_donor_neighbor)) == 4:
                best_possible_H_of_the_donor_neighbor = best_possible_H_of_the_donor_neighbor[0]

    elif branch_name.topology.atom(z).name == 'OD1' and branch_name.topology.atom(z).residue.name == 'ASN':  # ASN OD1 (since X-ray can't distinguish between N and O - OD1 could potentially be the ND2 atom. only consider both options when unknown ASN neighbored with HIS)
        print('While this heavy atom is the side chain O atom, we are locating 2 H atoms on it because of the 180 degree flip possibility')
        try:
            branch_name.xyz[0][branch_name.topology.select('name CB and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p1 = branch_name.xyz[0][branch_name.topology.select('name CB and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

        try:
            branch_name.xyz[0][branch_name.topology.select('name CG and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p2 = branch_name.xyz[0][branch_name.topology.select('name CG and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

        try:
            branch_name.xyz[0][branch_name.topology.select('name OD1 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p3 = branch_name.xyz[0][branch_name.topology.select('name OD1 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

        try:
            spX_tetrahedron_center_H_finder(branch_name, z, 1.00, 'sp2')
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p4 = spX_tetrahedron_center_H_finder(branch_name, z, 1.00, 'sp2')

        # p1 = branch_name.xyz[0][branch_name.topology.select('name CB and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        # p2 = branch_name.xyz[0][branch_name.topology.select('name CG and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        # p3 = branch_name.xyz[0][branch_name.topology.select('name OD1 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        # p4 = spX_tetrahedron_center_H_finder(branch_name, z, 1.00, 'sp2')

        if indexerror_count == 0:
            HD21 = z + 1
            HD22 = z + 2

            radians_of_the_dihedral_to_rotate = calculate_dihedral(p1, p2, p3, p4)

            HD21_xyz = rotate(p4, p2, p3, radians_of_the_dihedral_to_rotate)
            HD22_xyz = rotate(HD21_xyz, p2, p3, pi)

            best_possible_H_of_the_donor_neighbor = [[HD21, HD21_xyz], [HD22, HD22_xyz]]

    elif branch_name.topology.atom(z).name == 'ND2' and branch_name.topology.atom(z).residue.name == 'ASN':  # ASN ND2
        try:
            branch_name.xyz[0][branch_name.topology.select('name CB and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p1 = branch_name.xyz[0][branch_name.topology.select('name CB and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

        try:
            branch_name.xyz[0][branch_name.topology.select('name CG and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p2 = branch_name.xyz[0][branch_name.topology.select('name CG and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

        try:
            branch_name.xyz[0][branch_name.topology.select('name ND2 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p3 = branch_name.xyz[0][branch_name.topology.select('name ND2 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

        try:
            spX_tetrahedron_center_H_finder(branch_name, z, 1.00, 'sp2')
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p4 = spX_tetrahedron_center_H_finder(branch_name, z, 1.00, 'sp2')

        # p1 = branch_name.xyz[0][branch_name.topology.select('name CB and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        # p2 = branch_name.xyz[0][branch_name.topology.select('name CG and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        # p3 = branch_name.xyz[0][branch_name.topology.select('name ND2 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        # p4 = spX_tetrahedron_center_H_finder(branch_name, z, 1.00, 'sp2')

        if indexerror_count == 0:
            # z+3 and z+4 is used instead of z+1 and z+2.
            # if you use z+1 and z+2, it will append wrong atom. Explanation of an example: if OD1/OE1 is 2110 and ND2/NE2 is 2111, the 2H of OD1/OE1 would be 2111 and 2112 while the 2H of ND2/NE2 will be 2112 and 2113.
            # notice that you have duplicate of 2112. so, the solution is to make the Hs of OD1/OE1 z+1 and z+2 while the Hs of ND2/NE2 as z+3 and z+4
            HD21 = z + 3
            HD22 = z + 4

            radians_of_the_dihedral_to_rotate = calculate_dihedral(p1, p2, p3, p4)

            HD21_xyz = rotate(p4, p2, p3, radians_of_the_dihedral_to_rotate)
            HD22_xyz = rotate(HD21_xyz, p2, p3, pi)

            if HD21 not in branch_name.reduced_topology_not_available_donors:
                best_possible_H_of_the_donor_neighbor = [[HD21, HD21_xyz]]
            if HD22 not in branch_name.reduced_topology_not_available_donors:
                best_possible_H_of_the_donor_neighbor.append([HD22, HD22_xyz])

            if len(flatten(best_possible_H_of_the_donor_neighbor)) == 4:
                best_possible_H_of_the_donor_neighbor = best_possible_H_of_the_donor_neighbor[0]

    elif branch_name.topology.atom(z).name == 'OE1' and branch_name.topology.atom(z).residue.name == 'GLN':  # GLN OE1 (since X-ray can't distinguish between N and O - OE1 could potentially be the NE2 atom. only consider both options when unknown GLN neighbored with HIS)
        print('While this heavy atom is the side chain O atom, we are locating 2 H atoms on it because of the 180 degree flip possibility')
        try:
            branch_name.xyz[0][branch_name.topology.select('name CG and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p1 = branch_name.xyz[0][branch_name.topology.select('name CG and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

        try:
            branch_name.xyz[0][branch_name.topology.select('name CD and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p2 = branch_name.xyz[0][branch_name.topology.select('name CD and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

        try:
            branch_name.xyz[0][branch_name.topology.select('name OE1 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p3 = branch_name.xyz[0][branch_name.topology.select('name OE1 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

        try:
            spX_tetrahedron_center_H_finder(branch_name, z, 1.00, 'sp2')
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p4 = spX_tetrahedron_center_H_finder(branch_name, z, 1.00, 'sp2')

        # p1 = branch_name.xyz[0][branch_name.topology.select('name CG and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        # p2 = branch_name.xyz[0][branch_name.topology.select('name CD and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        # p3 = branch_name.xyz[0][branch_name.topology.select('name OE1 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        # p4 = spX_tetrahedron_center_H_finder(branch_name, z, 1.00, 'sp2')

        if indexerror_count == 0:
            HE21 = z + 1
            HE22 = z + 2

            radians_of_the_dihedral_to_rotate = calculate_dihedral(p1, p2, p3, p4)

            HE21_xyz = rotate(p4, p2, p3, radians_of_the_dihedral_to_rotate)
            HE22_xyz = rotate(HE21_xyz, p2, p3, pi)

            best_possible_H_of_the_donor_neighbor = [[HE21, HE21_xyz], [HE22, HE22_xyz]]

    elif branch_name.topology.atom(z).name == 'NE2' and branch_name.topology.atom(z).residue.name == 'GLN':  # GLN NE2
        try:
            branch_name.xyz[0][branch_name.topology.select('name CG and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p1 = branch_name.xyz[0][branch_name.topology.select('name CG and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

        try:
            branch_name.xyz[0][branch_name.topology.select('name CD and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p2 = branch_name.xyz[0][branch_name.topology.select('name CD and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

        try:
            branch_name.xyz[0][branch_name.topology.select('name NE2 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p3 = branch_name.xyz[0][branch_name.topology.select('name NE2 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

        try:
            spX_tetrahedron_center_H_finder(branch_name, z, 1.00, 'sp2')
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p4 = spX_tetrahedron_center_H_finder(branch_name, z, 1.00, 'sp2')

        # p1 = branch_name.xyz[0][branch_name.topology.select('name CG and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        # p2 = branch_name.xyz[0][branch_name.topology.select('name CD and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        # p3 = branch_name.xyz[0][branch_name.topology.select('name NE2 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        # p4 = spX_tetrahedron_center_H_finder(branch_name, z, 1.00, 'sp2')

        if indexerror_count == 0:
            # z+3 and z+4 is used instead of z+1 and z+2.
            # if you use z+1 and z+2, it will append wrong atom. Explanation of an example: if OD1/OE1 is 2110 and ND2/NE2 is 2111, the 2H of OD1/OE1 would be 2111 and 2112 while the 2H of ND2/NE2 will be 2112 and 2113.
            # notice that you have duplicate of 2112. so, the solution is to make the Hs of OD1/OE1 z+1 and z+2 while the Hs of ND2/NE2 as z+3 and z+4
            HE21 = z + 3
            HE22 = z + 4

            radians_of_the_dihedral_to_rotate = calculate_dihedral(p1, p2, p3, p4)

            HE21_xyz = rotate(p4, p2, p3, radians_of_the_dihedral_to_rotate)
            HE22_xyz = rotate(HE21_xyz, p2, p3, pi)

            if HE21 not in branch_name.reduced_topology_not_available_donors:
                best_possible_H_of_the_donor_neighbor = [[HE21, HE21_xyz]]
            if HE22 not in branch_name.reduced_topology_not_available_donors:
                best_possible_H_of_the_donor_neighbor.append([HE22, HE22_xyz])

            if len(flatten(best_possible_H_of_the_donor_neighbor)) == 4:
                best_possible_H_of_the_donor_neighbor = best_possible_H_of_the_donor_neighbor[0]

    elif branch_name.topology.atom(z).name in ['ND1', 'CE1', 'NE2', 'CD2'] and branch_name.topology.atom(z).residue.name in ['HIS', 'HIE', 'HID', 'HIP']:
        try:
            branch_name.topology.select('(resname HIS or resname HIE or resname HID or resname HIP) and name CG and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            cg_atom_index = branch_name.topology.select('(resname HIS or resname HIE or resname HID or resname HIP) and name CG and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]

        try:
            branch_name.topology.select('(resname HIS or resname HIE or resname HID or resname HIP) and name ND1 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            nd_atom_index = branch_name.topology.select('(resname HIS or resname HIE or resname HID or resname HIP) and name ND1 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]

        try:
            branch_name.topology.select('(resname HIS or resname HIE or resname HID or resname HIP) and name CE1 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            ce_atom_index = branch_name.topology.select('(resname HIS or resname HIE or resname HID or resname HIP) and name CE1 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]

        try:
            branch_name.topology.select('(resname HIS or resname HIE or resname HID or resname HIP) and name NE2 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            ne_atom_index = branch_name.topology.select('(resname HIS or resname HIE or resname HID or resname HIP) and name NE2 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]

        try:
            branch_name.topology.select('(resname HIS or resname HIE or resname HID or resname HIP) and name CD2 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            cd_atom_index = branch_name.topology.select('(resname HIS or resname HIE or resname HID or resname HIP) and name CD2 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]

        # cg_atom_index = branch_name.topology.select('(resname HIS or resname HIE or resname HID or resname HIP) and name CG and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]
        # nd_atom_index = branch_name.topology.select('(resname HIS or resname HIE or resname HID or resname HIP) and name ND1 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]
        # ce_atom_index = branch_name.topology.select('(resname HIS or resname HIE or resname HID or resname HIP) and name CE1 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]
        # ne_atom_index = branch_name.topology.select('(resname HIS or resname HIE or resname HID or resname HIP) and name NE2 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]
        # cd_atom_index = branch_name.topology.select('(resname HIS or resname HIE or resname HID or resname HIP) and name CD2 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]

        if indexerror_count == 0:
            if branch_name.topology.atom(z).name == 'ND1':
                # Hydrogen_1 if to be connected to atom 1 in the ring (ND). find coords of the midpoint between CD and NE.
                midpoint_CD_NE_x_coord = (branch_name.xyz[0][cd_atom_index][0] + branch_name.xyz[0][ne_atom_index][0]) / 2
                midpoint_CD_NE_y_coord = (branch_name.xyz[0][cd_atom_index][1] + branch_name.xyz[0][ne_atom_index][1]) / 2
                midpoint_CD_NE_z_coord = (branch_name.xyz[0][cd_atom_index][2] + branch_name.xyz[0][ne_atom_index][2]) / 2
                length_midpoint_CD_NE_to_ND = math.sqrt((branch_name.xyz[0][nd_atom_index][0] - midpoint_CD_NE_x_coord) ** 2 + (branch_name.xyz[0][nd_atom_index][1] - midpoint_CD_NE_y_coord) ** 2 + (branch_name.xyz[0][nd_atom_index][2] - midpoint_CD_NE_z_coord) ** 2)
                ND_dx = (branch_name.xyz[0][nd_atom_index][0] - midpoint_CD_NE_x_coord) / length_midpoint_CD_NE_to_ND
                ND_dy = (branch_name.xyz[0][nd_atom_index][1] - midpoint_CD_NE_y_coord) / length_midpoint_CD_NE_to_ND
                ND_dz = (branch_name.xyz[0][nd_atom_index][2] - midpoint_CD_NE_z_coord) / length_midpoint_CD_NE_to_ND
                ND_H_x_coord = branch_name.xyz[0][nd_atom_index][0] + 0.095 * ND_dx
                ND_H_y_coord = branch_name.xyz[0][nd_atom_index][1] + 0.095 * ND_dy
                ND_H_z_coord = branch_name.xyz[0][nd_atom_index][2] + 0.095 * ND_dz
                best_possible_H_of_the_donor_neighbor = [ND_H_x_coord, ND_H_y_coord, ND_H_z_coord]

            elif branch_name.topology.atom(z).name == 'CE1':
                # Hydrogen_2 if to be connected to atom 2 in the ring (CE). find coords of the midpoint between CG and CD.
                midpoint_CG_CD_x_coord = (branch_name.xyz[0][cg_atom_index][0] + branch_name.xyz[0][cd_atom_index][0]) / 2
                midpoint_CG_CD_y_coord = (branch_name.xyz[0][cg_atom_index][1] + branch_name.xyz[0][cd_atom_index][1]) / 2
                midpoint_CG_CD_z_coord = (branch_name.xyz[0][cg_atom_index][2] + branch_name.xyz[0][cd_atom_index][2]) / 2
                length_midpoint_CG_CD_to_CE = math.sqrt((branch_name.xyz[0][ce_atom_index][0] - midpoint_CG_CD_x_coord) ** 2 + (branch_name.xyz[0][ce_atom_index][1] - midpoint_CG_CD_y_coord) ** 2 + (branch_name.xyz[0][ce_atom_index][2] - midpoint_CG_CD_z_coord) ** 2)
                CE_dx = (branch_name.xyz[0][ce_atom_index][0] - midpoint_CG_CD_x_coord) / length_midpoint_CG_CD_to_CE
                CE_dy = (branch_name.xyz[0][ce_atom_index][1] - midpoint_CG_CD_y_coord) / length_midpoint_CG_CD_to_CE
                CE_dz = (branch_name.xyz[0][ce_atom_index][2] - midpoint_CG_CD_z_coord) / length_midpoint_CG_CD_to_CE
                CE_H_x_coord = branch_name.xyz[0][ce_atom_index][0] + 0.095 * CE_dx
                CE_H_y_coord = branch_name.xyz[0][ce_atom_index][1] + 0.095 * CE_dy
                CE_H_z_coord = branch_name.xyz[0][ce_atom_index][2] + 0.095 * CE_dz
                best_possible_H_of_the_donor_neighbor = [CE_H_x_coord, CE_H_y_coord, CE_H_z_coord]

            elif branch_name.topology.atom(z).name == 'NE2':
                # Hydrogen_3 if to be connected to atom 3 in the ring (NE). find coords of the midpoint between CG and ND.
                midpoint_CG_ND_x_coord = (branch_name.xyz[0][cg_atom_index][0] + branch_name.xyz[0][nd_atom_index][0]) / 2
                midpoint_CG_ND_y_coord = (branch_name.xyz[0][cg_atom_index][1] + branch_name.xyz[0][nd_atom_index][1]) / 2
                midpoint_CG_ND_z_coord = (branch_name.xyz[0][cg_atom_index][2] + branch_name.xyz[0][nd_atom_index][2]) / 2
                length_midpoint_CG_ND_to_NE = math.sqrt((branch_name.xyz[0][ne_atom_index][0] - midpoint_CG_ND_x_coord) ** 2 + (branch_name.xyz[0][ne_atom_index][1] - midpoint_CG_ND_y_coord) ** 2 + (branch_name.xyz[0][ne_atom_index][2] - midpoint_CG_ND_z_coord) ** 2)
                NE_dx = (branch_name.xyz[0][ne_atom_index][0] - midpoint_CG_ND_x_coord) / length_midpoint_CG_ND_to_NE
                NE_dy = (branch_name.xyz[0][ne_atom_index][1] - midpoint_CG_ND_y_coord) / length_midpoint_CG_ND_to_NE
                NE_dz = (branch_name.xyz[0][ne_atom_index][2] - midpoint_CG_ND_z_coord) / length_midpoint_CG_ND_to_NE
                NE_H_x_coord = branch_name.xyz[0][ne_atom_index][0] + 0.095 * NE_dx
                NE_H_y_coord = branch_name.xyz[0][ne_atom_index][1] + 0.095 * NE_dy
                NE_H_z_coord = branch_name.xyz[0][ne_atom_index][2] + 0.095 * NE_dz
                best_possible_H_of_the_donor_neighbor = [NE_H_x_coord, NE_H_y_coord, NE_H_z_coord]

            elif branch_name.topology.atom(z).name == 'CD2':
                # Hydrogen_4 if to be connected to atom 4 in the ring (CD). find coords of the midpoint between CE and ND.
                midpoint_CE_ND_x_coord = (branch_name.xyz[0][ce_atom_index][0] + branch_name.xyz[0][nd_atom_index][0]) / 2
                midpoint_CE_ND_y_coord = (branch_name.xyz[0][ce_atom_index][1] + branch_name.xyz[0][nd_atom_index][1]) / 2
                midpoint_CE_ND_z_coord = (branch_name.xyz[0][ce_atom_index][2] + branch_name.xyz[0][nd_atom_index][2]) / 2
                length_midpoint_CE_ND_to_CD = math.sqrt((branch_name.xyz[0][cd_atom_index][0] - midpoint_CE_ND_x_coord) ** 2 + (branch_name.xyz[0][cd_atom_index][1] - midpoint_CE_ND_y_coord) ** 2 + (branch_name.xyz[0][cd_atom_index][2] - midpoint_CE_ND_z_coord) ** 2)
                CD_dx = (branch_name.xyz[0][cd_atom_index][0] - midpoint_CE_ND_x_coord) / length_midpoint_CE_ND_to_CD
                CD_dy = (branch_name.xyz[0][cd_atom_index][1] - midpoint_CE_ND_y_coord) / length_midpoint_CE_ND_to_CD
                CD_dz = (branch_name.xyz[0][cd_atom_index][2] - midpoint_CE_ND_z_coord) / length_midpoint_CE_ND_to_CD
                CD_H_x_coord = branch_name.xyz[0][cd_atom_index][0] + 0.095 * CD_dx
                CD_H_y_coord = branch_name.xyz[0][cd_atom_index][1] + 0.095 * CD_dy
                CD_H_z_coord = branch_name.xyz[0][cd_atom_index][2] + 0.095 * CD_dz
                best_possible_H_of_the_donor_neighbor = [CD_H_x_coord, CD_H_y_coord, CD_H_z_coord]

    elif branch_name.topology.atom(z).name in ['OH', 'OG', 'OG1'] and branch_name.topology.atom(z).residue.name in ['TYR', 'SER', 'THR']:  # TYR OH, SER OG, THR OG1
        try:
            spX_tetrahedron_center_H_finder(branch_name, z, 0.95, 'sp3')  # this is the H atom on OH, OG or OG1 atoms. This makes us avoid using reduce/Haad or other H atom prep tools.
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p4 = spX_tetrahedron_center_H_finder(branch_name, z, 0.95, 'sp3')  # this is the H atom on OH, OG or OG1 atoms. This makes us avoid using reduce/Haad or other H atom prep tools.

        # p4 = spX_tetrahedron_center_H_finder(branch_name, z, 0.95, 'sp3')  # this is the H atom on OH, OG or OG1 atoms. This makes us avoid using reduce/Haad or other H atom prep tools.

        if branch_name.topology.atom(z).name == 'OH':  # TYR OH
            try:
                branch_name.xyz[0][branch_name.topology.select('name CE1 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
            except IndexError:
                print("IndexError")
                indexerror_count += 1
            else:
                p1 = branch_name.xyz[0][branch_name.topology.select('name CE1 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]  # CE1 atom

            try:
                branch_name.xyz[0][branch_name.topology.select('name CZ and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
            except IndexError:
                print("IndexError")
                indexerror_count += 1
            else:
                p2 = branch_name.xyz[0][branch_name.topology.select('name CZ and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]  # CZ atom

            try:
                branch_name.xyz[0][branch_name.topology.select('name OH and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
            except IndexError:
                print("IndexError")
                indexerror_count += 1
            else:
                p3 = branch_name.xyz[0][branch_name.topology.select('name OH and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]  # OH atom

            # p1 = branch_name.xyz[0][branch_name.topology.select('name CE1 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]  # CE1 atom
            # p2 = branch_name.xyz[0][branch_name.topology.select('name CZ and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]  # CZ atom
            # p3 = branch_name.xyz[0][branch_name.topology.select('name OH and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]  # OH atom
            # p4 is calculated outside of the if conditions

            if indexerror_count == 0:
                radians_of_the_dihedral_to_rotate = calculate_dihedral(p1, p2, p3, p4)
                first_possible_H_of_TYR_OH = rotate(p4, p2, p3, radians_of_the_dihedral_to_rotate)
                second_possible_H_of_TYR_OH = rotate(first_possible_H_of_TYR_OH, p2, p3, pi)

                possible_H_positions = [first_possible_H_of_TYR_OH, second_possible_H_of_TYR_OH]
                print(possible_H_positions)

        elif branch_name.topology.atom(z).name == 'OG':  # SER OG
            try:
                branch_name.xyz[0][branch_name.topology.select('name CA and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
            except IndexError:
                print("IndexError")
                indexerror_count += 1
            else:
                p1 = branch_name.xyz[0][branch_name.topology.select('name CA and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]  # CA atom

            try:
                branch_name.xyz[0][branch_name.topology.select('name CB and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
            except IndexError:
                print("IndexError")
                indexerror_count += 1
            else:
                p2 = branch_name.xyz[0][branch_name.topology.select('name CB and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]  # CB atom

            try:
                branch_name.xyz[0][branch_name.topology.select('name OG and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
            except IndexError:
                print("IndexError")
                indexerror_count += 1
            else:
                p3 = branch_name.xyz[0][branch_name.topology.select('name OG and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]  # OG atom

            # p1 = branch_name.xyz[0][branch_name.topology.select('name CA and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]  # CA atom
            # p2 = branch_name.xyz[0][branch_name.topology.select('name CB and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]  # CB atom
            # p3 = branch_name.xyz[0][branch_name.topology.select('name OG and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]  # OG atom
            # p4 is calculated outside of the if conditions

            if indexerror_count == 0:
                radians_of_the_dihedral_to_rotate = calculate_dihedral(p1, p2, p3, p4)
                H_positions_if_eclipsed_onto_CA_CB_bond = rotate(p4, p2, p3, radians_of_the_dihedral_to_rotate)

                first_possible_H_of_SER_OH = rotate(H_positions_if_eclipsed_onto_CA_CB_bond, p2, p3, pi)
                second_possible_H_of_SER_OH = rotate(H_positions_if_eclipsed_onto_CA_CB_bond, p2, p3, pi / 3)
                third_possible_H_of_SER_OH = rotate(H_positions_if_eclipsed_onto_CA_CB_bond, p2, p3, -pi / 3)
                first_possible_H_of_SER_OH_minus20 = rotate(first_possible_H_of_SER_OH, p2, p3, -0.371544279)  # while I should use +and- 0.349066 (pi/9 radians) which is equal to 20 degrees, I turned out that with the rounding in between, it gets to average 18.79 degrees instead of 20 so I increased the angle input in radians a bit so it can converge to 20 degrees which I want.
                first_possible_H_of_SER_OH_plus20 = rotate(first_possible_H_of_SER_OH, p2, p3, 0.371544279)
                second_possible_H_of_SER_OH_minus20 = rotate(second_possible_H_of_SER_OH, p2, p3, -0.371544279)
                second_possible_H_of_SER_OH_plus20 = rotate(second_possible_H_of_SER_OH, p2, p3, 0.371544279)
                third_possible_H_of_SER_OH_minus20 = rotate(third_possible_H_of_SER_OH, p2, p3, -0.371544279)
                third_possible_H_of_SER_OH_plus20 = rotate(third_possible_H_of_SER_OH, p2, p3, 0.371544279)

                possible_H_positions = [first_possible_H_of_SER_OH, second_possible_H_of_SER_OH, third_possible_H_of_SER_OH, first_possible_H_of_SER_OH_minus20, first_possible_H_of_SER_OH_plus20, second_possible_H_of_SER_OH_minus20, second_possible_H_of_SER_OH_plus20, third_possible_H_of_SER_OH_minus20, third_possible_H_of_SER_OH_plus20]
                print(possible_H_positions)

        elif branch_name.topology.atom(z).name == 'OG1':  # THR OG1
            try:
                branch_name.xyz[0][branch_name.topology.select('name CA and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
            except IndexError:
                print("IndexError")
                indexerror_count += 1
            else:
                p1 = branch_name.xyz[0][branch_name.topology.select('name CA and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]  # CA atom

            try:
                branch_name.xyz[0][branch_name.topology.select('name CB and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
            except IndexError:
                print("IndexError")
                indexerror_count += 1
            else:
                p2 = branch_name.xyz[0][branch_name.topology.select('name CB and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]  # CB atom

            try:
                branch_name.xyz[0][branch_name.topology.select('name OG1 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
            except IndexError:
                print("IndexError")
                indexerror_count += 1
            else:
                p3 = branch_name.xyz[0][branch_name.topology.select('name OG1 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]  # OG atom

            # p1 = branch_name.xyz[0][branch_name.topology.select('name CA and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]  # CA atom
            # p2 = branch_name.xyz[0][branch_name.topology.select('name CB and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]  # CB atom
            # p3 = branch_name.xyz[0][branch_name.topology.select('name OG1 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]  # OG1 atom
            # p4 is calculated outside of the if conditions

            if indexerror_count == 0:
                radians_of_the_dihedral_to_rotate = calculate_dihedral(p1, p2, p3, p4)
                H_positions_if_eclipsed_onto_CA_CB_bond = rotate(p4, p2, p3, radians_of_the_dihedral_to_rotate)

                first_possible_H_of_THR_OH = rotate(H_positions_if_eclipsed_onto_CA_CB_bond, p2, p3, pi)
                second_possible_H_of_THR_OH = rotate(H_positions_if_eclipsed_onto_CA_CB_bond, p2, p3, pi / 3)
                third_possible_H_of_THR_OH = rotate(H_positions_if_eclipsed_onto_CA_CB_bond, p2, p3, -pi / 3)
                first_possible_H_of_THR_OH_minus20 = rotate(first_possible_H_of_THR_OH, p2, p3, -0.371544279)
                first_possible_H_of_THR_OH_plus20 = rotate(first_possible_H_of_THR_OH, p2, p3, 0.371544279)
                second_possible_H_of_THR_OH_minus20 = rotate(second_possible_H_of_THR_OH, p2, p3, -0.371544279)
                second_possible_H_of_THR_OH_plus20 = rotate(second_possible_H_of_THR_OH, p2, p3, 0.371544279)
                third_possible_H_of_THR_OH_minus20 = rotate(third_possible_H_of_THR_OH, p2, p3, -0.371544279)
                third_possible_H_of_THR_OH_plus20 = rotate(third_possible_H_of_THR_OH, p2, p3, 0.371544279)

                possible_H_positions = [first_possible_H_of_THR_OH, second_possible_H_of_THR_OH, third_possible_H_of_THR_OH, first_possible_H_of_THR_OH_minus20, first_possible_H_of_THR_OH_plus20, second_possible_H_of_THR_OH_minus20, second_possible_H_of_THR_OH_plus20, third_possible_H_of_THR_OH_minus20, third_possible_H_of_THR_OH_plus20]
                print(possible_H_positions)

        if indexerror_count == 0:
            dist_dic = {}
            for ndx, j in enumerate(possible_H_positions):
                dist = math.sqrt(((branch_name.xyz[0][y][0] - j[0]) ** 2) + ((branch_name.xyz[0][y][1] - j[1]) ** 2) + ((branch_name.xyz[0][y][2] - j[2]) ** 2))
                dist_dic[ndx] = dist
            best_possible_H_of_the_donor_neighbor = possible_H_positions[sorted(dist_dic.items(), key=operator.itemgetter(1))[0][0]]

    elif branch_name.topology.atom(z).name in ['OD1', 'OD2'] and branch_name.topology.atom(z).residue.name in ['ASP', 'ASH']:
        print('While this heavy atom is the side chain O atom, we are locating 2 H atoms on it because of the 180 degree flip possibility')
        try:
            branch_name.xyz[0][branch_name.topology.select('name CB and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p1 = branch_name.xyz[0][branch_name.topology.select('name CB and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

        try:
            branch_name.xyz[0][branch_name.topology.select('name CG and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p2 = branch_name.xyz[0][branch_name.topology.select('name CG and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

        # p1 = branch_name.xyz[0][branch_name.topology.select('name CB and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        # p2 = branch_name.xyz[0][branch_name.topology.select('name CG and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

        if branch_name.topology.atom(z).name == 'OD1':
            try:
                branch_name.xyz[0][branch_name.topology.select('name OD1 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
            except IndexError:
                print("IndexError")
                indexerror_count += 1
            else:
                p3 = branch_name.xyz[0][branch_name.topology.select('name OD1 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

            # p3 = branch_name.xyz[0][branch_name.topology.select('name OD1 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        elif branch_name.topology.atom(z).name == 'OD2':
            try:
                branch_name.xyz[0][branch_name.topology.select('name OD2 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
            except IndexError:
                print("IndexError")
                indexerror_count += 1
            else:
                p3 = branch_name.xyz[0][branch_name.topology.select('name OD2 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

            # p3 = branch_name.xyz[0][branch_name.topology.select('name OD2 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

        try:
            spX_tetrahedron_center_H_finder(branch_name, z, 1.00, 'sp2')
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p4 = spX_tetrahedron_center_H_finder(branch_name, z, 1.00, 'sp2')

        # p4 = spX_tetrahedron_center_H_finder(branch_name, z, 1.00, 'sp2')

        if indexerror_count == 0:
            radians_of_the_dihedral_to_rotate = calculate_dihedral(p1, p2, p3, p4)

            H1_xyz = rotate(p4, p2, p3, radians_of_the_dihedral_to_rotate)
            H2_xyz = rotate(H1_xyz, p2, p3, pi)

            best_possible_H_of_the_donor_neighbor = [H1_xyz, H2_xyz]

    elif branch_name.topology.atom(z).name in ['OE1', 'OE2'] and branch_name.topology.atom(z).residue.name in ['GLU', 'GLH']:
        print('While this heavy atom is the side chain O atom, we are locating 2 H atoms on it because of the 180 degree flip possibility')
        try:
            branch_name.xyz[0][branch_name.topology.select('name CG and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p1 = branch_name.xyz[0][branch_name.topology.select('name CG and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

        try:
            branch_name.xyz[0][branch_name.topology.select('name CD and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p2 = branch_name.xyz[0][branch_name.topology.select('name CD and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

        # p1 = branch_name.xyz[0][branch_name.topology.select('name CG and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        # p2 = branch_name.xyz[0][branch_name.topology.select('name CD and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

        if branch_name.topology.atom(z).name == 'OE1':
            try:
                branch_name.xyz[0][branch_name.topology.select('name OE1 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
            except IndexError:
                print("IndexError")
                indexerror_count += 1
            else:
                p3 = branch_name.xyz[0][branch_name.topology.select('name OE1 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

            # p3 = branch_name.xyz[0][branch_name.topology.select('name OE1 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
        elif branch_name.topology.atom(z).name == 'OE2':
            try:
                branch_name.xyz[0][branch_name.topology.select('name OE2 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]
            except IndexError:
                print("IndexError")
                indexerror_count += 1
            else:
                p3 = branch_name.xyz[0][branch_name.topology.select('name OE2 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

            # p3 = branch_name.xyz[0][branch_name.topology.select('name OE2 and resid {}'.format(branch_name.topology.atom(z).residue.index))[0]]

        try:
            spX_tetrahedron_center_H_finder(branch_name, z, 1.00, 'sp2')
        except IndexError:
            print("IndexError")
            indexerror_count += 1
        else:
            p4 = spX_tetrahedron_center_H_finder(branch_name, z, 1.00, 'sp2')

        # p4 = spX_tetrahedron_center_H_finder(branch_name, z, 1.00, 'sp2')

        if indexerror_count == 0:
            radians_of_the_dihedral_to_rotate = calculate_dihedral(p1, p2, p3, p4)

            H1_xyz = rotate(p4, p2, p3, radians_of_the_dihedral_to_rotate)
            H2_xyz = rotate(H1_xyz, p2, p3, pi)

            best_possible_H_of_the_donor_neighbor = [H1_xyz, H2_xyz]

    else:
        # non_std_residues_warning = input("WARNING: The atom to which the H atom is attached is either a non-standard residue or ligand. Press any key to continue. make sure you double check if analysis handled this correctly.")
        print("The atom to which the H atom is attached is either a non-standard residue or ligand.\n"
              "The script will use reduce's output to find the H atom.\n")

        het_atm_dic_code = branch_name.topology.atom(z).residue.name + '_' + branch_name.topology.atom(z).name

        try:
            branch_name.het_atm_dict[het_atm_dic_code]
        except KeyError:
            print("KeyError")
            print("This atom has no H on it in the reduce dictionary")
            het_atm_hydrogens = []
        else:
            het_atm_hydrogens = branch_name.het_atm_dict[het_atm_dic_code]
            print("Neighbor atom is {} and its attached H(s) could be {}".format(het_atm_dic_code, het_atm_hydrogens))

        possible_H_positions = []
        for hydrogen_name in het_atm_hydrogens:
            try:
                branch_name.h_topology.select('name "{}" and resid {}'.format(hydrogen_name, branch_name.topology.atom(z).residue.index))[0]
            except IndexError:
                print("IndexError")
            else:
                hydrogen_coords = branch_name.h_xyz[0][branch_name.h_topology.select('name "{}" and resid {}'.format(hydrogen_name, branch_name.topology.atom(z).residue.index))[0]]
                possible_H_positions.append(list(hydrogen_coords))

        best_possible_H_of_the_donor_neighbor = possible_H_positions

        if len(flatten(best_possible_H_of_the_donor_neighbor)) == 3:
            best_possible_H_of_the_donor_neighbor = best_possible_H_of_the_donor_neighbor[0]

    if indexerror_count != 0 and len(best_possible_H_of_the_donor_neighbor) == 0:
        print("       It seems that there was one or more IndexError seen (if a necessary atom for the calculation is missing, H position calculation would not continue)")
        print("       best_possible_H_of_the_donor_neighbor is {}".format(best_possible_H_of_the_donor_neighbor))
        print("       We will give it another chance by letting reduce find the H atom on that atom")

        het_atm_dic_code = branch_name.topology.atom(z).residue.name + '_' + branch_name.topology.atom(z).name

        try:
            branch_name.het_atm_dict[het_atm_dic_code]
        except KeyError:
            print("KeyError")
            print("This atom has no H on it in the reduce dictionary")
            het_atm_hydrogens = []
        else:
            het_atm_hydrogens = branch_name.het_atm_dict[het_atm_dic_code]
            print("Neighbor atom is {} and its attached H(s) could be {}".format(het_atm_dic_code, het_atm_hydrogens))

        possible_H_positions = []
        for hydrogen_name in het_atm_hydrogens:
            try:
                branch_name.h_topology.select('name "{}" and resid {}'.format(hydrogen_name, branch_name.topology.atom(z).residue.index))[0]
            except IndexError:
                print("IndexError")
            else:
                hydrogen_coords = branch_name.h_xyz[0][branch_name.h_topology.select('name "{}" and resid {}'.format(hydrogen_name, branch_name.topology.atom(z).residue.index))[0]]
                possible_H_positions.append(list(hydrogen_coords))

        best_possible_H_of_the_donor_neighbor = possible_H_positions

        if len(flatten(best_possible_H_of_the_donor_neighbor)) == 3:
            best_possible_H_of_the_donor_neighbor = best_possible_H_of_the_donor_neighbor[0]

    if print_atom_line == 1:
        # print the PDB atom line in case we want to visualize it.
        temp_atm_nmbr_in_pdb = z + 1
        print('H atom line is:')
        if len(flatten(best_possible_H_of_the_donor_neighbor)) > 4:
            for i in best_possible_H_of_the_donor_neighbor:
                if len(flatten(i)) == 3:
                    best_possible_H_of_the_donor_neighbor_in_A = [i[0] * 10, i[1] * 10, i[2] * 10]

                elif len(flatten(i)) == 4:
                    best_possible_H_of_the_donor_neighbor_in_A = [i[1][0] * 10, i[1][1] * 10, i[1][2] * 10]

                atom_name = "XX"
                residue_info_1 = " {} X".format(branch_name.topology.atom(z).residue.name)
                residue_info_2 = branch_name.topology.atom(z).residue.index + 1
                rest_of_line = '  1.00 00.00          '
                print("{:6s}{:5d} {:^4s}{}{:4d}    {:8.3f}{:8.3f}{:8.3f}{}{}".format("ATOM", temp_atm_nmbr_in_pdb, atom_name, residue_info_1, residue_info_2, best_possible_H_of_the_donor_neighbor_in_A[0], best_possible_H_of_the_donor_neighbor_in_A[1], best_possible_H_of_the_donor_neighbor_in_A[2], rest_of_line, ' H  '))
                temp_atm_nmbr_in_pdb += 1

        elif len(flatten(best_possible_H_of_the_donor_neighbor)) == 3 or len(flatten(best_possible_H_of_the_donor_neighbor)) == 4:
            if len(flatten(best_possible_H_of_the_donor_neighbor)) == 3:
                best_possible_H_of_the_donor_neighbor_in_A = [best_possible_H_of_the_donor_neighbor[0] * 10, best_possible_H_of_the_donor_neighbor[1] * 10, best_possible_H_of_the_donor_neighbor[2] * 10]
            elif len(flatten(best_possible_H_of_the_donor_neighbor)) == 4:
                best_possible_H_of_the_donor_neighbor_in_A = [best_possible_H_of_the_donor_neighbor[1][0] * 10, best_possible_H_of_the_donor_neighbor[1][1] * 10, best_possible_H_of_the_donor_neighbor[1][2] * 10]

            atom_name = "XX"
            residue_info_1 = " {} X".format(branch_name.topology.atom(z).residue.name)
            residue_info_2 = branch_name.topology.atom(z).residue.index + 1
            rest_of_line = '  1.00 00.00          '
            print("{:6s}{:5d} {:^4s}{}{:4d}    {:8.3f}{:8.3f}{:8.3f}{}{}".format("ATOM", temp_atm_nmbr_in_pdb, atom_name, residue_info_1, residue_info_2, best_possible_H_of_the_donor_neighbor_in_A[0], best_possible_H_of_the_donor_neighbor_in_A[1], best_possible_H_of_the_donor_neighbor_in_A[2], rest_of_line, ' H  '))
            temp_atm_nmbr_in_pdb += 1

        else:
            print("       No H positions were able to get calculated for this neighbor at all.")

    return best_possible_H_of_the_donor_neighbor


##############################################################


##############################################################
def filter_out_non_hbond_asp_glu_pair_interaction(branch_name, interaction_pair):
    # there will be an angle filter to filter out non H-bond interaction.
    # There are two angles calculated here: 1) acc_don_donH angle 2) acc_H_don angle needs to be bigger than 90 degrees
    print('########################')
    new_neighbor_list = []

    the_atom_index_whose_neighbors_we_are_filtering = interaction_pair[0]
    neighbor = interaction_pair[1]

    acc_donor_donorH_angles = []
    acc_H_don_angles = []
    # thus, this makes it very easy to get the angles without repetitions. heavy_atom_1 is always going to be the atom that has the H attached to it
    heavy_atom_1 = the_atom_index_whose_neighbors_we_are_filtering  # the atom which has the list of the neighbor
    heavy_atom_1_coords = branch_name.xyz[0][heavy_atom_1]

    heavy_atom_2 = neighbor  # the neighbor is accepting, thus we will find the H atom(s) on heavy_atom_1
    heavy_atom_2_coords = branch_name.xyz[0][heavy_atom_2]

    mediating_hydrogen_xyz_coords = coords_of_don_neighbor_atom(branch_name, heavy_atom_1, heavy_atom_2)

    for j in mediating_hydrogen_xyz_coords:
        vector_1 = heavy_atom_2_coords - heavy_atom_1_coords
        vector_2 = j - heavy_atom_1_coords
        vector1_vector_2_cosine_angle = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
        acc_donor_donorH_angle_in_radians = np.arccos(vector1_vector_2_cosine_angle)
        acc_donor_donorH_angle_in_degrees = np.degrees(acc_donor_donorH_angle_in_radians)
        acc_donor_donorH_angles.append(acc_donor_donorH_angle_in_degrees)

        vector_1 = heavy_atom_1_coords - j
        vector_2 = heavy_atom_2_coords - j
        vector1_vector_2_cosine_angle = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
        acc_H_don_angle_in_radians = np.arccos(vector1_vector_2_cosine_angle)
        acc_H_don_angle_in_degrees = np.degrees(acc_H_don_angle_in_radians)
        acc_H_don_angles.append(acc_H_don_angle_in_degrees)

    for z in np.arange(len(acc_H_don_angles)):
        if acc_H_don_angles[z] >= 90 and neighbor not in new_neighbor_list:
            new_neighbor_list.append(neighbor)

    if len(new_neighbor_list) == 0:
        filter_it_out = True
        print('Neighbor atom {} ({}) of this neighbor list isn\'t making any H-bond due to bad X-H---X angles.'.format(neighbor, branch_name.topology.atom(neighbor)))
        print('          acc-H-donor angle(s) is {}'.format(acc_H_don_angles))
        # print('          acc-donor-donorH angle(s) is {}'.format(acc_donor_donorH_angles))
        print('          We will delete this pair interaction ({})\n'.format(interaction_pair))
    else:
        filter_it_out = False
        print('Neighbor atom {} ({}) of this neighbor list is making an H-bond.'.format(neighbor, branch_name.topology.atom(neighbor)))
        print('          acc-H-donor angle(s) is {}'.format(acc_H_don_angles))
        # print('          acc-donor-donorH angle(s) is {}'.format(acc_donor_donorH_angles))
        print('          We will keep this pair interaction ({})\n'.format(interaction_pair))

    print('########################')
    return filter_it_out


##############################################################


##############################################################
def filter_out_non_hbond_lys_arg_pair_interaction(branch_name, interaction_pair):
    # there will be an angle filter to filter out non H-bond interaction.
    # There are two angles calculated here: 1) acc_don_donH angle 2) acc_H_don angle needs to be bigger than 90 degrees
    print('########################')
    new_neighbor_list = []

    the_atom_index_whose_neighbors_we_are_filtering = interaction_pair[0]
    neighbor = interaction_pair[1]

    acc_donor_donorH_angles = []
    acc_H_don_angles = []
    # thus, this makes it very easy to get the angles without a repetitions. heavy_atom_1 is always going to be the atom that has the H attached to it
    heavy_atom_1 = the_atom_index_whose_neighbors_we_are_filtering  # the atom which has the list of the neighbor
    heavy_atom_1_coords = branch_name.xyz[0][heavy_atom_1]

    heavy_atom_2 = neighbor  # the neighbor is accepting, thus we will find the H atom(s) on heavy_atom_1
    heavy_atom_2_coords = branch_name.xyz[0][heavy_atom_2]

    mediating_hydrogen_xyz_coords = coords_of_don_neighbor_atom(branch_name, heavy_atom_1, heavy_atom_2)

    mediating_hydrogen_xyz_coords = [item[1] for item in mediating_hydrogen_xyz_coords]

    for j in mediating_hydrogen_xyz_coords:
        vector_1 = heavy_atom_2_coords - heavy_atom_1_coords
        vector_2 = j - heavy_atom_1_coords
        vector1_vector_2_cosine_angle = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
        acc_donor_donorH_angle_in_radians = np.arccos(vector1_vector_2_cosine_angle)
        acc_donor_donorH_angle_in_degrees = np.degrees(acc_donor_donorH_angle_in_radians)
        acc_donor_donorH_angles.append(acc_donor_donorH_angle_in_degrees)

        vector_1 = heavy_atom_1_coords - j
        vector_2 = heavy_atom_2_coords - j
        vector1_vector_2_cosine_angle = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
        acc_H_don_angle_in_radians = np.arccos(vector1_vector_2_cosine_angle)
        acc_H_don_angle_in_degrees = np.degrees(acc_H_don_angle_in_radians)
        acc_H_don_angles.append(acc_H_don_angle_in_degrees)

    for z in np.arange(len(acc_H_don_angles)):
        if acc_H_don_angles[z] >= 90 and neighbor not in new_neighbor_list:
            new_neighbor_list.append(neighbor)

    if len(new_neighbor_list) == 0:
        filter_it_out = True
        print('Neighbor atom {} ({}) of this neighbor list isn\'t making any H-bond due to bad X-H---X angles.'.format(neighbor, branch_name.topology.atom(neighbor)))
        print('          acc-H-donor angle(s) is {}'.format(acc_H_don_angles))
        # print('          acc-donor-donorH angle(s) is {}'.format(acc_donor_donorH_angles))
        print('          We will delete this pair interaction ({})\n'.format(interaction_pair))
    else:
        filter_it_out = False
        print('Neighbor atom {} ({}) of this neighbor list is making an H-bond.'.format(neighbor, branch_name.topology.atom(neighbor)))
        print('          acc-H-donor angle(s) is {}'.format(acc_H_don_angles))
        # print('          acc-donor-donorH angle(s) is {}'.format(acc_donor_donorH_angles))
        print('          We will keep this pair interaction ({})\n'.format(interaction_pair))

    print('########################')
    return filter_it_out

##############################################################


##############################################################
def filter_out_non_hbond_neighbors(branch_name, atom_neighbor_list, the_atom_index_whose_neighbors_we_are_filtering):
    # there will be an angle filter to filter out non H-bond interaction.
    # There are two angles calculated here: 1) acc_don_donH angle 2) acc_H_don angle needs to be bigger than 90 degrees
    print('########################')
    new_neighbor_list = []
    for neighbor in atom_neighbor_list:
        acc_donor_donorH_angles = []
        acc_H_don_angles = []
        # thus, this makes it very easy to get the angles without a repetitions. heavy_atom_1 is always going to be the atom that has the H attached to it
        if (neighbor in branch_name.known_acc and neighbor not in branch_name.known_don) or (neighbor in branch_name.known_don and neighbor not in branch_name.known_acc):
            if (neighbor in branch_name.known_acc) and (neighbor not in branch_name.known_don):
                heavy_atom_1 = the_atom_index_whose_neighbors_we_are_filtering  # the atom which has the list of the neighbor
                heavy_atom_1_coords = branch_name.xyz[0][heavy_atom_1]

                heavy_atom_2 = neighbor  # the neighbor is accepting, thus we will find the H atom(s) on heavy_atom_1
                heavy_atom_2_coords = branch_name.xyz[0][heavy_atom_2]

                mediating_hydrogen_xyz_coords = coords_of_don_neighbor_atom(branch_name, heavy_atom_1, heavy_atom_2)

                if len(flatten(mediating_hydrogen_xyz_coords)) == 3:
                    mediating_hydrogen_xyz_coords = [mediating_hydrogen_xyz_coords]
                elif len(flatten(mediating_hydrogen_xyz_coords)) == 4:  # here we just get rid of the first item which is the index of the H atoms. we only keep the three coords
                    mediating_hydrogen_xyz_coords = [mediating_hydrogen_xyz_coords[1]]
                elif len(flatten(mediating_hydrogen_xyz_coords)) in [8, 12]:  # here we do the same - we just keep coordinates in a list of lists
                    mediating_hydrogen_xyz_coords = [item[1] for item in mediating_hydrogen_xyz_coords]

            elif (neighbor in branch_name.known_don) and (neighbor not in branch_name.known_acc):
                heavy_atom_1 = neighbor  # the neighbor is donating, thus we will find the H atom(s) on heavy_atom_1
                heavy_atom_1_coords = branch_name.xyz[0][heavy_atom_1]

                heavy_atom_2 = the_atom_index_whose_neighbors_we_are_filtering  # the atom which has the list of the neighbor
                heavy_atom_2_coords = branch_name.xyz[0][heavy_atom_2]

                mediating_hydrogen_xyz_coords = coords_of_don_neighbor_atom(branch_name, heavy_atom_1, heavy_atom_2)

                if len(flatten(mediating_hydrogen_xyz_coords)) == 3:
                    mediating_hydrogen_xyz_coords = [mediating_hydrogen_xyz_coords]
                elif len(flatten(mediating_hydrogen_xyz_coords)) == 4:  # here we just get rid of the first item which is the index of the H atoms. we only keep the three coords
                    mediating_hydrogen_xyz_coords = [mediating_hydrogen_xyz_coords[1]]
                elif len(flatten(mediating_hydrogen_xyz_coords)) in [8, 12]:  # here we do the same - we just keep coordinates in a list of lists
                    mediating_hydrogen_xyz_coords = [item[1] for item in mediating_hydrogen_xyz_coords]

            for j in mediating_hydrogen_xyz_coords:
                vector_1 = heavy_atom_2_coords - heavy_atom_1_coords
                vector_2 = j - heavy_atom_1_coords
                vector1_vector_2_cosine_angle = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
                acc_donor_donorH_angle_in_radians = np.arccos(vector1_vector_2_cosine_angle)
                acc_donor_donorH_angle_in_degrees = np.degrees(acc_donor_donorH_angle_in_radians)
                acc_donor_donorH_angles.append(acc_donor_donorH_angle_in_degrees)

                vector_1 = heavy_atom_1_coords - j
                vector_2 = heavy_atom_2_coords - j
                vector1_vector_2_cosine_angle = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
                acc_H_don_angle_in_radians = np.arccos(vector1_vector_2_cosine_angle)
                acc_H_don_angle_in_degrees = np.degrees(acc_H_don_angle_in_radians)
                acc_H_don_angles.append(acc_H_don_angle_in_degrees)

        elif (neighbor not in branch_name.known_acc and neighbor not in branch_name.known_don and neighbor not in branch_name.grease) or (neighbor in branch_name.known_acc and neighbor in branch_name.known_don):

            ######################## the donating portion ########################
            heavy_atom_1 = neighbor  # the neighbor is donating, thus we will find the H atom(s) on heavy_atom_1
            heavy_atom_1_coords = branch_name.xyz[0][heavy_atom_1]

            heavy_atom_2 = the_atom_index_whose_neighbors_we_are_filtering  # the atom which has the list of the neighbor
            heavy_atom_2_coords = branch_name.xyz[0][heavy_atom_2]

            mediating_hydrogen_xyz_coords_donating_part = coords_of_don_neighbor_atom(branch_name, heavy_atom_1, heavy_atom_2)

            if len(flatten(mediating_hydrogen_xyz_coords_donating_part)) == 3:
                mediating_hydrogen_xyz_coords_donating_part = [mediating_hydrogen_xyz_coords_donating_part]
            elif len(flatten(mediating_hydrogen_xyz_coords_donating_part)) == 4:  # here we just get rid of the first item which is the index of the H atoms. we only keep the three coords
                mediating_hydrogen_xyz_coords_donating_part = [mediating_hydrogen_xyz_coords_donating_part[1]]
            elif len(flatten(mediating_hydrogen_xyz_coords_donating_part)) in [8, 12]:  # here we do the same - we just keep coordinates in a list of lists
                mediating_hydrogen_xyz_coords_donating_part = [item[1] for item in mediating_hydrogen_xyz_coords_donating_part]

            for j in mediating_hydrogen_xyz_coords_donating_part:
                vector_1 = heavy_atom_2_coords - heavy_atom_1_coords
                vector_2 = j - heavy_atom_1_coords
                vector1_vector_2_cosine_angle = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
                acc_donor_donorH_angle_in_radians = np.arccos(vector1_vector_2_cosine_angle)
                acc_donor_donorH_angle_in_degrees = np.degrees(acc_donor_donorH_angle_in_radians)
                acc_donor_donorH_angles.append(acc_donor_donorH_angle_in_degrees)

                vector_1 = heavy_atom_1_coords - j
                vector_2 = heavy_atom_2_coords - j
                vector1_vector_2_cosine_angle = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
                acc_H_don_angle_in_radians = np.arccos(vector1_vector_2_cosine_angle)
                acc_H_don_angle_in_degrees = np.degrees(acc_H_don_angle_in_radians)
                acc_H_don_angles.append(acc_H_don_angle_in_degrees)
            ######################################################################

            ######################## the accepting portion ########################
            heavy_atom_1 = the_atom_index_whose_neighbors_we_are_filtering  # the atom which has the list of the neighbor
            heavy_atom_1_coords = branch_name.xyz[0][heavy_atom_1]

            heavy_atom_2 = neighbor  # the neighbor is accepting, thus we will find the H atom(s) on heavy_atom_1
            heavy_atom_2_coords = branch_name.xyz[0][heavy_atom_2]

            mediating_hydrogen_xyz_coords_accepting_part = coords_of_don_neighbor_atom(branch_name, heavy_atom_1, heavy_atom_2)

            if len(flatten(mediating_hydrogen_xyz_coords_accepting_part)) == 3:
                mediating_hydrogen_xyz_coords_accepting_part = [mediating_hydrogen_xyz_coords_accepting_part]
            elif len(flatten(mediating_hydrogen_xyz_coords_accepting_part)) == 4:  # here we just get rid of the first item which is the index of the H atoms. we only keep the three coords
                mediating_hydrogen_xyz_coords_accepting_part = [mediating_hydrogen_xyz_coords_accepting_part[1]]
            elif len(flatten(mediating_hydrogen_xyz_coords_accepting_part)) in [8, 12]:  # here we do the same - we just keep coordinates in a list of lists
                mediating_hydrogen_xyz_coords_accepting_part = [item[1] for item in mediating_hydrogen_xyz_coords_accepting_part]

            for j in mediating_hydrogen_xyz_coords_accepting_part:
                vector_1 = heavy_atom_2_coords - heavy_atom_1_coords
                vector_2 = j - heavy_atom_1_coords
                vector1_vector_2_cosine_angle = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
                acc_donor_donorH_angle_in_radians = np.arccos(vector1_vector_2_cosine_angle)
                acc_donor_donorH_angle_in_degrees = np.degrees(acc_donor_donorH_angle_in_radians)
                acc_donor_donorH_angles.append(acc_donor_donorH_angle_in_degrees)

                vector_1 = heavy_atom_1_coords - j
                vector_2 = heavy_atom_2_coords - j
                vector1_vector_2_cosine_angle = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
                acc_H_don_angle_in_radians = np.arccos(vector1_vector_2_cosine_angle)
                acc_H_don_angle_in_degrees = np.degrees(acc_H_don_angle_in_radians)
                acc_H_don_angles.append(acc_H_don_angle_in_degrees)
            ######################################################################

        for z in np.arange(len(acc_H_don_angles)):
            if acc_H_don_angles[z] >= 90 and neighbor not in new_neighbor_list:
                new_neighbor_list.append(neighbor)

        if neighbor not in new_neighbor_list:
            print('Neighbor atom {} ({}) of this neighbor list isn\'t making any H-bond due to bad angles.'.format(neighbor, branch_name.topology.atom(neighbor)))
            #print('          acc-donor-donorH angle(s) is {}'.format(acc_donor_donorH_angles))
            print('          acc-H-donor angle(s) is {}'.format(acc_H_don_angles))
            print('          We will delete from the neighbor list\n')
        else:
            print('Neighbor atom {} ({}) of this neighbor list is making an H-bond.'.format(neighbor, branch_name.topology.atom(neighbor)))
            #print('          acc-donor-donorH angle(s) is {}'.format(acc_donor_donorH_angles))
            print('          acc-H-donor angle(s) is {}'.format(acc_H_don_angles))
            print('          We will keep it in the neighbor list\n')

    print('########################')
    return new_neighbor_list


##############################################################


##############################################################
def run_second_and_subsequent_passes_on_unknowns(branch_name):
    second_and_subsequent_passes = 0
    while (second_and_subsequent_passes == 0 or initial_number_of_unknowns_before_pass != final_number_of_unknowns_after_pass) and len(branch_name.unknown_residues) != 0:  # and second_and_subsequent_passes <= len(branch_name.unknown_residues):  # iterations will continue ifeither all unknowns are identified (for which all values of yet_unknown are converted to 0 - knowns
        second_and_subsequent_passes += 1
        initial_number_of_unknowns_before_pass = len(branch_name.unknown_residues)
        print('\n#################################################################################################################################')
        print('####################################################### RUNNING PASS # {} ########################################################'.format(second_and_subsequent_passes))
        print('####################################### Number of unknown residues before this pass is {} #######################################'.format(len(branch_name.unknown_residues)))
        print('#################################################################################################################################\n')

        for res_indx in branch_name.unknown_residues:
            if branch_name.topology.residue(res_indx).name in ['ASN', 'GLN']:
                # Uncharged residues Check
                ########################
                # Check ASN and GLN
                ########################
                print('This unknown residue is an uncharged residue (ASN/GLN):')
                print(branch_name.topology.residue(res_indx))

                uncharged_atm_ndx_O = branch_name.topology.select('((resname ASN and name OD1) or (resname GLN and name OE1)) and resid {}'.format(res_indx))
                uncharged_atm_ndx_N = branch_name.topology.select('((resname ASN and name ND2) or (resname GLN and name NE2)) and resid {}'.format(res_indx))

                uncharged_bond = len(uncharged_atm_ndx_O)

                if len(uncharged_atm_ndx_O) != 0 and len(uncharged_atm_ndx_N) != 0:
                    asnglnRname = branch_name.topology.atom(uncharged_atm_ndx_O[0]).residue
                elif len(uncharged_atm_ndx_O) == 0 or len(uncharged_atm_ndx_N) == 0:
                    print('It looks like this residue is missing one or more of the side chain amide atoms')
                    print('skipping this residue from the analysis and removing its index from the unknown_residues list')
                    # delete known residues from the unknown residues list
                    branch_name.unknown_residues = np.delete(branch_name.unknown_residues, np.where(branch_name.unknown_residues == res_indx)[0][0])
                    print('############################################################')

                asngln = defaultdict(dict)

                asngln_atm_nm = {0: "O1", 1: "O2", 2: "N1", 3: "N2"}

                for i in np.arange(uncharged_bond):
                    unchargedndx_O = md.compute_neighbors(branch_name.structure, h_bond_heavy_atm_cutoff, np.asarray([uncharged_atm_ndx_O[i]]), branch_name.allpos_prot_don_acc, periodic=False)[0]
                    unchargedndx_N = md.compute_neighbors(branch_name.structure, h_bond_heavy_atm_cutoff, np.asarray([uncharged_atm_ndx_N[i]]), branch_name.allpos_prot_don_acc, periodic=False)[0]

                    print('Atoms {} and {} are {} and {} of {} (res_indx {}), respectively.'.format(uncharged_atm_ndx_O[i], uncharged_atm_ndx_N[i], branch_name.topology.atom(uncharged_atm_ndx_O[i]).name, branch_name.topology.atom(uncharged_atm_ndx_N[i]).name, branch_name.topology.atom(uncharged_atm_ndx_O[i]).residue, res_indx))

                    asngln_atm_ndx = np.empty((0,), dtype=int)
                    asngln_atm_ndx = np.append(asngln_atm_ndx, (uncharged_atm_ndx_O[i], uncharged_atm_ndx_O[i], uncharged_atm_ndx_N[i], uncharged_atm_ndx_N[i]))

                    # Delete side chain O and N atoms from the neighbor list of each other.
                    for j in asngln_atm_ndx:
                        if j in unchargedndx_O:
                            unchargedndx_O = np.delete(unchargedndx_O, np.where(unchargedndx_O == j)[0][0])
                        if j in unchargedndx_N:
                            unchargedndx_N = np.delete(unchargedndx_N, np.where(unchargedndx_N == j)[0][0])

                    print('Initial neighbors list of side chain O atom: {}'.format(unchargedndx_O))
                    print('Initial neighbors list of side chain N atom: {}'.format(unchargedndx_N))
                    print('########################')

                    # Delete from neighbor lists of side chain O and N of unknown_residues or any neighbor that is not available anymore to H-bond. those can be:
                    # 1 - backbone N or O that is not available anymore for H-bond
                    # OR 2 - side chain atoms whose residue is not in unknown_residues list (means they were already identified but no longer able to H-bond as they were used already).
                    for k in unchargedndx_O:
                        if k not in branch_name.known_don and k not in branch_name.known_acc and ((branch_name.topology.atom(k).name == 'N' or branch_name.topology.atom(k).name == 'O') or ((branch_name.topology.atom(k).name != 'N' or branch_name.topology.atom(k).name != 'O') and branch_name.topology.atom(k).residue.index not in branch_name.unknown_residues)):
                            unchargedndx_O = np.delete(unchargedndx_O, np.where(unchargedndx_O == k)[0][0])

                    for k in unchargedndx_N:
                        if k not in branch_name.known_don and k not in branch_name.known_acc and ((branch_name.topology.atom(k).name == 'N' or branch_name.topology.atom(k).name == 'O') or ((branch_name.topology.atom(k).name != 'N' or branch_name.topology.atom(k).name != 'O') and branch_name.topology.atom(k).residue.index not in branch_name.unknown_residues)):
                            unchargedndx_N = np.delete(unchargedndx_N, np.where(unchargedndx_N == k)[0][0])

                    print('{} (Residue index {})\'s neighbors of O and N side chain atoms of this residue after deleting unavailable H-bonded neighbors:):'.format(branch_name.topology.atom(uncharged_atm_ndx_O[0]).residue, branch_name.topology.atom(uncharged_atm_ndx_O[0]).residue.index))
                    print('Neighbors list of side chain O atom: {}'.format(unchargedndx_O))
                    print('Neighbors list of side chain N atom: {}'.format(unchargedndx_N))
                    print('########################')

                    ############################################################
                    # filter out all non H bond interactions neighbors for which distance between the heavy atoms is 2.5 A or less
                    for k in unchargedndx_O:
                        dist_O_neighbor = math.sqrt(((10*branch_name.xyz[0][k][0] - 10*branch_name.xyz[0][uncharged_atm_ndx_O[i]][0]) ** 2) + ((10*branch_name.xyz[0][k][1] - 10*branch_name.xyz[0][uncharged_atm_ndx_O[i]][1]) ** 2) + ((10*branch_name.xyz[0][k][2] - 10*branch_name.xyz[0][uncharged_atm_ndx_O[i]][2]) ** 2))
                        if dist_O_neighbor < 2.5:
                            unchargedndx_O = np.delete(unchargedndx_O, np.where(unchargedndx_O == k)[0][0])

                    for k in unchargedndx_N:
                        dist_N_neighbor = math.sqrt(((10*branch_name.xyz[0][k][0] - 10*branch_name.xyz[0][uncharged_atm_ndx_N[i]][0]) ** 2) + ((10*branch_name.xyz[0][k][1] - 10*branch_name.xyz[0][uncharged_atm_ndx_N[i]][1]) ** 2) + ((10*branch_name.xyz[0][k][2] - 10*branch_name.xyz[0][uncharged_atm_ndx_N[i]][2]) ** 2))
                        if dist_N_neighbor < 2.5:
                            unchargedndx_N = np.delete(unchargedndx_N, np.where(unchargedndx_N == k)[0][0])
                    print('These are the list of neighbors of O and N side chain atoms of this residue after filtering out neighbors that can not make H-bonds due to bad distance (anything below 2.5 A distance is dropped):')
                    print('Neighbors list of side chain O atom: {}'.format(unchargedndx_O))
                    print('Neighbors list of side chain N atom: {}'.format(unchargedndx_N))
                    print('########################')

                    ############################################################
                    # We filter out all non H bond interactions neighbors
                    unchargedndx_O = np.array(filter_out_non_hbond_neighbors(branch_name, unchargedndx_O, uncharged_atm_ndx_O[0]))
                    unchargedndx_N = np.array(filter_out_non_hbond_neighbors(branch_name, unchargedndx_N, uncharged_atm_ndx_N[0]))

                    print('These are the list of neighbors of O and N side chain atoms of this residue after filtering out neighbors that can not make H-bonds due to bad angles:')
                    print('Neighbors list of side chain O atom: {}'.format(unchargedndx_O))
                    print('Neighbors list of side chain N atom: {}'.format(unchargedndx_N))
                    print('########################')
                    ############################################################

                    # Here, I want to categorize neighbors of each of those potential H atoms with distance cutoff between the H and the neighbor being 2.6 A since O/N---H bond is 1 A and the heavy to heavy atom cutoff is 3.6
                    uncharged_atm_ndx_O_H1_coords = coords_of_don_neighbor_atom(branch_name, uncharged_atm_ndx_O[0], 0)[0][1]
                    uncharged_atm_ndx_O_H2_coords = coords_of_don_neighbor_atom(branch_name, uncharged_atm_ndx_O[0], 0)[1][1]
                    uncharged_atm_ndx_N_H1_coords = coords_of_don_neighbor_atom(branch_name, uncharged_atm_ndx_N[0], 0)[0][1]
                    uncharged_atm_ndx_N_H2_coords = coords_of_don_neighbor_atom(branch_name, uncharged_atm_ndx_N[0], 0)[1][1]

                    unchargedndx_O_H1 = []
                    unchargedndx_O_H2 = []
                    for k in unchargedndx_O:
                        dist_O_H1_neighbor = math.sqrt(((branch_name.xyz[0][k][0] - uncharged_atm_ndx_O_H1_coords[0]) ** 2) + ((branch_name.xyz[0][k][1] - uncharged_atm_ndx_O_H1_coords[1]) ** 2) + ((branch_name.xyz[0][k][2] - uncharged_atm_ndx_O_H1_coords[2]) ** 2))
                        dist_O_H2_neighbor = math.sqrt(((branch_name.xyz[0][k][0] - uncharged_atm_ndx_O_H2_coords[0]) ** 2) + ((branch_name.xyz[0][k][1] - uncharged_atm_ndx_O_H2_coords[1]) ** 2) + ((branch_name.xyz[0][k][2] - uncharged_atm_ndx_O_H2_coords[2]) ** 2))

                        if dist_O_H1_neighbor < dist_O_H2_neighbor:
                            unchargedndx_O_H1.append(k)
                        else:
                            unchargedndx_O_H2.append(k)

                    unchargedndx_N_H1 = []
                    unchargedndx_N_H2 = []
                    for k in unchargedndx_N:
                        dist_N_H1_neighbor = math.sqrt(((branch_name.xyz[0][k][0] - uncharged_atm_ndx_N_H1_coords[0]) ** 2) + ((branch_name.xyz[0][k][1] - uncharged_atm_ndx_N_H1_coords[1]) ** 2) + ((branch_name.xyz[0][k][2] - uncharged_atm_ndx_N_H1_coords[2]) ** 2))
                        dist_N_H2_neighbor = math.sqrt(((branch_name.xyz[0][k][0] - uncharged_atm_ndx_N_H2_coords[0]) ** 2) + ((branch_name.xyz[0][k][1] - uncharged_atm_ndx_N_H2_coords[1]) ** 2) + ((branch_name.xyz[0][k][2] - uncharged_atm_ndx_N_H2_coords[2]) ** 2))

                        if dist_N_H1_neighbor < dist_N_H2_neighbor:
                            unchargedndx_N_H1.append(k)
                        else:
                            unchargedndx_N_H2.append(k)

                    print('Considering 4 potential interactions for this residue, here are the neighbors list')
                    print('Neighbors list of side chain O atom part 1: {}'.format(unchargedndx_O_H1))
                    print('Neighbors list of side chain O atom part 2: {}'.format(unchargedndx_O_H2))
                    print('Neighbors list of side chain N atom part 1: {}'.format(unchargedndx_N_H1))
                    print('Neighbors list of side chain N atom part 2: {}'.format(unchargedndx_N_H2))
                    print('########################')
                    ########################################################################################################################

                    # Finalizing neighbor list of side chain O atom (Part 1)
                    unchargedndx_O_H1_unknown = []
                    unchargedndx_O_H1_known = []
                    for k in unchargedndx_O_H1:
                        if k in branch_name.known_don or k in branch_name.known_acc:
                            unchargedndx_O_H1_known.append(k)
                        else:
                            unchargedndx_O_H1_unknown.append(k)
                            unchargedndx_O_H1 = np.delete(unchargedndx_O_H1, np.where(unchargedndx_O_H1 == k)[0][0])

                    print('unchargedndx_O_H1_unknown is {}'.format(unchargedndx_O_H1_unknown))
                    print('unchargedndx_O_H1_known is {}'.format(unchargedndx_O_H1_known))
                    print('########################')

                    ################################################################################################
                    # Finalizing neighbor list of side chain O atom (part 2)
                    unchargedndx_O_H2_unknown = []
                    unchargedndx_O_H2_known = []
                    for k in unchargedndx_O_H2:
                        if k in branch_name.known_don or k in branch_name.known_acc:
                            unchargedndx_O_H2_known.append(k)
                        else:
                            unchargedndx_O_H2_unknown.append(k)
                            unchargedndx_O_H2 = np.delete(unchargedndx_O_H2, np.where(unchargedndx_O_H2 == k)[0][0])

                    print('unchargedndx_O_H2_unknown is {}'.format(unchargedndx_O_H2_unknown))
                    print('unchargedndx_O_H2_known is {}'.format(unchargedndx_O_H2_known))
                    print('########################')

                    ################################################################################################
                    # Finalizing neighbor list of side chain N atom (Part 1)
                    unchargedndx_N_H1_unknown = []
                    unchargedndx_N_H1_known = []
                    for k in unchargedndx_N_H1:
                        if k in branch_name.known_don or k in branch_name.known_acc:
                            unchargedndx_N_H1_known.append(k)
                        else:
                            unchargedndx_N_H1_unknown.append(k)
                            unchargedndx_N_H1 = np.delete(unchargedndx_N_H1, np.where(unchargedndx_N_H1 == k)[0][0])

                    print('unchargedndx_N_H1_unknown is {}'.format(unchargedndx_N_H1_unknown))
                    print('unchargedndx_N_H1_known is {}'.format(unchargedndx_N_H1_known))
                    print('########################')

                    ################################################################################################
                    # Finalizing neighbor list of side chain N atom (part 2)
                    unchargedndx_N_H2_unknown = []
                    unchargedndx_N_H2_known = []
                    for k in unchargedndx_N_H2:
                        if k in branch_name.known_don or k in branch_name.known_acc:
                            unchargedndx_N_H2_known.append(k)
                        else:
                            unchargedndx_N_H2_unknown.append(k)
                            unchargedndx_N_H2 = np.delete(unchargedndx_N_H2, np.where(unchargedndx_N_H2 == k)[0][0])

                    print('unchargedndx_N_H2_unknown is {}'.format(unchargedndx_N_H2_unknown))
                    print('unchargedndx_N_H2_known is {}'.format(unchargedndx_N_H2_known))
                    print('########################')

                    ################################################################################################

                    asngln_atm_nn = []

                    asngln_atm_nn.append(flatten([unchargedndx_O_H1_known, unchargedndx_O_H1_unknown]))
                    asngln_atm_nn.append(flatten([unchargedndx_O_H2_known, unchargedndx_O_H2_unknown]))
                    asngln_atm_nn.append(flatten([unchargedndx_N_H1_known, unchargedndx_N_H1_unknown]))
                    asngln_atm_nn.append(flatten([unchargedndx_N_H2_known, unchargedndx_N_H2_unknown]))

                    known_neighbors_count = len(flatten([list(unchargedndx_O_H1_known), list(unchargedndx_O_H2_known), list(unchargedndx_N_H1_known), list(unchargedndx_N_H2_known)]))

                    for ndx, j in enumerate(asngln_atm_nn):
                        if len(j) == 0:
                            asngln_atm_nn[ndx] = 'E'

                    ############################################################################
                    if known_neighbors_count == 0 and (len(unchargedndx_O_H1_unknown) > 0 or len(unchargedndx_O_H2_unknown) > 0 or len(unchargedndx_N_H1_unknown) > 0 or len(unchargedndx_N_H2_unknown) > 0):
                        # This residue will remain as unknown until end of the first pass.
                        print('The 2 side chain atoms are NOT neighbored with any \'known\' neighbors from the 4 potential interactions and at least one of them is neighbored with an unknown residue\'s atom within the heavy-to-heavy atom distance cutoff.')
                        print('The residue {} will remain \'unknown\' until either identified by a \'known\' atom or branched out.'.format(branch_name.topology.atom(uncharged_atm_ndx_O[i]).residue))
                        print('\n############################################################')
                        continue  # this will stop the current looped residue and without continuing this iteration goes next to the next iteration.

                    else:
                        for k, asngln_X_nn in enumerate(asngln_atm_nn):
                            if len(asngln_X_nn) == 1 and asngln_X_nn == 'E':
                                asngln_X_nn_type = 'E'
                            elif len(asngln_X_nn) == 1:
                                if (asngln_X_nn[0] in branch_name.known_acc) and (asngln_X_nn[0] not in branch_name.known_don):
                                    asngln_X_nn_type = 'A'
                                elif (asngln_X_nn[0] in branch_name.known_don) and (asngln_X_nn[0] not in branch_name.known_acc):
                                    asngln_X_nn_type = 'D'
                                elif asngln_X_nn[0] not in branch_name.known_acc and asngln_X_nn[0] not in branch_name.known_don and asngln_X_nn[0] not in branch_name.grease:  # when neighbored to unknown residues, I need to consider the different types (don/acc/gre) the neighbor can be to make a sound decision
                                    if branch_name.topology.atom(asngln_X_nn[0]).residue.name in ['SER', 'THR', 'TYR', 'ASN', 'GLN']:  # unknown SER/THR/TYR/ASN/GLN
                                        asngln_X_nn_type = ['D/A_unknown']
                                    elif branch_name.topology.atom(asngln_X_nn[0]).residue.name in ['HIS', 'HIE', 'HID', 'HIP']:  # unknown HIS
                                        asngln_X_nn_type = ['D/A/G_unknown']
                                elif asngln_X_nn[0] in branch_name.known_acc and asngln_X_nn[0] in branch_name.known_don:
                                    asngln_X_nn_type = ['D/A_known']
                            elif len(asngln_X_nn) > 1:  # multiple known/unknown neighbors
                                asngln_X_nn_type = []
                                for j in asngln_X_nn:
                                    if (j in branch_name.known_acc) and (j not in branch_name.known_don):
                                        asngln_X_nn_type.append('A')
                                    elif (j in branch_name.known_don) and (j not in branch_name.known_acc):
                                        asngln_X_nn_type.append('D')
                                    elif j not in branch_name.known_acc and j not in branch_name.known_don and j not in branch_name.grease:  # when neighbored to unknown residues, I need to consider the different types (don/acc/gre) the neighbor can be to make a sound decision
                                        if branch_name.topology.atom(j).residue.name in ['SER', 'THR', 'TYR', 'ASN', 'GLN']:
                                            asngln_X_nn_type.append('D/A_unknown')
                                        elif branch_name.topology.atom(j).residue.name in ['HIS', 'HIE', 'HID', 'HIP']:
                                            asngln_X_nn_type.append('D/A/G_unknown')
                                    elif j in branch_name.known_acc and j in branch_name.known_don:
                                        asngln_X_nn_type.append('D/A_known')
                            asngln[str(asnglnRname)][asngln_atm_nm[k]] = asngln_X_nn_type

                        asngln_res = list(asngln.items())[0][0]
                        asngln_res_nn = list(asngln.items())[0][1]
                        print('{} \'s side chain ring atoms with type of their neighbor are: {}\n'.format(asngln_res, asngln_res_nn))

                        asngln_neighbor_configurations_by_atoms = list(itertools.product(*asngln_atm_nn))
                        asngln_neighbor_configurations_by_features = list(itertools.product(*list(asngln.values())[0].values()))
                        print('Number of neighbor configurations to be explored is {}\n'.format(len(asngln_neighbor_configurations_by_atoms)))
                        print('asngln_neighbor_configurations_by_atoms list is {}'.format(asngln_neighbor_configurations_by_atoms))
                        print('asngln_neighbor_configurations_by_features list is {}'.format(asngln_neighbor_configurations_by_features))
                        print('########################')
                        asngln_configuration_decisions_with_energies = defaultdict(dict)
                        asngln_configuration_winning_scenario_features_dict = defaultdict(dict)  # this dictionary will have keys as configuration index and values as the best scenario of this config (in case there was more than one scenario)
                        asngln_configuration_donorneighbors_used = defaultdict(dict)

                        for j, asngln_neighbor_set in enumerate(asngln_neighbor_configurations_by_atoms):
                            print('ASN/GLN neighbor configuration #{}'.format(j))
                            print('ASN/GLN neighbor configuration by atoms is {}'.format(asngln_neighbor_configurations_by_atoms[j]))
                            print('ASN/GLN neighbor configuration by features is {}'.format(asngln_neighbor_configurations_by_features[j]))

                            # https://stackoverflow.com/questions/27220219/how-to-find-a-third-point-given-both-2-points-on-a-line-and-distance-from-thi
                            coords_of_mediating_hydrogens_between_asngln_atoms_and_neighbors = [0, 0, 0, 0]

                            for k in np.arange(4):
                                if asngln_neighbor_configurations_by_features[j][k] == 'E':
                                    coords_of_mediating_hydrogens_between_asngln_atoms_and_neighbors[k] = "No H needed"  # No neighbor. No need to keep coordinates of mediating H.
                                elif asngln_neighbor_configurations_by_features[j][k] == 'D':
                                    coords_of_mediating_hydrogens_between_asngln_atoms_and_neighbors[k] = coords_of_don_neighbor_atom(branch_name, asngln_neighbor_set[k], asngln_atm_ndx[k])
                                elif asngln_neighbor_configurations_by_features[j][k] == 'A':
                                    if k == 0 or k == 2:
                                        coords_of_mediating_hydrogens_between_asngln_atoms_and_neighbors[k] = coords_of_don_neighbor_atom(branch_name, asngln_atm_ndx[k], asngln_neighbor_set[k])[0]
                                    elif k == 1 or k == 3:
                                        coords_of_mediating_hydrogens_between_asngln_atoms_and_neighbors[k] = coords_of_don_neighbor_atom(branch_name, asngln_atm_ndx[k], asngln_neighbor_set[k])[1]

                                elif asngln_neighbor_configurations_by_features[j][k] in ['D/A_unknown', 'D/A/G_unknown', 'D/A_known']:
                                    # FOR THOSE THAT ACT AS 'D/A_unknown' or 'D/A_known'. There are 6 total items in there: three coordinates (x, y, z) for neighbor's H (as donor) and three coordinates (x, y, z) are for the H attached to the histidine's atom if neighbor is acting as acceptor.
                                    # FOR THOSE THAT ACT AS 'D/A/G_unknown'. There are 7 total items in there: three coordinates (x, y, z) for neighbor's H (as donor), three coordinates (x, y, z) are for the H attached to the histidine's atom if neighbor is acting as acceptor, and 'No H needed' which is the grease H atom which will make no difference in the individual energy

                                    ############################################ Unknown neighbor being a donor ############################################
                                    mediating_hydrogen_xyz_coords = coords_of_don_neighbor_atom(branch_name, asngln_neighbor_set[k], asngln_atm_ndx[k])

                                    if asngln_neighbor_configurations_by_features[j][k] == 'D/A_known' and len(flatten(mediating_hydrogen_xyz_coords)) > 3:  # in case the unknown atom has more than one donor
                                        dist_dic = {}
                                        for ndx, m in enumerate(mediating_hydrogen_xyz_coords):
                                            dist = math.sqrt(((branch_name.xyz[0][asngln_atm_ndx[k]][0] - m[0]) ** 2) + ((branch_name.xyz[0][asngln_atm_ndx[k]][1] - m[1]) ** 2) + ((branch_name.xyz[0][asngln_atm_ndx[k]][2] - m[2]) ** 2))
                                            dist_dic[ndx] = dist
                                        the_best_possible_H_atom_of_the_donor = mediating_hydrogen_xyz_coords[sorted(dist_dic.items(), key=operator.itemgetter(1))[0][0]]
                                    elif len(flatten(mediating_hydrogen_xyz_coords)) > 4:
                                        dist_dic = {}
                                        for ndx, m in enumerate(mediating_hydrogen_xyz_coords):
                                            dist = math.sqrt(((branch_name.xyz[0][asngln_atm_ndx[k]][0] - m[1][0]) ** 2) + ((branch_name.xyz[0][asngln_atm_ndx[k]][1] - m[1][1]) ** 2) + ((branch_name.xyz[0][asngln_atm_ndx[k]][2] - m[1][2]) ** 2))
                                            dist_dic[ndx] = dist
                                        the_best_possible_H_atom_of_the_donor = mediating_hydrogen_xyz_coords[sorted(dist_dic.items(), key=operator.itemgetter(1))[0][0]]
                                    else:
                                        the_best_possible_H_atom_of_the_donor = mediating_hydrogen_xyz_coords
                                    ########################################################################################################################

                                    ########################################## Unknown neighbor being an acceptor ##########################################
                                    if k == 0 or k == 2:
                                        H_of_the_asngln_atm_ndx = coords_of_don_neighbor_atom(branch_name, asngln_atm_ndx[k], asngln_neighbor_set[k])[0]
                                    elif k == 1 or k == 3:
                                        H_of_the_asngln_atm_ndx = coords_of_don_neighbor_atom(branch_name, asngln_atm_ndx[k], asngln_neighbor_set[k])[1]
                                    ########################################################################################################################

                                    coords_of_mediating_hydrogens_between_asngln_atoms_and_neighbors[k] = [the_best_possible_H_atom_of_the_donor, H_of_the_asngln_atm_ndx]

                                    if asngln_neighbor_configurations_by_features[j][k] == 'D/A/G_unknown':
                                        coords_of_mediating_hydrogens_between_asngln_atoms_and_neighbors[k].append("No H needed")

                            angles_of_atoms_of_asngln_mediatingH_neighbor = []
                            energies_of_atoms_of_asngln_mediatingH_neighbor = []

                            for k in np.arange(4):
                                if asngln_neighbor_configurations_by_features[j][k] == 'E':  # emt
                                    print('Atom {}: No asngln_X---H---neighbor angle. There is no N or O neighbors to interact with it.'.format(k + 1))
                                    angles_of_atoms_of_asngln_mediatingH_neighbor.append('No Angle')
                                    print('       No asngln_X---H---neighbor energy for atom {} of the ring.'.format(k + 1))
                                    energies_of_atoms_of_asngln_mediatingH_neighbor.append(['No Energy'])
                                else:
                                    if asngln_neighbor_configurations_by_features[j][k] in ['D/A_unknown', 'D/A/G_unknown', 'D/A_known']:
                                        # FOR THOSE THAT ACT AS 'D/A_unknown' or 'D/A_known'. There are 6 total items inside coords_of_mediating_hydrogens_between_asngln_atoms_and_neighbors[k]:
                                        # three coordinates (x, y, z) for neighbor's H (as donor) and three coordinates (x, y, z) are for the H attached to the histidine's atom if neighbor is acting as acceptor.

                                        # FOR THOSE THAT ACT AS 'D/A/G_unknown'. There are 7 total items inside coords_of_mediating_hydrogens_between_asngln_atoms_and_neighbors[k]:
                                        # three coordinates (x, y, z) for neighbor's H (as donor), three coordinates (x, y, z) are for the H attached to the histidine's atom if neighbor is acting as acceptor, and 'No H needed' which is the grease H atom which will make no difference in the individual energy

                                        angle_dic = {}
                                        energy_dic = {}
                                        for l in np.arange(len(coords_of_mediating_hydrogens_between_asngln_atoms_and_neighbors[k])):
                                            # 0 is index in coords list that has a list of xyz coords of donor neighbor, # 1 is index in coords list that has a list of xyz coords of acceptor neighbor
                                            # 2 is index that is there only is the feature is 'D/A/G_unknown' and it refers to 'No H needed' which isn't important here since this part only gets angle of the interaction while 'No H needed' doesnt even refer to an interaction
                                            if l == 0 or l == 1:
                                                asngln_X_xyz_coords = branch_name.xyz[0][asngln_atm_ndx[k]]
                                                mediating_hydrogen_xyz_coords = coords_of_mediating_hydrogens_between_asngln_atoms_and_neighbors[k][l]
                                                if len(flatten(mediating_hydrogen_xyz_coords)) == 4:
                                                    mediating_hydrogen_xyz_coords = np.array(mediating_hydrogen_xyz_coords[1])
                                                asngln_neighbor_xyz_coords = branch_name.xyz[0][asngln_neighbor_set[k]]

                                                vector_1 = asngln_X_xyz_coords - mediating_hydrogen_xyz_coords
                                                vector_2 = asngln_neighbor_xyz_coords - mediating_hydrogen_xyz_coords
                                                vector1_vector_2_cosine_angle = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
                                                acc_H_don_angle_in_radians = np.arccos(vector1_vector_2_cosine_angle)
                                                acc_H_don_angle_in_degrees = np.degrees(acc_H_don_angle_in_radians)

                                                datoms = np.asarray([asngln_atm_ndx[k], asngln_neighbor_set[k]], dtype="int").reshape(1, 2)
                                                dist = md.compute_distances(branch_name.structure, datoms)[0][0]

                                                angle_dic[l] = acc_H_don_angle_in_degrees

                                                if 180 >= round(acc_H_don_angle_in_degrees, 1) >= 90 and 0.50 >= round(dist, 2) >= 0.25:  # distance range of 2.5-5.0 A (0.25-0.5 nm) and angle range of 90-180 degrees were used for the PES calculation. Any distance or angle beyond those two ranges will be 0 kcal/mol because it's not a H-bond interaction
                                                    energy_dic[l] = PES_lookup_table(branch_name, dist, acc_H_don_angle_in_degrees)

                                                else:
                                                    print('Atom {}:'.format(k + 1))
                                                    print('       The distance between the two heavy atoms is {} A, X---H---X angle is {}'.format(round(dist * 10, 3), round(acc_H_don_angle_in_degrees, 2)))
                                                    print('       The X---H---X angle is not between 90-180 degrees. \n'
                                                          '       Since this is not a H-bond interaction, we will keep energy of this interaction as 0 kcal/mol.')
                                                    energy_dic[l] = 0

                                        if asngln_neighbor_configurations_by_features[j][k] in ['D/A_unknown', 'D/A_known']:
                                            angles_of_atoms_of_asngln_mediatingH_neighbor.append([[asngln_neighbor_set[k], 'D', list(angle_dic.items())[0][1]], [asngln_neighbor_set[k], 'A', list(angle_dic.items())[1][1]]])
                                            energies_of_atoms_of_asngln_mediatingH_neighbor.append([[asngln_neighbor_set[k], 'D', list(energy_dic.items())[0][1]], [asngln_neighbor_set[k], 'A', list(energy_dic.items())[1][1]]])
                                        elif asngln_neighbor_configurations_by_features[j][k] == 'D/A/G_unknown':
                                            angles_of_atoms_of_asngln_mediatingH_neighbor.append([[asngln_neighbor_set[k], 'D', list(angle_dic.items())[0][1]], [asngln_neighbor_set[k], 'A', list(angle_dic.items())[1][1]], [asngln_neighbor_set[k], 'G', 'No Angle']])
                                            energies_of_atoms_of_asngln_mediatingH_neighbor.append([[asngln_neighbor_set[k], 'D', list(energy_dic.items())[0][1]], [asngln_neighbor_set[k], 'A', list(energy_dic.items())[1][1]], [asngln_neighbor_set[k], 'G', 'No Energy']])

                                        print('Atom {}: Calculated asngln_X---H---neighbor angles and they are {} and {} degrees, when considering the neighbor as a donor and as an acceptor, respectively.'.format(k + 1, list(angle_dic.items())[0][1], list(angle_dic.items())[1][1]))
                                        print('       Calculated asngln_X---H---neighbor energies for atom {} of the ring and they are {} and {} kcal/mol, when considering the neighbor as a donor and as an acceptor, respectively.'.format(k + 1, list(energy_dic.items())[0][1], list(energy_dic.items())[1][1]))

                                    elif len(flatten(coords_of_mediating_hydrogens_between_asngln_atoms_and_neighbors[k])) == 3 or len(flatten(coords_of_mediating_hydrogens_between_asngln_atoms_and_neighbors[k])) == 4:
                                        # 3 is an acc neighbor's 3 coords or a don neighbor's 3 coords such as (backbone N, ARG NE, TRP NE1).
                                        # 4 is any other don neighbor (except backbone N, ARG NE, TRP NE1, SER, THR, TYR) that while can have more than 1 H atom attached to it, only one is available at the moment.
                                        if len(flatten(coords_of_mediating_hydrogens_between_asngln_atoms_and_neighbors[k])) == 3:
                                            # https://medium.com/@manivannan_data/find-the-angle-between-three-points-from-2d-using-python-348c513e2cd
                                            asngln_X_xyz_coords = branch_name.xyz[0][asngln_atm_ndx[k]]
                                            mediating_hydrogen_xyz_coords = np.array(coords_of_mediating_hydrogens_between_asngln_atoms_and_neighbors[k])
                                            asngln_neighbor_xyz_coords = branch_name.xyz[0][asngln_neighbor_set[k]]

                                        elif len(flatten(coords_of_mediating_hydrogens_between_asngln_atoms_and_neighbors[k])) == 4:
                                            # https://medium.com/@manivannan_data/find-the-angle-between-three-points-from-2d-using-python-348c513e2cd
                                            asngln_X_xyz_coords = branch_name.xyz[0][asngln_atm_ndx[k]]
                                            mediating_hydrogen_xyz_coords = np.array(coords_of_mediating_hydrogens_between_asngln_atoms_and_neighbors[k][1])
                                            asngln_neighbor_xyz_coords = branch_name.xyz[0][asngln_neighbor_set[k]]

                                            # Here, I save the don neighbor H identification so we can put it in the reduced_topology_not_available_donors list for the winning configuration
                                            if asngln_neighbor_configurations_by_features[j][k] == 'D':
                                                if len(asngln_configuration_donorneighbors_used[j]) == 0:  # if this config doesnt have any don neighbor yet
                                                    asngln_configuration_donorneighbors_used[j] = [coords_of_mediating_hydrogens_between_asngln_atoms_and_neighbors[k][0]]
                                                elif len(asngln_configuration_donorneighbors_used[j]) != 0 and coords_of_mediating_hydrogens_between_asngln_atoms_and_neighbors[k][0] not in asngln_configuration_donorneighbors_used[j]:
                                                    asngln_configuration_donorneighbors_used[j].append(coords_of_mediating_hydrogens_between_asngln_atoms_and_neighbors[k][0])

                                        vector_1 = asngln_X_xyz_coords - mediating_hydrogen_xyz_coords
                                        vector_2 = asngln_neighbor_xyz_coords - mediating_hydrogen_xyz_coords
                                        vector1_vector_2_cosine_angle = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
                                        acc_H_don_angle_in_radians = np.arccos(vector1_vector_2_cosine_angle)
                                        acc_H_don_angle_in_degrees = np.degrees(acc_H_don_angle_in_radians)

                                        angles_of_atoms_of_asngln_mediatingH_neighbor.append(acc_H_don_angle_in_degrees)
                                        print('Atom {}: Calculated asngln_X---H---neighbor angle and it\'s {} degrees'.format(k + 1, acc_H_don_angle_in_degrees))
                                        datoms = np.asarray([asngln_atm_ndx[k], asngln_neighbor_set[k]], dtype="int").reshape(1, 2)
                                        dist = md.compute_distances(branch_name.structure, datoms)[0][0]

                                        if 180 >= round(acc_H_don_angle_in_degrees, 1) >= 90 and 0.50 >= round(dist, 2) >= 0.25:  # distance range of 2.5-5.0 A (0.25-0.5 nm) and angle range of 90-180 degrees were used for the PES calculation. Any distance or angle beyond those two ranges will be 0 kcal/mol because it's not a H-bond interaction
                                            energies_of_atoms_of_asngln_mediatingH_neighbor.append([PES_lookup_table(branch_name, dist, acc_H_don_angle_in_degrees)])
                                            print('       Atom {}: Calculated asngln_X---H---neighbor energy and it\'s {} kcal/mol'.format(k + 1, PES_lookup_table(branch_name, dist, acc_H_don_angle_in_degrees)))
                                        else:
                                            print('Atom {}:'.format(k + 1))
                                            print('       The distance between the two heavy atoms is {} A, X---H---X angle is {}'.format(round(dist * 10, 3), round(acc_H_don_angle_in_degrees, 2)))
                                            print('       The X---H---X angle is not between 90-180 degrees. \n'
                                                  '       Since this is not a H-bond interaction, we will keep energy of this interaction as 0 kcal/mol.')
                                            energies_of_atoms_of_asngln_mediatingH_neighbor.append([0])

                                    else:  # this else is when the neighbor is a donor that has physically more than one available donor to make the H-bond (such as LYS NZ (up to 3 H's), GLN NE2 (up to 2 H's), ASN ND2 (up to 2 H's), ARG NH2 (up to 2 H's), ARG NH1 (up to 2 H's))
                                        angle_dic = {}
                                        energy_dic = {}
                                        for l in np.arange(len(coords_of_mediating_hydrogens_between_asngln_atoms_and_neighbors[k])):
                                            asngln_X_xyz_coords = branch_name.xyz[0][asngln_atm_ndx[k]]
                                            mediating_hydrogen_xyz_coords = np.array(coords_of_mediating_hydrogens_between_asngln_atoms_and_neighbors[k][l][1])
                                            asngln_neighbor_xyz_coords = branch_name.xyz[0][asngln_neighbor_set[k]]

                                            vector_1 = asngln_X_xyz_coords - mediating_hydrogen_xyz_coords
                                            vector_2 = asngln_neighbor_xyz_coords - mediating_hydrogen_xyz_coords

                                            vector1_vector_2_cosine_angle = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
                                            acc_H_don_angle_in_radians = np.arccos(vector1_vector_2_cosine_angle)
                                            acc_H_don_angle_in_degrees = np.degrees(acc_H_don_angle_in_radians)

                                            datoms = np.asarray([asngln_atm_ndx[k], asngln_neighbor_set[k]], dtype="int").reshape(1, 2)
                                            dist = md.compute_distances(branch_name.structure, datoms)[0][0]

                                            angle_dic[l] = acc_H_don_angle_in_degrees

                                            if 180 >= round(acc_H_don_angle_in_degrees, 1) >= 90 and 0.50 >= round(dist, 2) >= 0.25:  # distance range of 2.5-5.0 A (0.25-0.5 nm) and angle range of 90-180 degrees were used for the PES calculation. Any distance or angle beyond those two ranges will be 0 kcal/mol because it's not a H-bond interaction
                                                energy_dic[l] = PES_lookup_table(branch_name, dist, acc_H_don_angle_in_degrees)
                                            else:
                                                print('Atom {}:'.format(k + 1))
                                                print('       The distance between the two heavy atoms is {} A, X---H---X angle is {}'.format(round(dist * 10, 3), round(acc_H_don_angle_in_degrees, 2)))
                                                print('       The X---H---X angle is not between 90-180 degrees. \n'
                                                      '       Since this is not a H-bond interaction, we will keep energy of this interaction as 0 kcal/mol.')
                                                energy_dic[l] = 0

                                        bval = sorted(energy_dic.items(), key=operator.itemgetter(1))[0][0]
                                        chosen_angle = list(angle_dic.items())[bval][1]
                                        chosen_energy = list(energy_dic.items())[bval][1]

                                        # Here, I save the don neighbor H identification so we can put it in the reduced_topology_not_available_donors list for the winning configuration
                                        if len(asngln_configuration_donorneighbors_used[j]) == 0:  # if this config doesnt have any don neighbor yet
                                            asngln_configuration_donorneighbors_used[j] = [coords_of_mediating_hydrogens_between_asngln_atoms_and_neighbors[k][bval][0]]
                                        elif len(asngln_configuration_donorneighbors_used[j]) != 0 and coords_of_mediating_hydrogens_between_asngln_atoms_and_neighbors[k][bval][0] not in asngln_configuration_donorneighbors_used[j]:
                                            asngln_configuration_donorneighbors_used[j].append(coords_of_mediating_hydrogens_between_asngln_atoms_and_neighbors[k][bval][0])

                                        angles_of_atoms_of_asngln_mediatingH_neighbor.append(chosen_angle)
                                        print('Atom {}: Calculated his_X---H---neighbor angle of the ring and it\'s {} degrees'.format(k + 1, chosen_angle))
                                        energies_of_atoms_of_asngln_mediatingH_neighbor.append([chosen_energy])
                                        print('       Calculated his_X---H---neighbor energy for atom {} of the ring and it\'s {} kcal/mol'.format(k + 1, chosen_energy))

                            if len(asngln_neighbor_configurations_by_atoms) == 1 and (asngln_neighbor_configurations_by_features[j][0] == 'E' and asngln_neighbor_configurations_by_features[j][1] == 'E' and asngln_neighbor_configurations_by_features[j][2] == 'E' and asngln_neighbor_configurations_by_features[j][3] == 'E'):
                                print('Since this ASN/GLN residue\'s side chain atoms neighbored with nothing, we will identify this residue as is. No need to calculate rotamer energies')
                            else:
                                if len(flatten(energies_of_atoms_of_asngln_mediatingH_neighbor)) == 4:
                                    ############################################################################
                                    # Calculate energies of the two states.
                                    # state a: the 0 and 1 index is the O atom and the 2 and 3 index is the N
                                    # state b: is vice versa
                                    if asngln_neighbor_configurations_by_features[j][0] == 'D':
                                        state_a = energies_of_atoms_of_asngln_mediatingH_neighbor[0][0]  # favorable
                                        state_b = -(energies_of_atoms_of_asngln_mediatingH_neighbor[0][0])  # clash
                                    elif asngln_neighbor_configurations_by_features[j][0] == 'A':
                                        state_a = -(energies_of_atoms_of_asngln_mediatingH_neighbor[0][0])  # clash
                                        state_b = energies_of_atoms_of_asngln_mediatingH_neighbor[0][0]  # favorable
                                    else:
                                        state_a = 0
                                        state_b = 0

                                    if asngln_neighbor_configurations_by_features[j][1] == 'D':
                                        state_a += energies_of_atoms_of_asngln_mediatingH_neighbor[1][0]  # favorable
                                        state_b += -(energies_of_atoms_of_asngln_mediatingH_neighbor[1][0])  # clash
                                    elif asngln_neighbor_configurations_by_features[j][1] == 'A':
                                        state_a += -(energies_of_atoms_of_asngln_mediatingH_neighbor[1][0])  # clash
                                        state_b += energies_of_atoms_of_asngln_mediatingH_neighbor[1][0]  # favorable
                                    else:
                                        state_a += 0
                                        state_b += 0

                                    if asngln_neighbor_configurations_by_features[j][2] == 'D':
                                        state_a += -(energies_of_atoms_of_asngln_mediatingH_neighbor[2][0])  # clash
                                        state_b += energies_of_atoms_of_asngln_mediatingH_neighbor[2][0]  # favorable
                                    elif asngln_neighbor_configurations_by_features[j][2] == 'A':
                                        state_a += energies_of_atoms_of_asngln_mediatingH_neighbor[2][0]  # favorable
                                        state_b += -(energies_of_atoms_of_asngln_mediatingH_neighbor[2][0])  # clash
                                    else:
                                        state_a += 0
                                        state_b += 0

                                    if asngln_neighbor_configurations_by_features[j][3] == 'D':
                                        state_a += -(energies_of_atoms_of_asngln_mediatingH_neighbor[3][0])  # clash
                                        state_b += energies_of_atoms_of_asngln_mediatingH_neighbor[3][0]  # favorable
                                    elif asngln_neighbor_configurations_by_features[j][3] == 'A':
                                        state_a += energies_of_atoms_of_asngln_mediatingH_neighbor[3][0]  # favorable
                                        state_b += -(energies_of_atoms_of_asngln_mediatingH_neighbor[3][0])  # clash
                                    else:
                                        state_a += 0
                                        state_b += 0

                                    ############################################################################

                                    ############################################################################
                                    # we will put all energies
                                    asngln_possible_rotamers_energies = {'state_a': state_a, 'state_b': state_b}

                                    asngln_rotamers_ordered_in_energy = sorted(asngln_possible_rotamers_energies.items(), key=lambda x: x[1])
                                    asngln_rotamers_ordered_in_energy = [list(ele) for ele in asngln_rotamers_ordered_in_energy]

                                    print('ASN/GLN possible rotamers and their energies are {}.'.format(asngln_rotamers_ordered_in_energy))

                                    difference_in_energy_between_lowest_2_possible_rotamers = round(abs(asngln_rotamers_ordered_in_energy[0][1] - asngln_rotamers_ordered_in_energy[1][1]), 2)
                                    print('The difference in energy between the lowest two possible rotamers is {}'.format(difference_in_energy_between_lowest_2_possible_rotamers))

                                    if difference_in_energy_between_lowest_2_possible_rotamers >= degenerate_states_e_cutoff:
                                        print('Since the difference in energy between the lowest two possible rotamers for this neighbor configuration is equal or above the degenerate states E cutoff.')
                                        print('     Keeping only the lowest energetic rotamer for this configuration ({}): {}'.format(j, asngln_rotamers_ordered_in_energy[0]))
                                        asngln_configuration_decisions_with_energies[j] = asngln_rotamers_ordered_in_energy[0]

                                    else:
                                        print('Since the difference in energy between the lowest two possible rotamers for this neighbors configuration is within the degenerate states E cutoff, we will keep both HIS possibilities!!')
                                        asngln_configuration_decisions_with_energies[j] = [asngln_rotamers_ordered_in_energy[0], asngln_rotamers_ordered_in_energy[1]]

                                else:
                                    energies_of_atoms_of_asngln_mediatingH_neighbor = list(itertools.product(*energies_of_atoms_of_asngln_mediatingH_neighbor))
                                    all_possibilities_lowest_rotamers = defaultdict(dict)
                                    for index_of_m, m in enumerate(energies_of_atoms_of_asngln_mediatingH_neighbor):
                                        ############################################################################
                                        # Calculate energies of the two states.
                                        # state a: the 0 index is the O atom and the 1 index is the N
                                        # state b: is vice versa
                                        if asngln_neighbor_configurations_by_features[j][0] == 'D':
                                            state_a = m[0]  # favorable
                                            state_b = -(m[0])  # clash
                                        elif asngln_neighbor_configurations_by_features[j][0] == 'A':
                                            state_a = -(m[0])  # clash
                                            state_b = m[0]  # favorable
                                        elif asngln_neighbor_configurations_by_features[j][0] in ['D/A_unknown', 'D/A/G_unknown', 'D/A_known']:
                                            if m[0][1] == 'D':
                                                state_a = m[0][2]  # favorable
                                                state_b = -(m[0][2])  # clash
                                            elif m[0][1] == 'A':
                                                state_a = -(m[0][2])  # clash
                                                state_b = m[0][2]  # favorable
                                            else:
                                                state_a = 0
                                                state_b = 0
                                        else:
                                            state_a = 0
                                            state_b = 0
                                        ############################################################################

                                        ############################################################################
                                        if asngln_neighbor_configurations_by_features[j][1] == 'D':
                                            state_a += m[1]  # favorable
                                            state_b += -(m[1])  # clash
                                        elif asngln_neighbor_configurations_by_features[j][1] == 'A':
                                            state_a += -(m[1])  # clash
                                            state_b += m[1]  # favorable
                                        elif asngln_neighbor_configurations_by_features[j][1] in ['D/A_unknown', 'D/A/G_unknown', 'D/A_known']:
                                            if m[1][1] == 'D':
                                                state_a += m[1][2]  # favorable
                                                state_b += -(m[1][2])  # clash
                                            elif m[1][1] == 'A':
                                                state_a += -(m[1][2])  # clash
                                                state_b += m[1][2]  # favorable
                                            else:
                                                state_a += 0
                                                state_b += 0
                                        else:
                                            state_a += 0
                                            state_b += 0
                                        ############################################################################

                                        ############################################################################
                                        if asngln_neighbor_configurations_by_features[j][2] == 'D':
                                            state_a += -(m[2])  # clash
                                            state_b += m[2]  # favorable
                                        elif asngln_neighbor_configurations_by_features[j][2] == 'A':
                                            state_a += m[2]  # favorable
                                            state_b += -(m[2])  # clash
                                        elif asngln_neighbor_configurations_by_features[j][2] in ['D/A_unknown', 'D/A/G_unknown', 'D/A_known']:
                                            if m[2][1] == 'D':
                                                state_a += -(m[2][2])  # clash
                                                state_b += m[2][2]  # favorable
                                            elif m[2][1] == 'A':
                                                state_a += m[2][2]  # favorable
                                                state_b += -(m[2][2])  # clash
                                            else:
                                                state_a += 0
                                                state_b += 0
                                        else:
                                            state_a += 0
                                            state_b += 0
                                        ############################################################################

                                        ############################################################################
                                        if asngln_neighbor_configurations_by_features[j][3] == 'D':
                                            state_a += -(m[3])  # clash
                                            state_b += m[3]  # favorable
                                        elif asngln_neighbor_configurations_by_features[j][3] == 'A':
                                            state_a += m[3]  # favorable
                                            state_b += -(m[3])  # clash
                                        elif asngln_neighbor_configurations_by_features[j][3] in ['D/A_unknown', 'D/A/G_unknown', 'D/A_known']:
                                            if m[3][1] == 'D':
                                                state_a += -(m[3][2])  # clash
                                                state_b += m[3][2]  # favorable
                                            elif m[3][1] == 'A':
                                                state_a += m[3][2]  # favorable
                                                state_b += -(m[3][2])  # clash
                                            else:
                                                state_a += 0
                                                state_b += 0
                                        else:
                                            state_a += 0
                                            state_b += 0
                                        ############################################################################

                                        ############################################################################
                                        # getting rid of bad scenarios where an unknown atom such as an ASN/GLN/HIS (it can act as a D, or A (or even G, for HIS))
                                        unk_neighbors_in_config = np.array([[key[0], key[1]] for idx, key in enumerate(m) if type(key) == list])
                                        unk_neighbors_wholeres_dict = defaultdict(dict)

                                        bad_scenario_count = 0

                                        # I cant have a HIS ND atom be D while CE (which is supposed to be Grease) be A to another side of the residue. I have to have consistency in the assignment possibility
                                        for k_ndx, k in enumerate(unk_neighbors_in_config):
                                            if int(k[0]) not in unk_neighbors_wholeres_dict:
                                                this_current_unk_resname = branch_name.topology.atom(int(k[0])).residue.name
                                                this_current_unk_resindx = branch_name.topology.atom(int(k[0])).residue.index
                                                if this_current_unk_resname in ['HIS', 'HIE', 'HID', 'HIP']:
                                                    this_current_unk_ND1 = branch_name.topology.select("resid {} and name ND1".format(this_current_unk_resindx))[0]
                                                    this_current_unk_CE1 = branch_name.topology.select("resid {} and name CE1".format(this_current_unk_resindx))[0]
                                                    this_current_unk_NE2 = branch_name.topology.select("resid {} and name NE2".format(this_current_unk_resindx))[0]
                                                    this_current_unk_CD2 = branch_name.topology.select("resid {} and name CD2".format(this_current_unk_resindx))[0]
                                                    if int(k[0]) == this_current_unk_ND1 or int(k[0]) == this_current_unk_NE2:
                                                        if k[1] in ['D', 'A']:
                                                            unk_neighbors_wholeres_dict[this_current_unk_ND1] = ['D', 'A']
                                                            unk_neighbors_wholeres_dict[this_current_unk_NE2] = ['D', 'A']
                                                            unk_neighbors_wholeres_dict[this_current_unk_CE1] = ['G']
                                                            unk_neighbors_wholeres_dict[this_current_unk_CD2] = ['G']
                                                        elif k[1] == 'G':
                                                            unk_neighbors_wholeres_dict[this_current_unk_ND1] = ['G']
                                                            unk_neighbors_wholeres_dict[this_current_unk_NE2] = ['G']
                                                            unk_neighbors_wholeres_dict[this_current_unk_CE1] = ['D', 'A']
                                                            unk_neighbors_wholeres_dict[this_current_unk_CD2] = ['D', 'A']
                                                    elif int(k[0]) == this_current_unk_CE1 or int(k[0]) == this_current_unk_CD2:
                                                        if k[1] in ['D', 'A']:
                                                            unk_neighbors_wholeres_dict[this_current_unk_CE1] = ['D', 'A']
                                                            unk_neighbors_wholeres_dict[this_current_unk_CD2] = ['D', 'A']
                                                            unk_neighbors_wholeres_dict[this_current_unk_ND1] = ['G']
                                                            unk_neighbors_wholeres_dict[this_current_unk_NE2] = ['G']
                                                        elif k[1] == 'G':
                                                            unk_neighbors_wholeres_dict[this_current_unk_CE1] = ['G']
                                                            unk_neighbors_wholeres_dict[this_current_unk_CD2] = ['G']
                                                            unk_neighbors_wholeres_dict[this_current_unk_ND1] = ['D', 'A']
                                                            unk_neighbors_wholeres_dict[this_current_unk_NE2] = ['D', 'A']

                                                elif this_current_unk_resname in ['ASN', 'GLN']:
                                                    this_current_unk_o_side = branch_name.topology.select('((resname ASN and name OD1) or (resname GLN and name OE1)) and resid {}'.format(this_current_unk_resindx))[0]
                                                    this_current_unk_n_side = branch_name.topology.select('((resname ASN and name ND2) or (resname GLN and name NE2)) and resid {}'.format(this_current_unk_resindx))[0]
                                                    if int(k[0]) == this_current_unk_o_side:
                                                        if k[1] == 'D':
                                                            unk_neighbors_wholeres_dict[this_current_unk_o_side] = ['D']
                                                            unk_neighbors_wholeres_dict[this_current_unk_n_side] = ['A']
                                                        elif k[1] == 'A':
                                                            unk_neighbors_wholeres_dict[this_current_unk_o_side] = ['A']
                                                            unk_neighbors_wholeres_dict[this_current_unk_n_side] = ['D']
                                                    elif int(k[0]) == this_current_unk_n_side:
                                                        if k[1] == 'D':
                                                            unk_neighbors_wholeres_dict[this_current_unk_n_side] = ['D']
                                                            unk_neighbors_wholeres_dict[this_current_unk_o_side] = ['A']
                                                        elif k[1] == 'A':
                                                            unk_neighbors_wholeres_dict[this_current_unk_n_side] = ['A']
                                                            unk_neighbors_wholeres_dict[this_current_unk_o_side] = ['D']

                                            elif int(k[0]) in unk_neighbors_wholeres_dict:
                                                if k[1] not in unk_neighbors_wholeres_dict[int(k[0])]:
                                                    bad_scenario_count += 1

                                        # An unknown atom can not be a feature to one atom of the looped residue and another feature with another atom of the looped residue
                                        # if this atom is neighbored with the O and N atoms of another ASN, I cant have a scenario where the neighbor is D to O and A to N. that atom will have to have the same feature.
                                        for k_ndx, k in enumerate(unk_neighbors_in_config):
                                            for l_ndx, l in enumerate(unk_neighbors_in_config):
                                                if k_ndx != l_ndx:
                                                    if k[0] == l[0] and k[1] != l[1] and branch_name.topology.atom(int(k[0])).residue.name in ['ASN', 'GLN', 'HIS', 'HIE', 'HID', 'HIP']:
                                                        bad_scenario_count += 1
                                        ############################################################################

                                        if bad_scenario_count == 0:
                                            asngln_possible_rotamers_energies = {'state_a': state_a, 'state_b': state_b}

                                            asngln_rotamers_ordered_in_energy = sorted(asngln_possible_rotamers_energies.items(), key=lambda x: x[1])
                                            asngln_rotamers_ordered_in_energy = [list(ele) for ele in asngln_rotamers_ordered_in_energy]

                                            print('Scenario {}:'.format(index_of_m))
                                            print('     Individual energies of 2 atoms of ASN/GLN of this arrangement : {}'.format(m))
                                            print('     ASN/GLN possible rotamers and their energies are {}.'.format(asngln_rotamers_ordered_in_energy))

                                            difference_in_energy_between_lowest_2_possible_rotamers = round(abs(asngln_rotamers_ordered_in_energy[0][1] - asngln_rotamers_ordered_in_energy[1][1]), 2)

                                            if difference_in_energy_between_lowest_2_possible_rotamers >= degenerate_states_e_cutoff:
                                                all_possibilities_lowest_rotamers[index_of_m] = asngln_rotamers_ordered_in_energy[0]
                                            else:
                                                all_possibilities_lowest_rotamers[index_of_m] = asngln_rotamers_ordered_in_energy

                                        else:
                                            print('Scenario {}:'.format(index_of_m))
                                            print('     Individual energies of 2 atoms of ASN/GLN of this arrangement : {}'.format(m))
                                            print('     This neighbor configuration is NOT a good scenario due to having conflicting unknown atom(s) features.')
                                            print('     It will not be considered as a scenario')

                                    ############################################################
                                    # the point of this block is to extract the BEST scenario and its features so we can use it after finding the BEST winning configuration
                                    # the point of this asngln_config_scenarios_energies_edited dictionary is to basically make the dictionary symmetric.
                                    # meaning, if a configuration was to have more than one rotamer energy, I will just keep one
                                    asngln_config_scenarios_energies_edited = defaultdict(dict)
                                    for configuration in list(all_possibilities_lowest_rotamers.items()):
                                        if len(flatten(configuration[1])) == 2:
                                            asngln_config_scenarios_energies_edited[configuration[0]] = configuration[1]
                                        else:
                                            asngln_config_scenarios_energies_edited[configuration[0]] = configuration[1][0]

                                    asngln_config_scenarios_energies = sorted(list(asngln_config_scenarios_energies_edited.items()), key=lambda x: x[1][1])
                                    asngln_configuration_winning_scenario_index = asngln_config_scenarios_energies[0][0]
                                    asngln_configuration_winning_scenario_features_dict[j] = energies_of_atoms_of_asngln_mediatingH_neighbor[asngln_configuration_winning_scenario_index]
                                    ############################################################

                                    asngln_rotamers_ordered_in_energy = [list(ele) for ele in asngln_config_scenarios_energies]
                                    print('ASN/GLN possible rotamers and their energies are {}.'.format(asngln_rotamers_ordered_in_energy))

                                    if len(asngln_rotamers_ordered_in_energy) == 1:
                                        print('There is only one rotamer that happen to be the lowest rotamer energetically for BOTH arrangements of this specific configuration')
                                        print('     Keeping only that lowest rotamer in energy for this configuration ({}): {}'.format(j, asngln_rotamers_ordered_in_energy[0]))
                                        asngln_configuration_decisions_with_energies[j] = asngln_rotamers_ordered_in_energy[0][1]
                                    else:
                                        difference_in_energy_between_lowest_2_possible_rotamers = round(abs(asngln_rotamers_ordered_in_energy[0][1][1] - asngln_rotamers_ordered_in_energy[1][1][1]), 2)
                                        print('The difference in energy between the lowest two possible rotamers is {}'.format(difference_in_energy_between_lowest_2_possible_rotamers))

                                        if difference_in_energy_between_lowest_2_possible_rotamers >= degenerate_states_e_cutoff:
                                            print('Since the difference in energy between the lowest two possible rotamers for this neighbors configuration is equal or above the degenerate states E cutoff.')
                                            print('     Keeping only the lowest energetic rotamer for this configuration ({}): {}'.format(j, asngln_rotamers_ordered_in_energy[0][1]))
                                            asngln_configuration_decisions_with_energies[j] = asngln_rotamers_ordered_in_energy[0][1]

                                        else:
                                            print('Since the difference in energy between the lowest two possible rotamers for this neighbors configuration is within the degenerate states E cutoff, we will keep both HIS possibilities!!')
                                            asngln_configuration_decisions_with_energies[j] = [asngln_rotamers_ordered_in_energy[0][1], asngln_rotamers_ordered_in_energy[1][1]]

                                    print('########################')

                            print('\n########################')

                        ########################################################################
                        ########################################################################

                        print('List of asngln_neighbor_configurations_by_atoms is {}'.format(asngln_neighbor_configurations_by_atoms))
                        print('List of asngln_neighbor_configurations_by_features is {}'.format(asngln_neighbor_configurations_by_features))
                        print('List of all ASN/GLN configurations most stable rotamers and their lowest energies:')
                        for configuration in list(asngln_configuration_decisions_with_energies.items()):
                            print('     ' + str(configuration))
                        print('########################\n')

                        if len(asngln_neighbor_configurations_by_atoms) == 1:  # Only one neighbor configuration to this ASN/GLN residue
                            print('There is only one ASN/GLN neighbor configuration')

                            if asngln_neighbor_configurations_by_features[j][0] == 'E' and asngln_neighbor_configurations_by_features[j][1] == 'E' and asngln_neighbor_configurations_by_features[j][2] == 'E' and asngln_neighbor_configurations_by_features[j][3] == 'E':
                                asngln_rotamer_winner = 'state_a'
                                print('Since this ASN/GLN residue\'s atoms neighbored with nothing, we will identify this residue as it is (state_a)')
                            else:
                                asngln_rotamers_ordered_in_energy = list(asngln_configuration_decisions_with_energies.items())[0][1]

                                if len(flatten(asngln_rotamers_ordered_in_energy)) == 2:
                                    asngln_rotamer_winner = asngln_rotamers_ordered_in_energy[0]
                                else:
                                    difference_in_energy_between_lowest_2_possible_rotamers = round(abs(asngln_rotamers_ordered_in_energy[0][1] - asngln_rotamers_ordered_in_energy[1][1]), 2)
                                    print('The difference in energy between the lowest two rotamers of this ONLY configuration is {}'.format(difference_in_energy_between_lowest_2_possible_rotamers))

                                    if difference_in_energy_between_lowest_2_possible_rotamers >= degenerate_states_e_cutoff:
                                        print('Since the difference in energy between the lowest two rotamers energetically is equal or above the degenerate states E cutoff.')
                                        asngln_rotamer_winner = asngln_rotamers_ordered_in_energy[0][0]
                                        print('     Keeping only the lowest rotamer energetically of this ONLY configuration of this ASN/GLN residue: {}'.format(asngln_rotamer_winner))

                                    else:
                                        print('Since the difference in energy between the lowest two rotamers energetically is within the degenerate states E cutoff, we will not identify this residue or make it known unless branching is necessary later on in the analysis!!')
                                        asngln_rotamer_winner = [asngln_rotamers_ordered_in_energy[0][0], asngln_rotamers_ordered_in_energy[1][0]]

                                        print('asngln_splits of this residue is {}'.format(asngln_rotamer_winner))
                                        index_of_this_asngln_split = len(branch_name.asngln_with_splits)
                                        branch_name.asngln_with_splits.append([branch_name.topology.atom(uncharged_atm_ndx_O[i]).residue.index])

                                        for x in asngln_rotamer_winner:
                                            branch_name.asngln_with_splits[index_of_this_asngln_split].append(x)

                        else:  # When there is more than one neighbor configuration to this ASN/GLN residue
                            print('There are more than one ASN/GLN neighbor configuration')

                            ############################################################
                            # the point of this asngln_configurations_energies_edited dictionary is to basically make the dictionary symmetric. meaning, if a configuration was to have more than one rotamer energy, I will just keep one so in the next step of the code I can compare one rotamer energy of each configuration without making errors in sorted() usage
                            asngln_configurations_energies_edited = defaultdict(dict)
                            for configuration in list(asngln_configuration_decisions_with_energies.items()):
                                if len(flatten(configuration[1])) == 2:
                                    asngln_configurations_energies_edited[configuration[0]] = configuration[1]
                                else:
                                    asngln_configurations_energies_edited[configuration[0]] = configuration[1][0]

                            asngln_rotamers_ordered_in_energy = sorted(asngln_configurations_energies_edited.items(), key=lambda x: x[1][1])
                            asngln_rotamers_ordered_in_energy = [list(ele) for ele in asngln_rotamers_ordered_in_energy]
                            print('List of all ASN/GLN configurations ORDERED by most stable rotamers and their lowest energies:')
                            for configuration in asngln_rotamers_ordered_in_energy:
                                print('     ' + str(configuration))
                            print('########################\n')
                            ############################################################

                            difference_in_energy_between_lowest_2_possible_rotamers = round(abs(asngln_rotamers_ordered_in_energy[0][1][1] - asngln_rotamers_ordered_in_energy[1][1][1]), 2)
                            print('The difference in energy between the two configurations\' lowest rotamers in energy is {}'.format(difference_in_energy_between_lowest_2_possible_rotamers))

                            initial_splits = []

                            if difference_in_energy_between_lowest_2_possible_rotamers >= degenerate_states_e_cutoff:
                                print('Since the difference in energy between the lowest two configuration rotamers energetically is equal or above the degenerate states E cutoff.')
                                winning_configuration_index = asngln_rotamers_ordered_in_energy[0][0]
                                print('     Keeping only the lowest configuration rotamer energetically for this ASN/GLN residue (configuration {}): {}'.format(winning_configuration_index, asngln_configuration_decisions_with_energies.get(winning_configuration_index)))

                                print('     The winning neighbor configuration of this ASN/GLN residue by atoms is: {}'.format(asngln_neighbor_configurations_by_atoms[winning_configuration_index]))
                                print('     The winning neighbor configuration of this ASN/GLN residue by features is: {}'.format(asngln_neighbor_configurations_by_features[winning_configuration_index]))

                                asngln_rotamer_winner = flatten(asngln_configuration_decisions_with_energies.get(winning_configuration_index))[::2]
                                asngln_rotamer_winner = list(np.unique(np.array(asngln_rotamer_winner)))

                                if len(asngln_rotamer_winner) == 1:
                                    asngln_rotamer_winner = asngln_rotamer_winner[0]

                            else:
                                print('Since the difference in energy between the lowest two configuration rotamers energetically is within the degenerate states E cutoff, we will not identify this residue or make it known unless branching is necessary later on in the analysis!!')

                                for configuration_index in [asngln_rotamers_ordered_in_energy[0][0], asngln_rotamers_ordered_in_energy[1][0]]:
                                    if len(flatten(asngln_configuration_decisions_with_energies.get(configuration_index))) == 2:
                                        initial_splits.append(asngln_configuration_decisions_with_energies.get(configuration_index)[0])
                                    else:
                                        for x in asngln_configuration_decisions_with_energies.get(configuration_index):
                                            initial_splits.append(x[0])

                                asngln_rotamer_winner = list(np.unique(np.array(initial_splits)))

                                if len(initial_splits) > 1 and len(flatten(asngln_rotamer_winner)) == 1:
                                    print('Even though the difference in energy between the lowest two configuration rotamers energetically is less than the degenerate states E cutoff, the most stable two configurations decided on the same rotamer identification')
                                    winning_configuration_index = asngln_rotamers_ordered_in_energy[0][0]
                                    print('     Keeping only the lowest configuration rotamer energetically for this ASN/GLN residue (configuration {}): {}'.format(winning_configuration_index, asngln_configuration_decisions_with_energies.get(winning_configuration_index)))

                                    print('     The winning neighbor configuration of this ASN/GLN residue by atoms is: {}'.format(asngln_neighbor_configurations_by_atoms[winning_configuration_index]))
                                    print('     The winning neighbor configuration of this ASN/GLN residue by features is: {}'.format(asngln_neighbor_configurations_by_features[winning_configuration_index]))

                                    asngln_rotamer_winner = list(np.unique(np.array(initial_splits)))[0]

                                else:
                                    print('asngln_splits of this residue is {}'.format(asngln_rotamer_winner))
                                    index_of_this_asngln_split = len(branch_name.asngln_with_splits)
                                    branch_name.asngln_with_splits.append([branch_name.topology.atom(uncharged_atm_ndx_N[i]).residue.index])

                                    for x in asngln_rotamer_winner:
                                        branch_name.asngln_with_splits[index_of_this_asngln_split].append(x)

                        if isinstance(asngln_rotamer_winner, list) is False:

                            ####################################################
                            # This is very important because if I only have one configuration, the winning configuration index will always be "0".
                            if len(asngln_neighbor_configurations_by_atoms) == 1:
                                winning_configuration_index = 0
                            ####################################################

                            ####################################################
                            # Now, that we know the winning_configuration_index, we can fetch the don neighbor H id which was used so we can append it to the branch_name.reduced_topology_not_available_donors list
                            for don_neighbor_id in asngln_configuration_donorneighbors_used[winning_configuration_index]:
                                branch_name.reduced_topology_not_available_donors.append(don_neighbor_id)
                                print('Atom {} just got added to reduced_topology_not_available_donors list. This ensures that this H can\'t be used later.'.format(don_neighbor_id))
                            ####################################################

                            ####################################################
                            if asngln_rotamer_winner == 'state_a':
                                print('This residue wins as "state_a" (means atom names are left as is).')
                                print('No atom flipping required!')

                            elif asngln_rotamer_winner == 'state_b':
                                print('This residue wins as "state_b" (means atom names will need to be flipped).')
                                print('Atom flipping required!')

                                # flipping atom names
                                print('Atoms were {} ({}) and {} ({})'.format(uncharged_atm_ndx_O[i], branch_name.topology.atom(uncharged_atm_ndx_O[i]).name, uncharged_atm_ndx_N[i], branch_name.topology.atom(uncharged_atm_ndx_N[i]).name))
                                branch_name.switched_atm_name_uncharged.append((uncharged_atm_ndx_O[i], uncharged_atm_ndx_N[i]))
                                tmp_O = branch_name.topology.atom(uncharged_atm_ndx_O[i]).name
                                tmp_N = branch_name.topology.atom(uncharged_atm_ndx_N[i]).name
                                branch_name.topology.atom(uncharged_atm_ndx_O[i]).name = tmp_N
                                branch_name.topology.atom(uncharged_atm_ndx_N[i]).name = tmp_O

                                tmp_O_element = branch_name.topology.atom(uncharged_atm_ndx_O[i]).element
                                tmp_N_element = branch_name.topology.atom(uncharged_atm_ndx_N[i]).element
                                branch_name.topology.atom(uncharged_atm_ndx_O[i]).element = tmp_N_element
                                branch_name.topology.atom(uncharged_atm_ndx_N[i]).element = tmp_O_element

                                branch_name.atm_name_change_to_N.append(uncharged_atm_ndx_O[i])
                                branch_name.atm_name_change_to_O.append(uncharged_atm_ndx_N[i])
                                print('Atoms just flipped to accommodate complementarity for H-bonding')
                                print('Atoms are now {} ({}) and {} ({})'.format(uncharged_atm_ndx_O[i], branch_name.topology.atom(uncharged_atm_ndx_O[i]).name, uncharged_atm_ndx_N[i], branch_name.topology.atom(uncharged_atm_ndx_N[i]).name))

                            ####################################################
                            if asngln_rotamer_winner == 'state_a':
                                # atom O (part 1)
                                if asngln_neighbor_configurations_by_features[winning_configuration_index][0] == 'D':
                                    branch_name.known_don = np.delete(branch_name.known_don, np.where(branch_name.known_don == asngln_neighbor_configurations_by_atoms[winning_configuration_index][0])[0][0])
                                    print('Atom {} just got deleted from known_don list ONCE because it was just used up in an H-bond interaction.'.format(asngln_neighbor_configurations_by_atoms[winning_configuration_index][0]))

                                    the_donor_neighbor = asngln_neighbor_configurations_by_atoms[winning_configuration_index][0]
                                    if branch_name.topology.atom(the_donor_neighbor).name in ['OH', 'OG', 'OG1'] and branch_name.topology.atom(the_donor_neighbor).residue.name in ['TYR', 'SER', 'THR']:
                                        print('       The donor atom {} ({}) which is used in this interaction happens to be SER/THR/TYR residue'.format(the_donor_neighbor, branch_name.topology.atom(the_donor_neighbor)))
                                        print('       This donor along with the acceptor will be added to donor_of_oxi_SER_THR_TYR_acc_neighbor_set list so the H can be printed in the output structure')
                                        branch_name.donor_of_oxi_SER_THR_TYR_acc_neighbor_set.append([[the_donor_neighbor, branch_name.topology.atom(the_donor_neighbor), branch_name.xyz[0][the_donor_neighbor]], [uncharged_atm_ndx_O[i], branch_name.topology.atom(uncharged_atm_ndx_O[i]), branch_name.xyz[0][uncharged_atm_ndx_O[i]]]])
                                elif asngln_neighbor_configurations_by_features[winning_configuration_index][0] == 'D/A_known':
                                    if asngln_configuration_winning_scenario_features_dict[winning_configuration_index][0][1] == 'D':
                                        branch_name.known_don = np.delete(branch_name.known_don, np.where(branch_name.known_don == asngln_neighbor_configurations_by_atoms[winning_configuration_index][0])[0][0])
                                        print('Atom {} just got deleted from known_don list ONCE because it was just used up in an H-bond interaction.'.format(asngln_neighbor_configurations_by_atoms[winning_configuration_index][0]))
                                    elif asngln_configuration_winning_scenario_features_dict[winning_configuration_index][0][1] == 'A':
                                        branch_name.known_acc = np.append(branch_name.known_acc, uncharged_atm_ndx_O[i])
                                        print('Atom {} just got added to known_acc list ONCE. No complementary H-bond interaction.'.format(uncharged_atm_ndx_O[i]))
                                else:
                                    branch_name.known_acc = np.append(branch_name.known_acc, uncharged_atm_ndx_O[i])
                                    print('Atom {} just got added to known_acc list ONCE. No complementary H-bond interaction.'.format(uncharged_atm_ndx_O[i]))

                                # atom O (part 2)
                                if asngln_neighbor_configurations_by_features[winning_configuration_index][1] == 'D':
                                    branch_name.known_don = np.delete(branch_name.known_don, np.where(branch_name.known_don == asngln_neighbor_configurations_by_atoms[winning_configuration_index][1])[0][0])
                                    print('Atom {} just got deleted from known_don list ONCE because it was just used up in an H-bond interaction.'.format(asngln_neighbor_configurations_by_atoms[winning_configuration_index][1]))

                                    the_donor_neighbor = asngln_neighbor_configurations_by_atoms[winning_configuration_index][1]
                                    if branch_name.topology.atom(the_donor_neighbor).name in ['OH', 'OG', 'OG1'] and branch_name.topology.atom(the_donor_neighbor).residue.name in ['TYR', 'SER', 'THR']:
                                        print('       The donor atom {} ({}) which is used in this interaction happens to be SER/THR/TYR residue'.format(the_donor_neighbor, branch_name.topology.atom(the_donor_neighbor)))
                                        print('       This donor along with the acceptor will be added to donor_of_oxi_SER_THR_TYR_acc_neighbor_set list so the H can be printed in the output structure')
                                        branch_name.donor_of_oxi_SER_THR_TYR_acc_neighbor_set.append([[the_donor_neighbor, branch_name.topology.atom(the_donor_neighbor), branch_name.xyz[0][the_donor_neighbor]], [uncharged_atm_ndx_O[i], branch_name.topology.atom(uncharged_atm_ndx_O[i]), branch_name.xyz[0][uncharged_atm_ndx_O[i]]]])
                                elif asngln_neighbor_configurations_by_features[winning_configuration_index][1] == 'D/A_known':
                                    if asngln_configuration_winning_scenario_features_dict[winning_configuration_index][1][1] == 'D':
                                        branch_name.known_don = np.delete(branch_name.known_don, np.where(branch_name.known_don == asngln_neighbor_configurations_by_atoms[winning_configuration_index][1])[0][0])
                                        print('Atom {} just got deleted from known_don list ONCE because it was just used up in an H-bond interaction.'.format(asngln_neighbor_configurations_by_atoms[winning_configuration_index][1]))
                                    elif asngln_configuration_winning_scenario_features_dict[winning_configuration_index][1][1] == 'A':
                                        branch_name.known_acc = np.append(branch_name.known_acc, uncharged_atm_ndx_O[i])
                                        print('Atom {} just got added to known_acc list ONCE. No complementary H-bond interaction.'.format(uncharged_atm_ndx_O[i]))
                                else:
                                    branch_name.known_acc = np.append(branch_name.known_acc, uncharged_atm_ndx_O[i])
                                    print('Atom {} just got added to known_acc list ONCE. No complementary H-bond interaction.'.format(uncharged_atm_ndx_O[i]))

                                # atom N (part 1)
                                if asngln_neighbor_configurations_by_features[winning_configuration_index][2] == 'A':
                                    branch_name.known_acc = np.delete(branch_name.known_acc, np.where(branch_name.known_acc == asngln_neighbor_configurations_by_atoms[winning_configuration_index][2])[0][0])
                                    print('Atom {} just got deleted from known_acc list ONCE because it was just used up in an H-bond interaction.'.format(asngln_neighbor_configurations_by_atoms[winning_configuration_index][2]))

                                    # This used-up donor H cant be used again
                                    branch_name.reduced_topology_not_available_donors.append(uncharged_atm_ndx_N[i] + 3)
                                    print('Atom {} just got added to reduced_topology_not_available_donors list. This ensures that this H can\'t be used later.'.format(uncharged_atm_ndx_N[i] + 3))
                                elif asngln_neighbor_configurations_by_features[winning_configuration_index][2] == 'D/A_known':
                                    if asngln_configuration_winning_scenario_features_dict[winning_configuration_index][2][1] == 'D':
                                        branch_name.known_don = np.append(branch_name.known_don, uncharged_atm_ndx_N[i])
                                        print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(uncharged_atm_ndx_N[i]))
                                    elif asngln_configuration_winning_scenario_features_dict[winning_configuration_index][2][1] == 'A':
                                        branch_name.known_acc = np.delete(branch_name.known_acc, np.where(branch_name.known_acc == asngln_neighbor_configurations_by_atoms[winning_configuration_index][2])[0][0])
                                        print('Atom {} just got deleted from known_acc list ONCE because it was just used up in an H-bond interaction.'.format(asngln_neighbor_configurations_by_atoms[winning_configuration_index][2]))
                                else:
                                    branch_name.known_don = np.append(branch_name.known_don, uncharged_atm_ndx_N[i])
                                    print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(uncharged_atm_ndx_N[i]))

                                # atom N (part 2)
                                if asngln_neighbor_configurations_by_features[winning_configuration_index][3] == 'A':
                                    branch_name.known_acc = np.delete(branch_name.known_acc, np.where(branch_name.known_acc == asngln_neighbor_configurations_by_atoms[winning_configuration_index][3])[0][0])
                                    print('Atom {} just got deleted from known_acc list ONCE because it was just used up in an H-bond interaction.'.format(asngln_neighbor_configurations_by_atoms[winning_configuration_index][3]))

                                    # This used-up donor H cant be used again
                                    branch_name.reduced_topology_not_available_donors.append(uncharged_atm_ndx_N[i] + 4)
                                    print('Atom {} just got added to reduced_topology_not_available_donors list. This ensures that this H can\'t be used later.'.format(uncharged_atm_ndx_N[i] + 4))
                                elif asngln_neighbor_configurations_by_features[winning_configuration_index][3] == 'D/A_known':
                                    if asngln_configuration_winning_scenario_features_dict[winning_configuration_index][3][1] == 'D':
                                        branch_name.known_don = np.append(branch_name.known_don, uncharged_atm_ndx_N[i])
                                        print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(uncharged_atm_ndx_N[i]))
                                    elif asngln_configuration_winning_scenario_features_dict[winning_configuration_index][3][1] == 'A':
                                        branch_name.known_acc = np.delete(branch_name.known_acc, np.where(branch_name.known_acc == asngln_neighbor_configurations_by_atoms[winning_configuration_index][3])[0][0])
                                        print('Atom {} just got deleted from known_acc list ONCE because it was just used up in an H-bond interaction.'.format(asngln_neighbor_configurations_by_atoms[winning_configuration_index][3]))
                                else:
                                    branch_name.known_don = np.append(branch_name.known_don, uncharged_atm_ndx_N[i])
                                    print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(uncharged_atm_ndx_N[i]))

                            elif asngln_rotamer_winner == 'state_b':
                                # atom N (part 1)
                                if asngln_neighbor_configurations_by_features[winning_configuration_index][0] == 'A':
                                    branch_name.known_acc = np.delete(branch_name.known_acc, np.where(branch_name.known_acc == asngln_neighbor_configurations_by_atoms[winning_configuration_index][0])[0][0])
                                    print('Atom {} just got deleted from known_acc list ONCE because it was just used up in an H-bond interaction.'.format(asngln_neighbor_configurations_by_atoms[winning_configuration_index][0]))

                                    # This used-up donor H cant be used again
                                    branch_name.reduced_topology_not_available_donors.append(uncharged_atm_ndx_O[i] + 3)
                                    print('Atom {} just got added to reduced_topology_not_available_donors list. This ensures that this H can\'t be used later.'.format(uncharged_atm_ndx_O[i] + 1))
                                elif asngln_neighbor_configurations_by_features[winning_configuration_index][0] == 'D/A_known':
                                    if asngln_configuration_winning_scenario_features_dict[winning_configuration_index][0][1] == 'D':
                                        branch_name.known_don = np.append(branch_name.known_don, uncharged_atm_ndx_O[i])
                                        print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(uncharged_atm_ndx_O[i]))
                                    elif asngln_configuration_winning_scenario_features_dict[winning_configuration_index][0][1] == 'A':
                                        branch_name.known_acc = np.delete(branch_name.known_acc, np.where(branch_name.known_acc == asngln_neighbor_configurations_by_atoms[winning_configuration_index][0])[0][0])
                                        print('Atom {} just got deleted from known_acc list ONCE because it was just used up in an H-bond interaction.'.format(asngln_neighbor_configurations_by_atoms[winning_configuration_index][0]))
                                else:
                                    branch_name.known_don = np.append(branch_name.known_don, uncharged_atm_ndx_O[i])
                                    print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(uncharged_atm_ndx_O[i]))

                                # atom N (part 2)
                                if asngln_neighbor_configurations_by_features[winning_configuration_index][1] == 'A':
                                    branch_name.known_acc = np.delete(branch_name.known_acc, np.where(branch_name.known_acc == asngln_neighbor_configurations_by_atoms[winning_configuration_index][1])[0][0])
                                    print('Atom {} just got deleted from known_acc list ONCE because it was just used up in an H-bond interaction.'.format(asngln_neighbor_configurations_by_atoms[winning_configuration_index][1]))

                                    # This used-up donor H cant be used again
                                    branch_name.reduced_topology_not_available_donors.append(uncharged_atm_ndx_O[i] + 4)
                                    print('Atom {} just got added to reduced_topology_not_available_donors list. This ensures that this H can\'t be used later.'.format(uncharged_atm_ndx_O[i] + 2))
                                elif asngln_neighbor_configurations_by_features[winning_configuration_index][1] == 'D/A_known':
                                    if asngln_configuration_winning_scenario_features_dict[winning_configuration_index][1][1] == 'D':
                                        branch_name.known_don = np.append(branch_name.known_don, uncharged_atm_ndx_O[i])
                                        print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(uncharged_atm_ndx_O[i]))
                                    elif asngln_configuration_winning_scenario_features_dict[winning_configuration_index][1][1] == 'A':
                                        branch_name.known_acc = np.delete(branch_name.known_acc, np.where(branch_name.known_acc == asngln_neighbor_configurations_by_atoms[winning_configuration_index][1])[0][0])
                                        print('Atom {} just got deleted from known_acc list ONCE because it was just used up in an H-bond interaction.'.format(asngln_neighbor_configurations_by_atoms[winning_configuration_index][1]))
                                else:
                                    branch_name.known_don = np.append(branch_name.known_don, uncharged_atm_ndx_O[i])
                                    print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(uncharged_atm_ndx_O[i]))

                                # atom O (part 1)
                                if asngln_neighbor_configurations_by_features[winning_configuration_index][2] == 'D':
                                    branch_name.known_don = np.delete(branch_name.known_don, np.where(branch_name.known_don == asngln_neighbor_configurations_by_atoms[winning_configuration_index][2])[0][0])
                                    print('Atom {} just got deleted from known_don list ONCE because it was just used up in an H-bond interaction.'.format(asngln_neighbor_configurations_by_atoms[winning_configuration_index][2]))

                                    the_donor_neighbor = asngln_neighbor_configurations_by_atoms[winning_configuration_index][2]
                                    if branch_name.topology.atom(the_donor_neighbor).name in ['OH', 'OG', 'OG1'] and branch_name.topology.atom(the_donor_neighbor).residue.name in ['TYR', 'SER', 'THR']:
                                        print('       The donor atom {} ({}) which is used in this interaction happens to be SER/THR/TYR residue'.format(the_donor_neighbor, branch_name.topology.atom(the_donor_neighbor)))
                                        print('       This donor along with the acceptor will be added to donor_of_oxi_SER_THR_TYR_acc_neighbor_set list so the H can be printed in the output structure')
                                        branch_name.donor_of_oxi_SER_THR_TYR_acc_neighbor_set.append([[the_donor_neighbor, branch_name.topology.atom(the_donor_neighbor), branch_name.xyz[0][the_donor_neighbor]], [uncharged_atm_ndx_N[i], branch_name.topology.atom(uncharged_atm_ndx_N[i]), branch_name.xyz[0][uncharged_atm_ndx_N[i]]]])
                                elif asngln_neighbor_configurations_by_features[winning_configuration_index][2] == 'D/A_known':
                                    if asngln_configuration_winning_scenario_features_dict[winning_configuration_index][2][1] == 'D':
                                        branch_name.known_don = np.delete(branch_name.known_don, np.where(branch_name.known_don == asngln_neighbor_configurations_by_atoms[winning_configuration_index][2])[0][0])
                                        print('Atom {} just got deleted from known_don list ONCE because it was just used up in an H-bond interaction.'.format(asngln_neighbor_configurations_by_atoms[winning_configuration_index][2]))
                                    elif asngln_configuration_winning_scenario_features_dict[winning_configuration_index][2][1] == 'A':
                                        branch_name.known_acc = np.append(branch_name.known_acc, uncharged_atm_ndx_N[i])
                                        print('Atom {} just got added to known_acc list ONCE. No complementary H-bond interaction.'.format(uncharged_atm_ndx_N[i]))
                                else:
                                    branch_name.known_acc = np.append(branch_name.known_acc, uncharged_atm_ndx_N[i])
                                    print('Atom {} just got added to known_acc list ONCE. No complementary H-bond interaction.'.format(uncharged_atm_ndx_N[i]))

                                # atom O (part 2)
                                if asngln_neighbor_configurations_by_features[winning_configuration_index][3] == 'D':
                                    branch_name.known_don = np.delete(branch_name.known_don, np.where(branch_name.known_don == asngln_neighbor_configurations_by_atoms[winning_configuration_index][3])[0][0])
                                    print('Atom {} just got deleted from known_don list ONCE because it was just used up in an H-bond interaction.'.format(asngln_neighbor_configurations_by_atoms[winning_configuration_index][3]))

                                    the_donor_neighbor = asngln_neighbor_configurations_by_atoms[winning_configuration_index][3]
                                    if branch_name.topology.atom(the_donor_neighbor).name in ['OH', 'OG', 'OG1'] and branch_name.topology.atom(the_donor_neighbor).residue.name in ['TYR', 'SER', 'THR']:
                                        print('       The donor atom {} ({}) which is used in this interaction happens to be SER/THR/TYR residue'.format(the_donor_neighbor, branch_name.topology.atom(the_donor_neighbor)))
                                        print('       This donor along with the acceptor will be added to donor_of_oxi_SER_THR_TYR_acc_neighbor_set list so the H can be printed in the output structure')
                                        branch_name.donor_of_oxi_SER_THR_TYR_acc_neighbor_set.append([[the_donor_neighbor, branch_name.topology.atom(the_donor_neighbor), branch_name.xyz[0][the_donor_neighbor]], [uncharged_atm_ndx_N[i], branch_name.topology.atom(uncharged_atm_ndx_N[i]), branch_name.xyz[0][uncharged_atm_ndx_N[i]]]])
                                elif asngln_neighbor_configurations_by_features[winning_configuration_index][3] == 'D/A_known':
                                    if asngln_configuration_winning_scenario_features_dict[winning_configuration_index][3][1] == 'D':
                                        branch_name.known_don = np.delete(branch_name.known_don, np.where(branch_name.known_don == asngln_neighbor_configurations_by_atoms[winning_configuration_index][3])[0][0])
                                        print('Atom {} just got deleted from known_don list ONCE because it was just used up in an H-bond interaction.'.format(asngln_neighbor_configurations_by_atoms[winning_configuration_index][3]))
                                    elif asngln_configuration_winning_scenario_features_dict[winning_configuration_index][3][1] == 'A':
                                        branch_name.known_acc = np.append(branch_name.known_acc, uncharged_atm_ndx_N[i])
                                        print('Atom {} just got added to known_acc list ONCE. No complementary H-bond interaction.'.format(uncharged_atm_ndx_N[i]))
                                else:
                                    branch_name.known_acc = np.append(branch_name.known_acc, uncharged_atm_ndx_N[i])
                                    print('Atom {} just got added to known_acc list ONCE. No complementary H-bond interaction.'.format(uncharged_atm_ndx_N[i]))

                            ####################################################

                            # delete known residues from the unknown residues list
                            branch_name.unknown_residues = np.delete(branch_name.unknown_residues, np.where(branch_name.unknown_residues == branch_name.topology.atom(uncharged_atm_ndx_O[i]).residue.index)[0][0])

                            # print('known_acc has {} atoms now'.format(len(branch_name.known_acc)))
                            # print('last 20 items in known_acc are {}'.format(branch_name.known_acc[-20:]))
                            # print('known_don has {} atoms now'.format(len(branch_name.known_don)))
                            # print('last 20 items in known_don are {}'.format(branch_name.known_don[-20:]))

                        print('\n############################################################')

                ########################  # END ASN and GLN  ########################

            elif branch_name.topology.residue(res_indx).name in ['SER', 'THR', 'TYR']:

                ########################
                # Check SER, THR and TYR
                ########################
                print('This unknown residue is hydroxyl containing residues (SER, THR, or TYR):')
                print(branch_name.topology.residue(res_indx))

                oxi_atm_ndx = branch_name.topology.select('((resname SER and name OG) or (resname THR and name OG1) or (resname TYR and name OH)) and resid {}'.format(res_indx))

                if len(oxi_atm_ndx) == 0:
                    print('It looks like this residue is missing the side chain hydroxyl group oxygen atom.')
                    print('skipping this residue from the analysis and removing its index from the unknown_residues list')
                    # delete known residues from the unknown residues list
                    branch_name.unknown_residues = np.delete(branch_name.unknown_residues, np.where(branch_name.unknown_residues == res_indx)[0][0])
                    print('############################################################')

                oxi_bond = len(oxi_atm_ndx)

                ohRes = defaultdict(dict)

                for i in np.arange(oxi_bond):
                    oxi_nn = md.compute_neighbors(branch_name.structure, h_bond_heavy_atm_cutoff, np.asarray([oxi_atm_ndx[i]]), branch_name.allpos_prot_don_acc, periodic=False)[0]

                    print('Atom {} is the hydroxyl group oxygen atom of {} (res_indx {})'.format(oxi_atm_ndx[i], branch_name.topology.atom(oxi_atm_ndx[i]).residue, res_indx))

                    oxi_Rid = branch_name.topology.atom(oxi_atm_ndx[i]).residue.index

                    # Delete backbone O and N of same residue from neighbor list
                    oxi_bb = [atom.index for atom in branch_name.topology.atoms if (((atom.element.symbol == 'N') or (atom.element.symbol == 'O')) and (atom.residue.index == oxi_Rid) and atom.is_backbone)]  # list of backbone O and N of the looped residue
                    for j in oxi_bb:
                        if j in oxi_nn:
                            oxi_nn = np.delete(oxi_nn, np.where(oxi_nn == j)[0])

                    # Delete from neighbor lists of side chain O and N of unknown_residues or any neighbor that is not available anymore to H-bond. those can be:
                    # 1 - backbone N or O that is not available anymore for H-bond
                    # OR 2 - side chain atoms whose residue is not in unknown_residues list (means they were already identified but no longer able to H-bond as they were used already).
                    for k in oxi_nn:
                        if k not in branch_name.known_don and k not in branch_name.known_acc and ((branch_name.topology.atom(k).name == 'N' or branch_name.topology.atom(k).name == 'O') or ((branch_name.topology.atom(k).name != 'N' or branch_name.topology.atom(k).name != 'O') and branch_name.topology.atom(k).residue.index not in branch_name.unknown_residues)):
                            oxi_nn = np.delete(oxi_nn, np.where(oxi_nn == k)[0][0])

                    print('List of neighbors of the side chain O atom of {}:'.format(branch_name.topology.atom(oxi_atm_ndx[i]).residue))
                    print(oxi_nn)

                    ############################################################
                    # filter out all non H bond interactions neighbors for which distance between the heavy atoms is 2.5 A or less
                    for k in oxi_nn:
                        dist_O_neighbor = math.sqrt(((10*branch_name.xyz[0][k][0] - 10*branch_name.xyz[0][oxi_atm_ndx[i]][0]) ** 2) + ((10*branch_name.xyz[0][k][1] - 10*branch_name.xyz[0][oxi_atm_ndx[i]][1]) ** 2) + ((10*branch_name.xyz[0][k][2] - 10*branch_name.xyz[0][oxi_atm_ndx[i]][2]) ** 2))
                        if dist_O_neighbor < 2.5:
                            oxi_nn = np.delete(oxi_nn, np.where(oxi_nn == k)[0][0])

                    print('List of neighbors of the side chain O atom of {} after dropping neighbors with distance lower than 2.5 A:'.format(branch_name.topology.atom(oxi_atm_ndx[i]).residue))
                    print(oxi_nn)
                    print('########################')
                    ############################################################

                    # Here we filter out non Hbond neighbors that are having bad angles
                    oxi_nn = filter_out_non_hbond_neighbors(branch_name, oxi_nn, oxi_atm_ndx[i])
                    print('List of neighbors of the side chain O atom of {} after filtering out non H-bond neighbors:'.format(branch_name.topology.atom(oxi_atm_ndx[i]).residue))
                    print(oxi_nn)

                    oxi_nn_don = []
                    oxi_nn_acc = []
                    oxi_nn_unknown = []
                    if len(oxi_nn) != 0:
                        for k in oxi_nn:
                            if k in branch_name.known_don and k not in branch_name.known_acc:
                                oxi_nn_don.append(k)
                            elif k in branch_name.known_acc and k not in branch_name.known_don:
                                oxi_nn_acc.append(k)
                            elif k in branch_name.known_acc and k in branch_name.known_don:  # this is when there is an OH group of a ligand being used in the analysis. it usually has one donor and one acc on it
                                oxi_nn_don.append(k)
                                oxi_nn_acc.append(k)
                            else:
                                oxi_nn_unknown.append(k)

                    ####################################################################################################################################################################################

                    if len(oxi_nn_unknown) != 0 and len(oxi_nn_don) == 0 and len(oxi_nn_acc) == 0:
                        # this will remain an unknown residue until we get to the second pass
                        print('{} (residue {}) seems to not have any neighbor atom of \'known\' residues to set itself with. It will remain \'unknown\' until either identified by a \'known\' atom or branched out.'.format(branch_name.topology.atom(oxi_atm_ndx[i]).residue, branch_name.topology.atom(oxi_atm_ndx[i]).residue.index))
                        print('\n############################################################')
                    else:
                        oxiRname = branch_name.topology.atom(oxi_atm_ndx[i]).residue

                        print('Neighbors acting as H-bond ACCEPTORS:')
                        if len(oxi_nn_acc) == 0:
                            print('There are NO neighbors acting as H-bond acceptors at all.')
                            branch_name.known_don = np.append(branch_name.known_don, oxi_atm_ndx[i])
                            print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(oxi_atm_ndx[i]))
                        elif len(oxi_nn_acc) >= 1:
                            if len(oxi_nn_acc) == 1:
                                bval = oxi_nn_acc[0]
                                print('There is only one neighbor acting as a H-bond acceptor: {}.'.format(bval))
                            elif len(oxi_nn_acc) > 1:
                                print('There are more than one neighbor acting as H-bond acceptors: {},\n     I will only keep the most stable H bond acceptor neighbor energetically to set the H atom coordinates based on pointing the donor towards the acceptor of the H-bond'.format(oxi_nn_acc))
                                energy_dic_acceptor_neighbors = {}
                                for l in oxi_nn_acc:
                                    heavy_atom_1 = oxi_atm_ndx[i]  # the atom which has the list of the neighbor
                                    heavy_atom_1_coords = branch_name.xyz[0][heavy_atom_1]

                                    heavy_atom_2 = l  # the neighbor is accepting, thus we will find the H atom(s) on heavy_atom_1
                                    heavy_atom_2_coords = branch_name.xyz[0][heavy_atom_2]

                                    mediating_hydrogen_xyz_coords = coords_of_don_neighbor_atom(branch_name, heavy_atom_1, heavy_atom_2)

                                    vector_1 = heavy_atom_1_coords - mediating_hydrogen_xyz_coords
                                    vector_2 = heavy_atom_2_coords - mediating_hydrogen_xyz_coords
                                    vector1_vector_2_cosine_angle = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
                                    acc_H_don_angle_in_radians = np.arccos(vector1_vector_2_cosine_angle)
                                    acc_H_don_angle_in_degrees = np.degrees(acc_H_don_angle_in_radians)

                                    datoms = np.asarray([oxi_atm_ndx[i], l], dtype="int").reshape(1, 2)
                                    dist = md.compute_distances(branch_name.structure, datoms)[0][0]

                                    energy_dic_acceptor_neighbors[l] = PES_lookup_table(branch_name, dist, acc_H_don_angle_in_degrees)

                                print(energy_dic_acceptor_neighbors)
                                bval = sorted(energy_dic_acceptor_neighbors.items(), key=operator.itemgetter(1))[0][0]
                                print('Neighbor atom {} is the best H-bond making acceptor energetically.'.format(bval))

                            print('The donor (atom {}) will not be added to known_don list because it was just used up in an H-bond interaction.'.format(oxi_atm_ndx[i]))

                            branch_name.known_acc = np.delete(branch_name.known_acc, np.where(branch_name.known_acc == bval)[0][0])
                            print('The acceptor (neighbor) (atom {}) just got deleted ONCE from known_acc list because it was just used up in an H-bond interaction.'.format(bval))
                            oxi_nn_acc = [(bval, branch_name.topology.atom(bval))]
                            branch_name.directly_set_donor_of_oxi_SER_THR_TYR.append([[oxi_atm_ndx[i], branch_name.topology.atom(oxi_atm_ndx[i]), branch_name.xyz[0][oxi_atm_ndx[i]]], [oxi_nn_acc[0][0], branch_name.topology.atom(oxi_nn_acc[0][0]), branch_name.xyz[0][oxi_nn_acc[0][0]]]])

                        print('###################')
                        ####################################################################################################################################################################################

                        print('Neighbors acting as H-bond DONORS:')
                        if len(oxi_nn_don) == 0:
                            print('There are NO neighbors acting as H-bond donors at all.')
                            branch_name.known_acc = np.append(branch_name.known_acc, oxi_atm_ndx[i])
                            print('Atom {} just got added to known_acc list ONCE. No complementary H-bond interaction.'.format(oxi_atm_ndx[i]))
                        elif len(oxi_nn_don) >= 1:
                            if len(oxi_nn_don) == 1:
                                bval = oxi_nn_don[0]
                                print('There is only one neighbor acting as a H-bond donor: {}.'.format(bval))
                            elif len(oxi_nn_don) > 1:
                                print('There are more than one neighbor acting as H-bond donors: {},\n     I will only keep the closest donor neighbor.'.format(oxi_nn_don))
                                energy_dic_donor_neighbors = {}
                                for l in oxi_nn_don:
                                    datoms = np.asarray([oxi_atm_ndx[i], l], dtype="int").reshape(1, 2)
                                    dist = md.compute_distances(branch_name.structure, datoms)[0][0]

                                    heavy_atom_1 = l  # the neighbor is donating, thus we will find the H atom(s) on heavy_atom_1
                                    heavy_atom_1_coords = branch_name.xyz[0][heavy_atom_1]

                                    heavy_atom_2 = oxi_atm_ndx[i]  # the atom which has the list of the neighbor
                                    heavy_atom_2_coords = branch_name.xyz[0][heavy_atom_2]

                                    mediating_hydrogen_xyz_coords = coords_of_don_neighbor_atom(branch_name, heavy_atom_1, heavy_atom_2)

                                    if len(flatten(mediating_hydrogen_xyz_coords)) == 3:
                                        mediating_hydrogen_xyz_coords = [mediating_hydrogen_xyz_coords]
                                    elif len(flatten(mediating_hydrogen_xyz_coords)) == 4:  # here we just get rid of the first item which is the index of the H atoms. we only keep the three coords
                                        mediating_hydrogen_xyz_coords = [mediating_hydrogen_xyz_coords[1]]
                                    elif len(flatten(mediating_hydrogen_xyz_coords)) in [8, 12]:  # here we do the same - we just keep coordinates in a list of lists
                                        mediating_hydrogen_xyz_coords = [item[1] for item in mediating_hydrogen_xyz_coords]

                                    heavy_H_heavy_angles = []
                                    for s in mediating_hydrogen_xyz_coords:
                                        vector_1 = heavy_atom_1_coords - s
                                        vector_2 = heavy_atom_2_coords - s
                                        vector1_vector_2_cosine_angle = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
                                        acc_H_don_angle_in_radians = np.arccos(vector1_vector_2_cosine_angle)
                                        acc_H_don_angle_in_degrees = np.degrees(acc_H_don_angle_in_radians)
                                        heavy_H_heavy_angles.append(acc_H_don_angle_in_degrees)

                                    possible_energies_for_one_atom = []
                                    for s in heavy_H_heavy_angles:
                                        possible_energies_for_one_atom.append(PES_lookup_table(branch_name, dist, s))
                                    energy_dic_donor_neighbors[l] = min(possible_energies_for_one_atom)

                                print(energy_dic_donor_neighbors)
                                bval = sorted(energy_dic_donor_neighbors.items(), key=operator.itemgetter(1))[0][0]
                                print('Neighbor atom {} is the best H-bond making donor energetically'.format(bval))

                            print('The acceptor (atom {}) will not be added to known_acc list because it was just used up in an H-bond interaction.'.format(oxi_atm_ndx[i]))

                            branch_name.known_don = np.delete(branch_name.known_don, np.where(branch_name.known_don == bval)[0][0])
                            print('The donor (neighbor) (atom {}) just got deleted ONCE from known_don list because it was just used up in an H-bond interaction.'.format(bval))
                            oxi_nn_don = [(bval, branch_name.topology.atom(bval))]

                            if (branch_name.topology.atom(oxi_nn_don[0][0]).residue.name == 'SER' and branch_name.topology.atom(oxi_nn_don[0][0]).name == 'OG') or (branch_name.topology.atom(oxi_nn_don[0][0]).residue.name == 'TYR' and branch_name.topology.atom(oxi_nn_don[0][0]).name == 'OH') or (branch_name.topology.atom(oxi_nn_don[0][0]).residue.name == 'THR' and branch_name.topology.atom(oxi_nn_don[0][0]).name == 'OG1'):
                                print('The acceptor of atom {} is H-bonded with the donor of atom {},\n     Just appended this donor-acc pair in \'indirectly_set_donor_of_oxi_SER_THR_TYR\' list'.format(oxi_atm_ndx[i], oxi_nn_don[0][0]))
                                branch_name.indirectly_set_donor_of_oxi_SER_THR_TYR.append([oxi_nn_don[0][0], oxi_atm_ndx[i]])

                        print('###################')
                        ####################################################################################################################################################################################

                        # delete known residues from the unknown residues list
                        branch_name.unknown_residues = np.delete(branch_name.unknown_residues, np.where(branch_name.unknown_residues == branch_name.topology.atom(oxi_atm_ndx[i]).residue.index)[0][0])

                        ohRes[str(oxiRname)] = {'don': oxi_nn_don, 'acc': oxi_nn_acc}

                        for oxi_res, oxi_res_nn in list(ohRes.items()):
                            if oxi_res == str(branch_name.topology.atom(oxi_atm_ndx[i]).residue):
                                print('{} \'s side chain hydroxyl oxygen is neighbored with the following: {}'.format(oxi_res, oxi_res_nn))

                        # print('known_acc has {} atoms now'.format(len(branch_name.known_acc)))
                        # print('last 20 items in known_acc are {}'.format(branch_name.known_acc[-20:]))
                        # print('known_don has {} atoms now'.format(len(branch_name.known_don)))
                        # print('last 20 items in known_don are {}'.format(branch_name.known_don[-20:]))

                        print('\n############################################################')

                branch_name.donor_of_oxi_SER_THR_TYR_acc_neighbor_set = branch_name.directly_set_donor_of_oxi_SER_THR_TYR

                if len(branch_name.indirectly_set_donor_of_oxi_SER_THR_TYR) > 0:
                    for i in branch_name.indirectly_set_donor_of_oxi_SER_THR_TYR:
                        branch_name.donor_of_oxi_SER_THR_TYR_acc_neighbor_set.append([[i[0], branch_name.topology.atom(i[0]), branch_name.xyz[0][i[0]]], [i[1], branch_name.topology.atom(i[1]), branch_name.xyz[0][i[1]]]])

                ########################  # END SER, THR, and TYR  ########################

            elif branch_name.topology.residue(res_indx).name in ['HIS', 'HIE', 'HID', 'HIP']:
                ########################
                # Check HIS
                ########################
                print('This unknown residue is a HIS residue:')
                print(branch_name.topology.residue(res_indx))

                his_CG = branch_name.topology.select('(resname HIS or resname HIE or resname HID or resname HIP) and name CG and resid {}'.format(res_indx))  # 113      # I need this for when calculating the 4 H that are connected to each atom of the HIS
                his_ND = branch_name.topology.select('(resname HIS or resname HIE or resname HID or resname HIP) and name ND1 and resid {}'.format(res_indx))  # 114
                his_CE = branch_name.topology.select('(resname HIS or resname HIE or resname HID or resname HIP) and name CE1 and resid {}'.format(res_indx))  # 116
                his_NE = branch_name.topology.select('(resname HIS or resname HIE or resname HID or resname HIP) and name NE2 and resid {}'.format(res_indx))  # 117
                his_CD = branch_name.topology.select('(resname HIS or resname HIE or resname HID or resname HIP) and name CD2 and resid {}'.format(res_indx))  # 115

                his_bond = len(his_NE)

                his = defaultdict(dict)

                his_atm_nm = {0: "ND", 1: "CE", 2: "NE", 3: "CD"}

                if len(his_CG) == 0 or len(his_ND) == 0 or len(his_CE) == 0 or len(his_NE) == 0 or len(his_CD) == 0:
                    print('It looks like this residue is missing one or more of the histidine\'s imidazole ring heavy atoms')
                    print('skipping this residue from the analysis and removing its index from the unknown_residues list')
                    # delete known residues from the unknown residues list
                    branch_name.unknown_residues = np.delete(branch_name.unknown_residues, np.where(branch_name.unknown_residues == res_indx)[0][0])
                    print('############################################################')

                elif len(his_CG) == 1 and len(his_ND) == 1 and len(his_CE) == 1 and len(his_NE) == 1 and len(his_CD) == 1:
                    for i in np.arange(his_bond):
                        print('Atoms {} {} {} {} are the ND, CE, NE, CD atoms of {} (res_indx {}), respectively.'.format(his_ND[i], his_CE[i], his_NE[i], his_CD[i], branch_name.topology.atom(his_ND[i]).residue, res_indx))

                        # We will use grease carbon atoms and still available known don and acc for HIS neighboring calculations.
                        branch_name.G_and_allpos_prot_don_acc = np.sort(np.append(branch_name.grease, branch_name.allpos_prot_don_acc))

                        his_ND_nn = md.compute_neighbors(branch_name.structure, h_bond_heavy_atm_cutoff, np.asarray([his_ND[i]]), branch_name.G_and_allpos_prot_don_acc, periodic=False)[0]
                        his_CE_nn = md.compute_neighbors(branch_name.structure, h_bond_heavy_atm_cutoff, np.asarray([his_CE[i]]), branch_name.G_and_allpos_prot_don_acc, periodic=False)[0]
                        his_NE_nn = md.compute_neighbors(branch_name.structure, h_bond_heavy_atm_cutoff, np.asarray([his_NE[i]]), branch_name.G_and_allpos_prot_don_acc, periodic=False)[0]
                        his_CD_nn = md.compute_neighbors(branch_name.structure, h_bond_heavy_atm_cutoff, np.asarray([his_CD[i]]), branch_name.G_and_allpos_prot_don_acc, periodic=False)[0]

                        hisRname = branch_name.topology.atom(his_ND[i]).residue

                        ########################################################################################################

                        # Delete ND, CE, NE, CD atoms from each other's neighbor lists.
                        his_atm_ndx = np.empty((0,), dtype=int)
                        his_atm_ndx = np.append(his_atm_ndx, (his_ND[i], his_CE[i], his_NE[i], his_CD[i]))

                        for j in his_atm_ndx:
                            his_ND_nn = np.delete(his_ND_nn, np.where(his_ND_nn == j)[0])
                            his_CE_nn = np.delete(his_CE_nn, np.where(his_CE_nn == j)[0])
                            his_NE_nn = np.delete(his_NE_nn, np.where(his_NE_nn == j)[0])
                            his_CD_nn = np.delete(his_CD_nn, np.where(his_CD_nn == j)[0])

                        print('Initial his_ND_nn (neighbors list to ND atom) is {}'.format(his_ND_nn))
                        print('Initial his_CE_nn (neighbors list to CE atom) is {}'.format(his_CE_nn))
                        print('Initial his_NE_nn (neighbors list to NE atom) is {}'.format(his_NE_nn))
                        print('Initial his_CD_nn (neighbors list to CD atom) is {}'.format(his_CD_nn))
                        print('########################')

                        ########################################################################################################

                        # Delete from the neighbor lists the neighbor that can NOT make H-bond anymore. Those can be:
                        # 1 - backbone N or O that is not available anymore for H-bond
                        # OR 2 - side chain atoms whose residue is not in unknown_residues list (means they were already identified but no longer able to H-bond as they were used already and not in known_don or known_acc lists).
                        for k in his_ND_nn:
                            if list(branch_name.topology.atom(k).element)[2] != 'C' and k not in branch_name.known_don and k not in branch_name.known_acc and ((branch_name.topology.atom(k).name == 'N' or branch_name.topology.atom(k).name == 'O') or ((branch_name.topology.atom(k).name != 'N' or branch_name.topology.atom(k).name != 'O') and branch_name.topology.atom(k).residue.index not in branch_name.unknown_residues)):
                                his_ND_nn = np.delete(his_ND_nn, np.where(his_ND_nn == k)[0][0])
                        for k in his_CE_nn:
                            if list(branch_name.topology.atom(k).element)[2] != 'C' and k not in branch_name.known_don and k not in branch_name.known_acc and ((branch_name.topology.atom(k).name == 'N' or branch_name.topology.atom(k).name == 'O') or ((branch_name.topology.atom(k).name != 'N' or branch_name.topology.atom(k).name != 'O') and branch_name.topology.atom(k).residue.index not in branch_name.unknown_residues)):
                                his_CE_nn = np.delete(his_CE_nn, np.where(his_CE_nn == k)[0][0])
                        for k in his_NE_nn:
                            if list(branch_name.topology.atom(k).element)[2] != 'C' and k not in branch_name.known_don and k not in branch_name.known_acc and ((branch_name.topology.atom(k).name == 'N' or branch_name.topology.atom(k).name == 'O') or ((branch_name.topology.atom(k).name != 'N' or branch_name.topology.atom(k).name != 'O') and branch_name.topology.atom(k).residue.index not in branch_name.unknown_residues)):
                                his_NE_nn = np.delete(his_NE_nn, np.where(his_NE_nn == k)[0][0])
                        for k in his_CD_nn:
                            if list(branch_name.topology.atom(k).element)[2] != 'C' and k not in branch_name.known_don and k not in branch_name.known_acc and ((branch_name.topology.atom(k).name == 'N' or branch_name.topology.atom(k).name == 'O') or ((branch_name.topology.atom(k).name != 'N' or branch_name.topology.atom(k).name != 'O') and branch_name.topology.atom(k).residue.index not in branch_name.unknown_residues)):
                                his_CD_nn = np.delete(his_CD_nn, np.where(his_CD_nn == k)[0][0])

                        print('{} (Residue index {}) atoms\' neighbors after deletion of their own atoms in the nn lists and unavailable neighbors (while they were knowns, they are already in H-bonds and can\'t make H-bonds now...):'.format(branch_name.topology.atom(his_ND[i]).residue, branch_name.topology.atom(his_ND[i]).residue.index))
                        print('Neighbors list to ND atom is {}'.format(his_ND_nn))
                        print('Neighbors list to CE atom is {}'.format(his_CE_nn))
                        print('Neighbors list to NE atom is {}'.format(his_NE_nn))
                        print('Neighbors list to CD atom is {}'.format(his_CD_nn))
                        print('########################')
                        ########################################################################################################

                        ########################################################################################################
                        # filter out all non H bond interactions neighbors for which distance between the heavy atoms is 2.5 A or less
                        for k in his_ND_nn:
                            dist_nd_neighbor = math.sqrt(((10 * branch_name.xyz[0][k][0] - 10 * branch_name.xyz[0][his_ND[i]][0]) ** 2) + ((10 * branch_name.xyz[0][k][1] - 10 * branch_name.xyz[0][his_ND[i]][1]) ** 2) + ((10 * branch_name.xyz[0][k][2] - 10 * branch_name.xyz[0][his_ND[i]][2]) ** 2))
                            if dist_nd_neighbor < 2.5:
                                his_ND_nn = np.delete(his_ND_nn, np.where(his_ND_nn == k)[0][0])

                        for k in his_CE_nn:
                            dist_ce_neighbor = math.sqrt(((10 * branch_name.xyz[0][k][0] - 10 * branch_name.xyz[0][his_CE[i]][0]) ** 2) + ((10 * branch_name.xyz[0][k][1] - 10 * branch_name.xyz[0][his_CE[i]][1]) ** 2) + ((10 * branch_name.xyz[0][k][2] - 10 * branch_name.xyz[0][his_CE[i]][2]) ** 2))
                            if dist_ce_neighbor < 2.5:
                                his_CE_nn = np.delete(his_CE_nn, np.where(his_CE_nn == k)[0][0])

                        for k in his_NE_nn:
                            dist_ne_neighbor = math.sqrt(((10 * branch_name.xyz[0][k][0] - 10 * branch_name.xyz[0][his_NE[i]][0]) ** 2) + ((10 * branch_name.xyz[0][k][1] - 10 * branch_name.xyz[0][his_NE[i]][1]) ** 2) + ((10 * branch_name.xyz[0][k][2] - 10 * branch_name.xyz[0][his_NE[i]][2]) ** 2))
                            if dist_ne_neighbor < 2.5:
                                his_NE_nn = np.delete(his_NE_nn, np.where(his_NE_nn == k)[0][0])

                        for k in his_CD_nn:
                            dist_cd_neighbor = math.sqrt(((10 * branch_name.xyz[0][k][0] - 10 * branch_name.xyz[0][his_CD[i]][0]) ** 2) + ((10 * branch_name.xyz[0][k][1] - 10 * branch_name.xyz[0][his_CD[i]][1]) ** 2) + ((10 * branch_name.xyz[0][k][2] - 10 * branch_name.xyz[0][his_CD[i]][2]) ** 2))
                            if dist_cd_neighbor < 2.5:
                                his_CD_nn = np.delete(his_CD_nn, np.where(his_CD_nn == k)[0][0])

                        print('{} (Residue index {}) atoms\' neighbors after dropping neighbors with distances lower than 2.5 A:'.format(branch_name.topology.atom(his_ND[i]).residue, branch_name.topology.atom(his_ND[i]).residue.index))
                        print('Neighbors list to ND atom is {}'.format(his_ND_nn))
                        print('Neighbors list to CE atom is {}'.format(his_CE_nn))
                        print('Neighbors list to NE atom is {}'.format(his_NE_nn))
                        print('Neighbors list to CD atom is {}'.format(his_CD_nn))
                        print('########################')
                        ########################################################################################################

                        ########################################################################################################
                        # We filter out all non H bond interactions neighbors

                        his_ND_nn = filter_out_non_hbond_neighbors(branch_name, his_ND_nn, his_atm_ndx[0])
                        his_CE_nn = filter_out_non_hbond_neighbors(branch_name, his_CE_nn, his_atm_ndx[1])
                        his_NE_nn = filter_out_non_hbond_neighbors(branch_name, his_NE_nn, his_atm_ndx[2])
                        his_CD_nn = filter_out_non_hbond_neighbors(branch_name, his_CD_nn, his_atm_ndx[3])

                        print('These are the list of neighbors after filtering out neighbors that are not making H-bond interactions due to bad angles:')
                        print('his_ND_nn (neighbors list to ND atom) is {}'.format(his_ND_nn))
                        print('his_CE_nn (neighbors list to CE atom) is {}'.format(his_CE_nn))
                        print('his_NE_nn (neighbors list to NE atom) is {}'.format(his_NE_nn))
                        print('his_CD_nn (neighbors list to CD atom) is {}'.format(his_CD_nn))
                        print('########################')
                        ########################################################################################################

                        # Finalizing neighbor list of ND atom
                        his_ND_nn_grease = []
                        his_ND_nn_unknown = []
                        his_ND_nn_known = []
                        for k in his_ND_nn:
                            if k in branch_name.known_don or k in branch_name.known_acc:
                                his_ND_nn_known.append(k)
                            elif k in branch_name.grease:
                                his_ND_nn_grease.append(k)
                                his_ND_nn = np.delete(his_ND_nn, np.where(his_ND_nn == k)[0][0])
                            else:
                                his_ND_nn_unknown.append(k)
                                his_ND_nn = np.delete(his_ND_nn, np.where(his_ND_nn == k)[0][0])

                        print('his_ND_nn_grease is {}'.format(his_ND_nn_grease))
                        print('his_ND_nn_unknown is {}'.format(his_ND_nn_unknown))
                        print('his_ND_nn_known is {}'.format(his_ND_nn_known))
                        print('########################')
                        ########################################################################################################
                        # Finalizing neighbor list of CE atom

                        his_CE_nn_grease = []
                        his_CE_nn_unknown = []
                        his_CE_nn_known = []

                        for k in his_CE_nn:
                            if k in branch_name.known_don or k in branch_name.known_acc:
                                his_CE_nn_known.append(k)
                            elif k in branch_name.grease:
                                his_CE_nn_grease.append(k)
                                his_CE_nn = np.delete(his_CE_nn, np.where(his_CE_nn == k)[0][0])
                            else:
                                his_CE_nn_unknown.append(k)
                                his_CE_nn = np.delete(his_CE_nn, np.where(his_CE_nn == k)[0][0])

                        print('his_CE_nn_grease is {}'.format(his_CE_nn_grease))
                        print('his_CE_nn_unknown is {}'.format(his_CE_nn_unknown))
                        print('his_CE_nn_known is {}'.format(his_CE_nn_known))
                        print('########################')
                        ########################################################################################################
                        # Finalizing neighbor list of NE atom

                        his_NE_nn_grease = []
                        his_NE_nn_unknown = []
                        his_NE_nn_known = []

                        for k in his_NE_nn:
                            if k in branch_name.known_don or k in branch_name.known_acc:
                                his_NE_nn_known.append(k)
                            elif k in branch_name.grease:
                                his_NE_nn_grease.append(k)
                                his_NE_nn = np.delete(his_NE_nn, np.where(his_NE_nn == k)[0][0])
                            else:
                                his_NE_nn_unknown.append(k)
                                his_NE_nn = np.delete(his_NE_nn, np.where(his_NE_nn == k)[0][0])

                        print('his_NE_nn_grease is {}'.format(his_NE_nn_grease))
                        print('his_NE_nn_unknown is {}'.format(his_NE_nn_unknown))
                        print('his_NE_nn_known is {}'.format(his_NE_nn_known))
                        print('########################')
                        ########################################################################################################
                        # Finalizing neighbor list of CD atom

                        his_CD_nn_grease = []
                        his_CD_nn_unknown = []
                        his_CD_nn_known = []

                        for k in his_CD_nn:
                            if k in branch_name.known_don or k in branch_name.known_acc:
                                his_CD_nn_known.append(k)
                            elif k in branch_name.grease:
                                his_CD_nn_grease.append(k)
                                his_CD_nn = np.delete(his_CD_nn, np.where(his_CD_nn == k)[0][0])
                            else:
                                his_CD_nn_unknown.append(k)
                                his_CD_nn = np.delete(his_CD_nn, np.where(his_CD_nn == k)[0][0])

                        print('his_CD_nn_grease is {}'.format(his_CD_nn_grease))
                        print('his_CD_nn_unknown is {}'.format(his_CD_nn_unknown))
                        print('his_CD_nn_known is {}'.format(his_CD_nn_known))
                        print('########################')
                        ############################################################################

                        # Categorizing the type of neighbor (type should be one of the following: acc, don, gre, emt, both don/acc, don/acc/gre,)
                        # Testing to check if I have at least one of the ring atoms neighbored with a 'known' donor or a 'known' acceptor.
                        # This is needed because while his_ND_nn can have a neighbor, it could be a grease atom which doesn't help us identify the residue.
                        # There needs to be AT LEAST ONE known acc/don atom neighbored to the residue's atom to be switched from unknown to known according to meeting with Dan and Tom on 3/30/2021
                        # According to 4/14/2021 meeting with Dan and Tom, if there is at least one known neighbor to to the 4 ring atoms and some unknowns as well, we still use the unknowns to see if we can set the residue:
                        # An unknown neighbor that's a SER/THR/TYR residue can act as either one donor (-OH) or one acceptor(-OH's lone pair).
                        # An unknown neighbor that's a ASN/GLN residue can act as either one donor (side chain amide N-H) or one acceptor(side chain amide O' lone pair).
                        # An unknown neighbor that's a HIS residue can act as either one donor (N-H), one acceptor (N's lone pair) or one grease (C-H)

                        his_atm_nn = []

                        his_atm_nn.append(flatten([his_ND_nn_known, his_ND_nn_unknown]))
                        his_atm_nn.append(flatten([his_CE_nn_known, his_CE_nn_unknown]))
                        his_atm_nn.append(flatten([his_NE_nn_known, his_NE_nn_unknown]))
                        his_atm_nn.append(flatten([his_CD_nn_known, his_CD_nn_unknown]))

                        known_neighbors_count = len(flatten([list(his_ND_nn_known), list(his_CE_nn_known), list(his_NE_nn_known), list(his_CD_nn_known)]))

                        for ndx, j in enumerate(his_atm_nn):
                            if len(j) == 0:
                                his_atm_nn[ndx] = 'E'

                        ############################################################################
                        if known_neighbors_count == 0 and (len(his_ND_nn_unknown) > 0 or len(his_CE_nn_unknown) > 0 or len(his_NE_nn_unknown) > 0 or len(his_CD_nn_unknown) > 0) and (len(his_ND_nn_unknown) + len(his_CE_nn_unknown) + len(his_NE_nn_unknown) + len(his_CD_nn_unknown)) > 1:
                            # This residue will remain as unknown until end of the first pass.
                            print('The 4 ring atoms (ND, CE, NE, CD) are NOT neighbored with any \'known\' neighbors and at least one of them is neighbored with an unknown residue\'s atom within the heavy-to-heavy atom distance cutoff.')
                            print('The residue {} will remain \'unknown\' until either identified by a \'known\' atom or branched out.'.format(branch_name.topology.atom(his_ND[i]).residue))
                            print('\n############################################################')
                            continue  # this will stop the current looped residue and without continuing this iteration goes next to the next iteration.

                        else:
                            for k, his_X_nn in enumerate(his_atm_nn):
                                if len(his_X_nn) == 1 and his_X_nn == 'E':
                                    his_X_nn_type = 'E'
                                elif len(his_X_nn) == 1:
                                    if (his_X_nn[0] in branch_name.known_acc) and (his_X_nn[0] not in branch_name.known_don):
                                        his_X_nn_type = 'A'
                                    elif (his_X_nn[0] in branch_name.known_don) and (his_X_nn[0] not in branch_name.known_acc):
                                        his_X_nn_type = 'D'
                                    elif his_X_nn[0] in branch_name.grease:
                                        his_X_nn_type = 'G'
                                    elif his_X_nn[0] not in branch_name.known_acc and his_X_nn[0] not in branch_name.known_don and his_X_nn[0] not in branch_name.grease:  # when neighbored to unknown residues, I need to consider the different types (don/acc/gre) the neighbor can be to make a sound decision
                                        if branch_name.topology.atom(his_X_nn[0]).residue.name in ['SER', 'THR', 'TYR', 'ASN', 'GLN']:  # unknown SER/THR/TYR/ASN/GLN
                                            his_X_nn_type = ['D/A_unknown']
                                        elif branch_name.topology.atom(his_X_nn[0]).residue.name in ['HIS', 'HIE', 'HID', 'HIP']:  # unknown HIS
                                            his_X_nn_type = ['D/A/G_unknown']
                                    elif his_X_nn[0] in branch_name.known_acc and his_X_nn[0] in branch_name.known_don:  # when neighbored to a heavy atom that acts as both do and acc.
                                        his_X_nn_type = ['D/A_known']
                                elif len(his_X_nn) > 1:  # multiple known/unknown neighbors
                                    his_X_nn_type = []
                                    for j in his_X_nn:
                                        if (j in branch_name.known_acc) and (j not in branch_name.known_don):
                                            his_X_nn_type.append('A')
                                        elif (j in branch_name.known_don) and (j not in branch_name.known_acc):
                                            his_X_nn_type.append('D')
                                        elif j in branch_name.grease:
                                            his_X_nn_type.append('G')
                                        elif j not in branch_name.known_acc and j not in branch_name.known_don and j not in branch_name.grease:  # when neighbored to unknown residues, I need to consider the different types (don/acc/gre) the neighbor can be to make a sound decision
                                            if branch_name.topology.atom(j).residue.name in ['SER', 'THR', 'TYR', 'ASN', 'GLN']:
                                                his_X_nn_type.append('D/A_unknown')
                                            elif branch_name.topology.atom(j).residue.name in ['HIS', 'HIE', 'HID', 'HIP']:
                                                his_X_nn_type.append('D/A/G_unknown')
                                        elif j in branch_name.known_acc and j in branch_name.known_don:  # when neighbored to a heavy atom that acts as both do and acc.
                                            his_X_nn_type.append('D/A_known')
                                his[str(hisRname)][his_atm_nm[k]] = his_X_nn_type

                            his_res = list(his.items())[0][0]
                            his_res_nn = list(his.items())[0][1]
                            print('{} \'s side chain ring atoms with type of their neighbor are: {}\n'.format(his_res, his_res_nn))

                            his_neighbor_configurations_by_atoms = list(itertools.product(*his_atm_nn))
                            his_neighbor_configurations_by_features = list(itertools.product(*list(his.values())[0].values()))
                            print('Number of neighbor configurations to be explored is {}\n'.format(len(his_neighbor_configurations_by_atoms)))
                            print('his_neighbor_configurations_by_atoms list is {}\n'.format(his_neighbor_configurations_by_atoms))
                            print('his_neighbor_configurations_by_features list is {}\n'.format(his_neighbor_configurations_by_features))
                            print('########################')
                            his_configuration_decisions_with_energies = defaultdict(dict)
                            his_configuration_winning_scenario_features_dict = defaultdict(dict)  # this dictionary will have keys as configuration index and values as the best scenario of this config (in case there was more than one scenario)
                            his_configuration_donorneighbors_used = defaultdict(dict)

                            for j, his_neighbor_set in enumerate(his_neighbor_configurations_by_atoms):
                                print('HIS neighbor configuration #{}'.format(j))
                                print('HIS neighbor configuration by atoms is {}'.format(his_neighbor_configurations_by_atoms[j]))
                                print('HIS neighbor configuration by features is {}'.format(his_neighbor_configurations_by_features[j]))

                                # Here we start by getting positions of 4 H's that are mediating the interaction between atoms 1-4 (ND, CE, NE, CD) of the imidazole ring of histidines and the neighbors
                                # If the neighbor is an ACC neighbor, H is to be attached to the atom of the ring. However, For donor neighbors, we will need to know the position of the hydrogen based on atom name, residue and heavy atom's hybridization.
                                # We need the H positions to get the NE/ND-H-----neighbor(acc) angle or NE/ND-----H-neighbor(don) angle then plug angle and distance between the two heavy atoms in the QM function to find interaction energy.
                                # https://stackoverflow.com/questions/27220219/how-to-find-a-third-point-given-both-2-points-on-a-line-and-distance-from-thi
                                coords_of_4_potential_mediating_hydrogens_between_his_atoms_and_neighbors = [0, 0, 0, 0]

                                for k in np.arange(4):
                                    if his_neighbor_configurations_by_features[j][k] == 'E' or his_neighbor_configurations_by_features[j][k] == 'G':
                                        coords_of_4_potential_mediating_hydrogens_between_his_atoms_and_neighbors[k] = "No H needed"  # No neighbor. No need to keep coordinates of mediating H.
                                    elif his_neighbor_configurations_by_features[j][k] == 'D':
                                        coords_of_4_potential_mediating_hydrogens_between_his_atoms_and_neighbors[k] = coords_of_don_neighbor_atom(branch_name, his_neighbor_set[k], his_atm_ndx[k])
                                    elif his_neighbor_configurations_by_features[j][k] == 'A':
                                        coords_of_4_potential_mediating_hydrogens_between_his_atoms_and_neighbors[k] = coords_of_don_neighbor_atom(branch_name, his_atm_ndx[k], his_neighbor_set[k])
                                    elif his_neighbor_configurations_by_features[j][k] in ['D/A_unknown', 'D/A/G_unknown', 'D/A_known']:
                                        # FOR THOSE THAT ACT AS 'D/A_unknown' or 'D/A_known'. There are 6 total items in there: three coordinates (x, y, z) for neighbor's H (as donor) and three coordinates (x, y, z) are for the H attached to the histidine's atom if neighbor is acting as acceptor.
                                        # FOR THOSE THAT ACT AS 'D/A/G_unknown'. There are 7 total items in there: three coordinates (x, y, z) for neighbor's H (as donor), three coordinates (x, y, z) are for the H attached to the histidine's atom if neighbor is acting as acceptor, and 'No H needed' which is the grease H atom which will make no difference in the individual energy

                                        ############################################ Unknown neighbor being a donor ############################################
                                        mediating_hydrogen_xyz_coords = coords_of_don_neighbor_atom(branch_name, his_neighbor_set[k], his_atm_ndx[k])

                                        if his_neighbor_configurations_by_features[j][k] == 'D/A_known' and len(flatten(mediating_hydrogen_xyz_coords)) > 3:  # in case the unknown atom has more than one donor
                                            dist_dic = {}
                                            for ndx, m in enumerate(mediating_hydrogen_xyz_coords):
                                                dist = math.sqrt(((branch_name.xyz[0][his_atm_ndx[k]][0] - m[0]) ** 2) + ((branch_name.xyz[0][his_atm_ndx[k]][1] - m[1]) ** 2) + ((branch_name.xyz[0][his_atm_ndx[k]][2] - m[2]) ** 2))
                                                dist_dic[ndx] = dist
                                            the_best_possible_H_atom_of_the_donor = mediating_hydrogen_xyz_coords[sorted(dist_dic.items(), key=operator.itemgetter(1))[0][0]]
                                        elif len(flatten(mediating_hydrogen_xyz_coords)) > 4:
                                            dist_dic = {}
                                            for ndx, m in enumerate(mediating_hydrogen_xyz_coords):
                                                dist = math.sqrt(((branch_name.xyz[0][his_atm_ndx[k]][0] - m[1][0]) ** 2) + ((branch_name.xyz[0][his_atm_ndx[k]][1] - m[1][1]) ** 2) + ((branch_name.xyz[0][his_atm_ndx[k]][2] - m[1][2]) ** 2))
                                                dist_dic[ndx] = dist
                                            the_best_possible_H_atom_of_the_donor = mediating_hydrogen_xyz_coords[sorted(dist_dic.items(), key=operator.itemgetter(1))[0][0]]
                                        else:
                                            the_best_possible_H_atom_of_the_donor = mediating_hydrogen_xyz_coords
                                        ########################################################################################################################

                                        ########################################## Unknown neighbor being an acceptor ##########################################
                                        H_of_the_his_atm_ndx = coords_of_don_neighbor_atom(branch_name, his_atm_ndx[k], his_neighbor_set[k])
                                        ########################################################################################################################

                                        coords_of_4_potential_mediating_hydrogens_between_his_atoms_and_neighbors[k] = [the_best_possible_H_atom_of_the_donor, H_of_the_his_atm_ndx]

                                        if his_neighbor_configurations_by_features[j][k] == 'D/A/G_unknown':
                                            coords_of_4_potential_mediating_hydrogens_between_his_atoms_and_neighbors[k].append("No H needed")

                                angles_of_4_atoms_of_his_mediatingH_neighbor = []
                                energies_of_4_atoms_of_his_mediatingH_neighbor = []

                                for k in np.arange(4):
                                    if his_neighbor_configurations_by_features[j][k] == 'E' or his_neighbor_configurations_by_features[j][k] == 'G':  # emt or grease
                                        print('No his_X---H---neighbor angle for atom {} of the ring. Either there is no N or O neighbors to interact with or it\'s just interacting with grease.'.format(k + 1))
                                        angles_of_4_atoms_of_his_mediatingH_neighbor.append('No Angle')
                                        print('       No his_X---H---neighbor energy for atom {} of the ring.'.format(k + 1))
                                        energies_of_4_atoms_of_his_mediatingH_neighbor.append(['No Energy'])
                                    else:
                                        if his_neighbor_configurations_by_features[j][k] in ['D/A_unknown', 'D/A/G_unknown', 'D/A_known']:
                                            # FOR THOSE THAT ACT AS 'D/A_unknown' or 'D/A_known'. There are 6 total items inside coords_of_4_potential_mediating_hydrogens_between_his_atoms_and_neighbors[k]:
                                            # three coordinates (x, y, z) for neighbor's H (as donor) and three coordinates (x, y, z) are for the H attached to the histidine's atom if neighbor is acting as acceptor.

                                            # FOR THOSE THAT ACT AS 'D/A/G_unknown'. There are 7 total items inside coords_of_4_potential_mediating_hydrogens_between_his_atoms_and_neighbors[k]:
                                            # three coordinates (x, y, z) for neighbor's H (as donor), three coordinates (x, y, z) are for the H attached to the histidine's atom if neighbor is acting as acceptor, and 'No H needed' which is the grease H atom which will make no difference in the individual energy

                                            angle_dic = {}
                                            energy_dic = {}
                                            for l in np.arange(len(coords_of_4_potential_mediating_hydrogens_between_his_atoms_and_neighbors[k])):
                                                # 0 is index in coords list that has a list of xyz coords of donor neighbor, # 1 is index in coords list that has a list of xyz coords of acceptor neighbor
                                                # 2 is index that is there only is the feature is 'D/A/G_unknown' and it refers to 'No H needed' which isn't important here since this part only gets angle of the interaction while 'No H needed' doesnt even refer to an interaction
                                                if l == 0 or l == 1:
                                                    his_X_xyz_coords = branch_name.xyz[0][his_atm_ndx[k]]
                                                    mediating_hydrogen_xyz_coords = coords_of_4_potential_mediating_hydrogens_between_his_atoms_and_neighbors[k][l]
                                                    if len(flatten(mediating_hydrogen_xyz_coords)) == 4:
                                                        mediating_hydrogen_xyz_coords = np.array(mediating_hydrogen_xyz_coords[1])
                                                    else:  # makes sure I have numpy array of this list so it doesnt give this error:
                                                        # "VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated.
                                                        # If you meant to do this, you must specify 'dtype=object' when creating the ndarray."
                                                        mediating_hydrogen_xyz_coords = np.array(mediating_hydrogen_xyz_coords)
                                                    his_neighbor_xyz_coords = branch_name.xyz[0][his_neighbor_set[k]]

                                                    vector_1 = his_X_xyz_coords - mediating_hydrogen_xyz_coords
                                                    vector_2 = his_neighbor_xyz_coords - mediating_hydrogen_xyz_coords
                                                    vector1_vector_2_cosine_angle = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
                                                    acc_H_don_angle_in_radians = np.arccos(vector1_vector_2_cosine_angle)
                                                    acc_H_don_angle_in_degrees = np.degrees(acc_H_don_angle_in_radians)

                                                    datoms = np.asarray([his_atm_ndx[k], his_neighbor_set[k]], dtype="int").reshape(1, 2)
                                                    dist = md.compute_distances(branch_name.structure, datoms)[0][0]

                                                    angle_dic[l] = acc_H_don_angle_in_degrees

                                                    if 180 >= round(acc_H_don_angle_in_degrees, 1) >= 90 and 0.50 >= round(dist, 2) >= 0.25:  # distance range of 2.5-5.0 A (0.25-0.5 nm) and angle range of 90-180 degrees were used for the PES calculation. Any distance or angle beyond those two ranges will be 0 kcal/mol because it's not a H-bond interaction
                                                        energy_dic[l] = PES_lookup_table(branch_name, dist, acc_H_don_angle_in_degrees)
                                                    else:
                                                        print('Atom {}:'.format(k + 1))
                                                        print('       The distance between the two heavy atoms is {} A, X---H---X angle is {}'.format(round(dist * 10, 3), round(acc_H_don_angle_in_degrees, 2)))
                                                        print('       The X---H---X angle is not between 90-180 degrees. \n'
                                                              '       Since this is not a H-bond interaction, we will keep energy of this interaction as 0 kcal/mol.')
                                                        energy_dic[l] = 0

                                            if his_neighbor_configurations_by_features[j][k] in ['D/A_unknown', 'D/A_known']:
                                                angles_of_4_atoms_of_his_mediatingH_neighbor.append([[his_neighbor_set[k], 'D', list(angle_dic.items())[0][1]], [his_neighbor_set[k], 'A', list(angle_dic.items())[1][1]]])
                                                energies_of_4_atoms_of_his_mediatingH_neighbor.append([[his_neighbor_set[k], 'D', list(energy_dic.items())[0][1]], [his_neighbor_set[k], 'A', list(energy_dic.items())[1][1]]])
                                            elif his_neighbor_configurations_by_features[j][k] == 'D/A/G_unknown':
                                                angles_of_4_atoms_of_his_mediatingH_neighbor.append([[his_neighbor_set[k], 'D', list(angle_dic.items())[0][1]], [his_neighbor_set[k], 'A', list(angle_dic.items())[1][1]], [his_neighbor_set[k], 'G', 'No Angle']])
                                                energies_of_4_atoms_of_his_mediatingH_neighbor.append([[his_neighbor_set[k], 'D', list(energy_dic.items())[0][1]], [his_neighbor_set[k], 'A', list(energy_dic.items())[1][1]], [his_neighbor_set[k], 'G', 'No Energy']])

                                            print('Atom {}: Calculated his_X---H---neighbor angle of the ring and they are {} and {} degrees, when considering the neighbor as a donor and as an acceptor, respectively.'.format(k + 1, list(angle_dic.items())[0][1], list(angle_dic.items())[1][1]))
                                            print('       Calculated his_X---H---neighbor energies for atom {} of the ring and they are {} and {} kcal/mol, when considering the neighbor as a donor and as an acceptor, respectively.'.format(k + 1, list(energy_dic.items())[0][1], list(energy_dic.items())[1][1]))

                                        elif len(flatten(coords_of_4_potential_mediating_hydrogens_between_his_atoms_and_neighbors[k])) == 3 or len(flatten(coords_of_4_potential_mediating_hydrogens_between_his_atoms_and_neighbors[k])) == 4:
                                            # 3 is an acc neighbor's 3 coords or a don neighbor's 3 coords such as (backbone N, ARG NE, TRP NE1).
                                            # 4 is any other don neighbor (except backbone N, ARG NE, TRP NE1, SER, THR, TYR) that while can have more than 1 H atom attached to it, only one is available at the moment.
                                            if len(flatten(coords_of_4_potential_mediating_hydrogens_between_his_atoms_and_neighbors[k])) == 3:
                                                # https://medium.com/@manivannan_data/find-the-angle-between-three-points-from-2d-using-python-348c513e2cd
                                                his_X_xyz_coords = branch_name.xyz[0][his_atm_ndx[k]]
                                                mediating_hydrogen_xyz_coords = np.array(coords_of_4_potential_mediating_hydrogens_between_his_atoms_and_neighbors[k])
                                                his_neighbor_xyz_coords = branch_name.xyz[0][his_neighbor_set[k]]

                                            elif len(flatten(coords_of_4_potential_mediating_hydrogens_between_his_atoms_and_neighbors[k])) == 4:
                                                # https://medium.com/@manivannan_data/find-the-angle-between-three-points-from-2d-using-python-348c513e2cd
                                                his_X_xyz_coords = branch_name.xyz[0][his_atm_ndx[k]]
                                                mediating_hydrogen_xyz_coords = np.array(coords_of_4_potential_mediating_hydrogens_between_his_atoms_and_neighbors[k][1])
                                                his_neighbor_xyz_coords = branch_name.xyz[0][his_neighbor_set[k]]

                                                # Here, I save the don neighbor H identification so we can put it in the reduced_topology_not_available_donors list for the winning configuration
                                                if his_neighbor_configurations_by_features[j][k] == 'D':
                                                    if len(his_configuration_donorneighbors_used[j]) == 0:  # if this config doesnt have any don neighbor yet
                                                        his_configuration_donorneighbors_used[j] = [coords_of_4_potential_mediating_hydrogens_between_his_atoms_and_neighbors[k][0]]
                                                    elif len(his_configuration_donorneighbors_used[j]) != 0 and coords_of_4_potential_mediating_hydrogens_between_his_atoms_and_neighbors[k][0] not in his_configuration_donorneighbors_used[j]:
                                                        his_configuration_donorneighbors_used[j].append(coords_of_4_potential_mediating_hydrogens_between_his_atoms_and_neighbors[k][0])

                                            vector_1 = his_X_xyz_coords - mediating_hydrogen_xyz_coords
                                            vector_2 = his_neighbor_xyz_coords - mediating_hydrogen_xyz_coords
                                            vector1_vector_2_cosine_angle = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
                                            acc_H_don_angle_in_radians = np.arccos(vector1_vector_2_cosine_angle)
                                            acc_H_don_angle_in_degrees = np.degrees(acc_H_don_angle_in_radians)

                                            angles_of_4_atoms_of_his_mediatingH_neighbor.append(acc_H_don_angle_in_degrees)
                                            print('Atom {}: Calculated his_X---H---neighbor angle of the ring and it\'s {} degrees'.format(k + 1, acc_H_don_angle_in_degrees))
                                            datoms = np.asarray([his_atm_ndx[k], his_neighbor_set[k]], dtype="int").reshape(1, 2)
                                            dist = md.compute_distances(branch_name.structure, datoms)[0][0]

                                            if 180 >= round(acc_H_don_angle_in_degrees, 1) >= 90 and 0.50 >= round(dist, 2) >= 0.25:  # distance range of 2.5-5.0 A (0.25-0.5 nm) and angle range of 90-180 degrees were used for the PES calculation. Any distance or angle beyond those two ranges will be 0 kcal/mol because it's not a H-bond interaction
                                                energies_of_4_atoms_of_his_mediatingH_neighbor.append([PES_lookup_table(branch_name, dist, acc_H_don_angle_in_degrees)])
                                                print('       Calculated his_X---H---neighbor energy for atom {} of the ring and it\'s {} kcal/mol'.format(k + 1, PES_lookup_table(branch_name, dist, acc_H_don_angle_in_degrees)))
                                            else:
                                                print('Atom {}:'.format(k + 1))
                                                print('       The distance between the two heavy atoms is {} A, X---H---X angle is {}'.format(round(dist * 10, 3), round(acc_H_don_angle_in_degrees, 2)))
                                                print('       The X---H---X angle is not between 90-180 degrees. \n'
                                                      '       Since this is not a H-bond interaction, we will keep energy of this interaction as 0 kcal/mol.')
                                                energies_of_4_atoms_of_his_mediatingH_neighbor.append([0])

                                        else:  # this else is when the neighbor is a donor that has physically more than one available donor to make the H-bond (such as LYS NZ (up to 3 H's), GLN NE2 (up to 2 H's), ASN ND2 (up to 2 H's), ARG NH2 (up to 2 H's), ARG NH1 (up to 2 H's))
                                            angle_dic = {}
                                            energy_dic = {}
                                            for l in np.arange(len(coords_of_4_potential_mediating_hydrogens_between_his_atoms_and_neighbors[k])):
                                                his_X_xyz_coords = branch_name.xyz[0][his_atm_ndx[k]]
                                                mediating_hydrogen_xyz_coords = np.array(coords_of_4_potential_mediating_hydrogens_between_his_atoms_and_neighbors[k][l][1])
                                                his_neighbor_xyz_coords = branch_name.xyz[0][his_neighbor_set[k]]

                                                vector_1 = his_X_xyz_coords - mediating_hydrogen_xyz_coords
                                                vector_2 = his_neighbor_xyz_coords - mediating_hydrogen_xyz_coords

                                                vector1_vector_2_cosine_angle = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
                                                acc_H_don_angle_in_radians = np.arccos(vector1_vector_2_cosine_angle)
                                                acc_H_don_angle_in_degrees = np.degrees(acc_H_don_angle_in_radians)

                                                datoms = np.asarray([his_atm_ndx[k], his_neighbor_set[k]], dtype="int").reshape(1, 2)
                                                dist = md.compute_distances(branch_name.structure, datoms)[0][0]

                                                angle_dic[l] = acc_H_don_angle_in_degrees

                                                if 180 >= round(acc_H_don_angle_in_degrees, 1) >= 90 and 0.50 >= round(dist, 2) >= 0.25:  # distance range of 2.5-5.0 A (0.25-0.5 nm) and angle range of 90-180 degrees were used for the PES calculation. Any distance or angle beyond those two ranges will be 0 kcal/mol because it's not a H-bond interaction
                                                    energy_dic[l] = PES_lookup_table(branch_name, dist, acc_H_don_angle_in_degrees)
                                                else:
                                                    print('Atom {}:'.format(k + 1))
                                                    print('       The distance between the two heavy atoms is {} A, X---H---X angle is {}'.format(round(dist * 10, 3), round(acc_H_don_angle_in_degrees, 2)))
                                                    print('       The X---H---X angle is not between 90-180 degrees. \n'
                                                          '       Since this is not a H-bond interaction, we will keep energy of this interaction as 0 kcal/mol.')
                                                    energy_dic[l] = 0

                                            bval = sorted(energy_dic.items(), key=operator.itemgetter(1))[0][0]
                                            chosen_angle = list(angle_dic.items())[bval][1]
                                            chosen_energy = list(energy_dic.items())[bval][1]

                                            # Here, I save the don neighbor H identification so we can put it in the reduced_topology_not_available_donors list for the winning configuration
                                            if len(his_configuration_donorneighbors_used[j]) == 0:  # if this config doesnt have any don neighbor yet
                                                his_configuration_donorneighbors_used[j] = [coords_of_4_potential_mediating_hydrogens_between_his_atoms_and_neighbors[k][bval][0]]
                                            elif len(his_configuration_donorneighbors_used[j]) != 0 and coords_of_4_potential_mediating_hydrogens_between_his_atoms_and_neighbors[k][bval][0] not in his_configuration_donorneighbors_used[j]:
                                                his_configuration_donorneighbors_used[j].append(coords_of_4_potential_mediating_hydrogens_between_his_atoms_and_neighbors[k][bval][0])

                                            angles_of_4_atoms_of_his_mediatingH_neighbor.append(chosen_angle)
                                            print('Atom {}: Calculated his_X---H---neighbor angle of the ring and it\'s {} degrees'.format(k + 1, chosen_angle))
                                            energies_of_4_atoms_of_his_mediatingH_neighbor.append([chosen_energy])
                                            print('       Calculated his_X---H---neighbor energy for atom {} of the ring and it\'s {} kcal/mol'.format(k + 1, chosen_energy))

                                if len(his_neighbor_configurations_by_atoms) == 1 and (his_neighbor_configurations_by_features[j][0] == 'E' or his_neighbor_configurations_by_features[j][0] == 'G') and (his_neighbor_configurations_by_features[j][1] == 'E' or his_neighbor_configurations_by_features[j][1] == 'G') and (his_neighbor_configurations_by_features[j][2] == 'E' or his_neighbor_configurations_by_features[j][2] == 'G') and (his_neighbor_configurations_by_features[j][3] == 'E' or his_neighbor_configurations_by_features[j][3] == 'G'):
                                    print('Since this HIS residue\'s 4 ring atoms neighbored with either grease or nothing, we will identify this residue as HIE13. No need to calculate rotamer energies')
                                else:
                                    if len(flatten(energies_of_4_atoms_of_his_mediatingH_neighbor)) == 4:
                                        ############################################################################
                                        # Calculate energies of the HIE13, HID13, HIE24, HIE24 rotamers.
                                        if his_neighbor_configurations_by_features[j][0] == 'D':
                                            HIE13 = energies_of_4_atoms_of_his_mediatingH_neighbor[0][0]  # favorable
                                            HID13 = -(energies_of_4_atoms_of_his_mediatingH_neighbor[0][0])  # clash

                                        elif his_neighbor_configurations_by_features[j][0] == 'A':
                                            HIE13 = -(energies_of_4_atoms_of_his_mediatingH_neighbor[0][0])  # clash
                                            HID13 = energies_of_4_atoms_of_his_mediatingH_neighbor[0][0]  # favorable
                                        else:
                                            HIE13 = 0
                                            HID13 = 0

                                        if his_neighbor_configurations_by_features[j][2] == 'D':
                                            HIE13 += -(energies_of_4_atoms_of_his_mediatingH_neighbor[2][0])  # clash
                                            HID13 += energies_of_4_atoms_of_his_mediatingH_neighbor[2][0]  # favorable
                                        elif his_neighbor_configurations_by_features[j][2] == 'A':
                                            HIE13 += energies_of_4_atoms_of_his_mediatingH_neighbor[2][0]  # favorable
                                            HID13 += -(energies_of_4_atoms_of_his_mediatingH_neighbor[2][0])  # clash
                                        else:
                                            HIE13 += 0
                                            HID13 += 0

                                        if his_neighbor_configurations_by_features[j][1] == 'D':
                                            HIE24 = -(energies_of_4_atoms_of_his_mediatingH_neighbor[1][0])  # clash
                                            HID24 = energies_of_4_atoms_of_his_mediatingH_neighbor[1][0]  # favorable
                                        elif his_neighbor_configurations_by_features[j][1] == 'A':
                                            HIE24 = energies_of_4_atoms_of_his_mediatingH_neighbor[1][0]  # favorable
                                            HID24 = -(energies_of_4_atoms_of_his_mediatingH_neighbor[1][0])  # clash
                                        else:
                                            HIE24 = 0
                                            HID24 = 0

                                        if his_neighbor_configurations_by_features[j][3] == 'D':
                                            HIE24 += energies_of_4_atoms_of_his_mediatingH_neighbor[3][0]  # favorable
                                            HID24 += -(energies_of_4_atoms_of_his_mediatingH_neighbor[3][0])  # clash
                                        elif his_neighbor_configurations_by_features[j][3] == 'A':
                                            HIE24 += -(energies_of_4_atoms_of_his_mediatingH_neighbor[3][0])  # clash
                                            HID24 += energies_of_4_atoms_of_his_mediatingH_neighbor[3][0]  # favorable
                                        else:
                                            HIE24 += 0
                                            HID24 += 0
                                        ############################################################################

                                        ############################################################################
                                        # We will include calculating energies of the HIP13 and HIP24 rotamers only under the conditions that there are two acceptors at position 1 and 3 or positions 2 and 4 of the ring, the h-to-h atoms distance below 3.4 A (0.34 nm) and the angles between 150 and 180 degrees (meaning 180 +- 30 degrees).
                                        #######################################
                                        # HIP13
                                        #######################################
                                        if his_neighbor_configurations_by_atoms[j][0] != 'E' and his_neighbor_configurations_by_atoms[j][2] != 'E':
                                            datoms = np.asarray([his_atm_ndx[0], his_neighbor_configurations_by_atoms[j][0]], dtype="int").reshape(1, 2)
                                            distance_at_atom_0 = md.compute_distances(branch_name.structure, datoms)[0][0]

                                            datoms = np.asarray([his_atm_ndx[2], his_neighbor_configurations_by_atoms[j][2]], dtype="int").reshape(1, 2)
                                            distance_at_atom_2 = md.compute_distances(branch_name.structure, datoms)[0][0]

                                        if his_neighbor_configurations_by_features[j][0] == 'A' and his_neighbor_configurations_by_features[j][2] == 'A':
                                            if 0.34 >= distance_at_atom_0 and 0.34 >= distance_at_atom_2 and 180 >= angles_of_4_atoms_of_his_mediatingH_neighbor[0] >= 150 and 180 >= angles_of_4_atoms_of_his_mediatingH_neighbor[2] >= 150:
                                                HIP13 = energies_of_4_atoms_of_his_mediatingH_neighbor[0][0]  # favorable
                                                HIP13 += energies_of_4_atoms_of_his_mediatingH_neighbor[2][0]  # favorable
                                            else:
                                                HIP13 = 'N/A'
                                        else:
                                            HIP13 = 'N/A'
                                        #######################################

                                        #######################################
                                        # HIP24
                                        #######################################
                                        if his_neighbor_configurations_by_atoms[j][1] != 'E' and his_neighbor_configurations_by_atoms[j][3] != 'E':
                                            datoms = np.asarray([his_atm_ndx[1], his_neighbor_configurations_by_atoms[j][1]], dtype="int").reshape(1, 2)
                                            distance_at_atom_1 = md.compute_distances(branch_name.structure, datoms)[0][0]

                                            datoms = np.asarray([his_atm_ndx[3], his_neighbor_configurations_by_atoms[j][3]], dtype="int").reshape(1, 2)
                                            distance_at_atom_3 = md.compute_distances(branch_name.structure, datoms)[0][0]

                                        if his_neighbor_configurations_by_features[j][1] == 'A' and his_neighbor_configurations_by_features[j][3] == 'A':
                                            if 0.34 >= distance_at_atom_1 and 0.34 >= distance_at_atom_3 and 180 >= angles_of_4_atoms_of_his_mediatingH_neighbor[1] >= 150 and 180 >= angles_of_4_atoms_of_his_mediatingH_neighbor[3] >= 150:
                                                HIP24 = energies_of_4_atoms_of_his_mediatingH_neighbor[1][0]  # favorable
                                                HIP24 += energies_of_4_atoms_of_his_mediatingH_neighbor[3][0]  # favorable
                                            else:
                                                HIP24 = 'N/A'
                                        else:
                                            HIP24 = 'N/A'
                                        #######################################
                                        ############################################################################

                                        ############################################################################
                                        # we will include HIP13 and HIP24 energies only if the conditions above were fulfilled
                                        if HIP13 == 'N/A' and HIP24 == 'N/A':
                                            his_possible_rotamers_energies = {'HIE13': HIE13, 'HID13': HID13, 'HIE24': HIE24, 'HID24': HID24}
                                        elif HIP13 != 'N/A' and HIP24 != 'N/A':
                                            his_possible_rotamers_energies = {'HIE13': HIE13, 'HID13': HID13, 'HIE24': HIE24, 'HID24': HID24, 'HIP13': HIP13, 'HIP24': HIP24}
                                        elif HIP13 != 'N/A':
                                            his_possible_rotamers_energies = {'HIE13': HIE13, 'HID13': HID13, 'HIE24': HIE24, 'HID24': HID24, 'HIP13': HIP13}
                                        elif HIP24 != 'N/A':
                                            his_possible_rotamers_energies = {'HIE13': HIE13, 'HID13': HID13, 'HIE24': HIE24, 'HID24': HID24, 'HIP24': HIP24}

                                        his_rotamers_ordered_in_energy = sorted(his_possible_rotamers_energies.items(), key=lambda x: x[1])
                                        his_rotamers_ordered_in_energy = [list(ele) for ele in his_rotamers_ordered_in_energy]

                                        print('HIS possible rotamers and their energies are {}.'.format(his_rotamers_ordered_in_energy))

                                        difference_in_energy_between_lowest_2_possible_rotamers = round(abs(his_rotamers_ordered_in_energy[0][1] - his_rotamers_ordered_in_energy[1][1]), 2)
                                        print('The difference in energy between the lowest two possible rotamers is {}'.format(difference_in_energy_between_lowest_2_possible_rotamers))

                                        if difference_in_energy_between_lowest_2_possible_rotamers >= degenerate_states_e_cutoff:
                                            print('Since the difference in energy between the lowest two possible rotamers for this neighbors configuration is equal or above the degenerate states E cutoff.')
                                            print('     Keeping only the lowest energetic rotamer for this configuration ({}): {}'.format(j, his_rotamers_ordered_in_energy[0]))
                                            his_configuration_decisions_with_energies[j] = [his_rotamers_ordered_in_energy[0]]

                                        else:
                                            print('Since the difference in energy between the lowest two possible rotamers for this neighbors configuration is within the degenerate states E cutoff, we will keep both HIS possibilities!!')
                                            his_configuration_decisions_with_energies[j] = [his_rotamers_ordered_in_energy[0], his_rotamers_ordered_in_energy[1]]

                                    else:
                                        energies_of_4_atoms_of_his_mediatingH_neighbor = list(itertools.product(*energies_of_4_atoms_of_his_mediatingH_neighbor))
                                        all_possibilities_lowest_rotamers = defaultdict(dict)
                                        for index_of_m, m in enumerate(energies_of_4_atoms_of_his_mediatingH_neighbor):
                                            ############################################################################
                                            # Calculate energies of the HIE13, HID13, HIE24, HIE24 rotamers.
                                            if his_neighbor_configurations_by_features[j][0] == 'D':
                                                HIE13 = m[0]  # favorable
                                                HID13 = -(m[0])  # clash
                                            elif his_neighbor_configurations_by_features[j][0] == 'A':
                                                HIE13 = -(m[0])  # clash
                                                HID13 = m[0]  # favorable
                                            elif his_neighbor_configurations_by_features[j][0] in ['D/A_unknown', 'D/A/G_unknown', 'D/A_known']:
                                                if m[0][1] == 'D':
                                                    HIE13 = m[0][2]  # favorable
                                                    HID13 = -(m[0][2])  # clash
                                                elif m[0][1] == 'A':
                                                    HIE13 = -(m[0][2])  # clash
                                                    HID13 = m[0][2]  # favorable
                                                else:
                                                    HIE13 = 0
                                                    HID13 = 0
                                            else:
                                                HIE13 = 0
                                                HID13 = 0

                                            if his_neighbor_configurations_by_features[j][2] == 'D':
                                                HIE13 += -(m[2])  # clash
                                                HID13 += m[2]  # favorable
                                            elif his_neighbor_configurations_by_features[j][2] == 'A':
                                                HIE13 += m[2]  # favorable
                                                HID13 += -(m[2])  # clash
                                            elif his_neighbor_configurations_by_features[j][2] in ['D/A_unknown', 'D/A/G_unknown', 'D/A_known']:
                                                if m[2][1] == 'D':
                                                    HIE13 += -(m[2][2])  # clash
                                                    HID13 += m[2][2]  # favorable
                                                elif m[2][1] == 'A':
                                                    HIE13 += m[2][2]  # favorable
                                                    HID13 += -(m[2][2])  # clash
                                                else:
                                                    HIE13 += 0
                                                    HID13 += 0
                                            else:
                                                HIE13 += 0
                                                HID13 += 0

                                            if his_neighbor_configurations_by_features[j][1] == 'D':
                                                HIE24 = -(m[1])  # clash
                                                HID24 = m[1]  # favorable
                                            elif his_neighbor_configurations_by_features[j][1] == 'A':
                                                HIE24 = m[1]  # favorable
                                                HID24 = -(m[1])  # clash
                                            elif his_neighbor_configurations_by_features[j][1] in ['D/A_unknown', 'D/A/G_unknown', 'D/A_known']:
                                                if m[1][1] == 'D':
                                                    HIE24 = -(m[1][2])  # clash
                                                    HID24 = m[1][2]  # favorable
                                                elif m[1][1] == 'A':
                                                    HIE24 = m[1][2]  # favorable
                                                    HID24 = -(m[1][2])  # clash
                                                else:
                                                    HIE24 = 0
                                                    HID24 = 0
                                            else:
                                                HIE24 = 0
                                                HID24 = 0

                                            if his_neighbor_configurations_by_features[j][3] == 'D':
                                                HIE24 += m[3]  # favorable
                                                HID24 += -(m[3])  # clash
                                            elif his_neighbor_configurations_by_features[j][3] == 'A':
                                                HIE24 += -(m[3])  # clash
                                                HID24 += m[3]  # favorable
                                            elif his_neighbor_configurations_by_features[j][3] in ['D/A_unknown', 'D/A/G_unknown', 'D/A_known']:
                                                if m[3][1] == 'D':
                                                    HIE24 += m[3][2]  # favorable
                                                    HID24 += -(m[3][2])  # clash
                                                elif m[3][1] == 'A':
                                                    HIE24 += -(m[3][2])  # clash
                                                    HID24 += m[3][2]  # favorable
                                                else:
                                                    HIE24 += 0
                                                    HID24 += 0
                                            else:
                                                HIE24 += 0
                                                HID24 += 0
                                            ############################################################################

                                            ############################################################################
                                            # We will include calculating energies of the HIP13 and HIP24 rotamers only under the conditions that there are two acceptors at position 1 and 3 or positions 2 and 4 of the ring, the h-to-h atoms distance below 3.4 A (0.34 nm) and the angles between 150 and 180 degrees.
                                            #######################################
                                            # HIP13
                                            #######################################
                                            if his_neighbor_configurations_by_atoms[j][0] != 'E' and his_neighbor_configurations_by_atoms[j][2] != 'E':
                                                datoms = np.asarray([his_atm_ndx[0], his_neighbor_configurations_by_atoms[j][0]], dtype="int").reshape(1, 2)
                                                distance_at_atom_0 = md.compute_distances(branch_name.structure, datoms)[0][0]

                                                datoms = np.asarray([his_atm_ndx[2], his_neighbor_configurations_by_atoms[j][2]], dtype="int").reshape(1, 2)
                                                distance_at_atom_2 = md.compute_distances(branch_name.structure, datoms)[0][0]

                                            if (his_neighbor_configurations_by_features[j][0] == 'A' or ((his_neighbor_configurations_by_features[j][0] in ['D/A_unknown', 'D/A/G_unknown', 'D/A_known']) and m[0][1] == 'A')) and (his_neighbor_configurations_by_features[j][2] == 'A' or ((his_neighbor_configurations_by_features[j][2] in ['D/A_unknown', 'D/A/G_unknown', 'D/A_known']) and m[2][1] == 'A')):
                                                if 0.34 >= distance_at_atom_0 and 0.34 >= distance_at_atom_2 and ((his_neighbor_configurations_by_features[j][0] == 'A' and 180 >= angles_of_4_atoms_of_his_mediatingH_neighbor[0] >= 150) or ((his_neighbor_configurations_by_features[j][0] in ['D/A_unknown', 'D/A/G_unknown', 'D/A_known']) and m[0][1] == 'A' and 180 >= angles_of_4_atoms_of_his_mediatingH_neighbor[0][1][2] >= 150)) and (
                                                        (his_neighbor_configurations_by_features[j][2] == 'A' and 180 >= angles_of_4_atoms_of_his_mediatingH_neighbor[2] >= 150) or ((his_neighbor_configurations_by_features[j][2] in ['D/A_unknown', 'D/A/G_unknown', 'D/A_known']) and m[2][1] == 'A' and 180 >= angles_of_4_atoms_of_his_mediatingH_neighbor[2][1][2] >= 150)):
                                                    if his_neighbor_configurations_by_features[j][0] == 'A' and 180 >= angles_of_4_atoms_of_his_mediatingH_neighbor[0] >= 150:
                                                        HIP13 = m[0]  # favorable
                                                    elif (his_neighbor_configurations_by_features[j][0] in ['D/A_unknown', 'D/A/G_unknown', 'D/A_known']) and m[0][1] == 'A' and 180 >= angles_of_4_atoms_of_his_mediatingH_neighbor[0][1][2] >= 150:
                                                        HIP13 = m[0][2]  # favorable

                                                    if his_neighbor_configurations_by_features[j][2] == 'A' and 180 >= angles_of_4_atoms_of_his_mediatingH_neighbor[2] >= 150:
                                                        HIP13 += m[2]  # favorable
                                                    elif his_neighbor_configurations_by_features[j][2] in ['D/A_unknown', 'D/A/G_unknown', 'D/A_known'] and m[2][1] == 'A' and 180 >= angles_of_4_atoms_of_his_mediatingH_neighbor[2][1][2] >= 150:
                                                        HIP13 += m[2][2]  # favorable

                                                else:
                                                    HIP13 = 'N/A'
                                            else:
                                                HIP13 = 'N/A'
                                            #######################################

                                            #######################################
                                            # HIP24
                                            #######################################
                                            if his_neighbor_configurations_by_atoms[j][1] != 'E' and his_neighbor_configurations_by_atoms[j][3] != 'E':
                                                datoms = np.asarray([his_atm_ndx[1], his_neighbor_configurations_by_atoms[j][1]], dtype="int").reshape(1, 2)
                                                distance_at_atom_1 = md.compute_distances(branch_name.structure, datoms)[0][0]

                                                datoms = np.asarray([his_atm_ndx[3], his_neighbor_configurations_by_atoms[j][3]], dtype="int").reshape(1, 2)
                                                distance_at_atom_3 = md.compute_distances(branch_name.structure, datoms)[0][0]

                                            if (his_neighbor_configurations_by_features[j][1] == 'A' or ((his_neighbor_configurations_by_features[j][1] in ['D/A_unknown', 'D/A/G_unknown', 'D/A_known']) and m[1][1] == 'A')) and (his_neighbor_configurations_by_features[j][3] == 'A' or ((his_neighbor_configurations_by_features[j][3] in ['D/A_unknown', 'D/A/G_unknown', 'D/A_known']) and m[3][1] == 'A')):
                                                if 0.34 >= distance_at_atom_1 and 0.34 >= distance_at_atom_3 and ((his_neighbor_configurations_by_features[j][1] == 'A' and 180 >= angles_of_4_atoms_of_his_mediatingH_neighbor[1] >= 150) or ((his_neighbor_configurations_by_features[j][1] in ['D/A_unknown', 'D/A/G_unknown', 'D/A_known']) and m[1][1] == 'A' and 180 >= angles_of_4_atoms_of_his_mediatingH_neighbor[1][1][2] >= 150)) and (
                                                        (his_neighbor_configurations_by_features[j][3] == 'A' and 180 >= angles_of_4_atoms_of_his_mediatingH_neighbor[3] >= 150) or ((his_neighbor_configurations_by_features[j][3] in ['D/A_unknown', 'D/A/G_unknown', 'D/A_known']) and m[3][1] == 'A' and 180 >= angles_of_4_atoms_of_his_mediatingH_neighbor[3][1][2] >= 150)):
                                                    if his_neighbor_configurations_by_features[j][1] == 'A' and 180 >= angles_of_4_atoms_of_his_mediatingH_neighbor[1] >= 150:
                                                        HIP24 = m[1]  # favorable
                                                    elif (his_neighbor_configurations_by_features[j][1] in ['D/A_unknown', 'D/A/G_unknown', 'D/A_known']) and m[1][1] == 'A' and 180 >= angles_of_4_atoms_of_his_mediatingH_neighbor[1][1][2] >= 150:
                                                        HIP24 = m[1][2]  # favorable

                                                    if his_neighbor_configurations_by_features[j][3] == 'A' and 180 >= angles_of_4_atoms_of_his_mediatingH_neighbor[3] >= 150:
                                                        HIP24 += m[3]  # favorable
                                                    elif his_neighbor_configurations_by_features[j][3] in ['D/A_unknown', 'D/A/G_unknown', 'D/A_known'] and m[3][1] == 'A' and 180 >= angles_of_4_atoms_of_his_mediatingH_neighbor[3][1][2] >= 150:
                                                        HIP24 += m[3][2]  # favorable
                                                else:
                                                    HIP24 = 'N/A'
                                            else:
                                                HIP24 = 'N/A'
                                            #######################################
                                            ############################################################################

                                            ############################################################################
                                            # we will include HIP13 and HIP24 energies only if the conditions above were fulfilled
                                            if HIP13 == 'N/A' and HIP24 == 'N/A':
                                                his_possible_rotamers_energies = {'HIE13': HIE13, 'HID13': HID13, 'HIE24': HIE24, 'HID24': HID24}
                                            elif HIP13 != 'N/A' and HIP24 != 'N/A':
                                                his_possible_rotamers_energies = {'HIE13': HIE13, 'HID13': HID13, 'HIE24': HIE24, 'HID24': HID24, 'HIP13': HIP13, 'HIP24': HIP24}
                                            elif HIP13 != 'N/A':
                                                his_possible_rotamers_energies = {'HIE13': HIE13, 'HID13': HID13, 'HIE24': HIE24, 'HID24': HID24, 'HIP13': HIP13}
                                            elif HIP24 != 'N/A':
                                                his_possible_rotamers_energies = {'HIE13': HIE13, 'HID13': HID13, 'HIE24': HIE24, 'HID24': HID24, 'HIP24': HIP24}

                                            his_rotamers_ordered_in_energy = sorted(his_possible_rotamers_energies.items(), key=lambda x: x[1])
                                            his_rotamers_ordered_in_energy = [list(ele) for ele in his_rotamers_ordered_in_energy]

                                            ############################################################################
                                            # getting rid of bad scenarios where an unknown atom such as an ASN/GLN/HIS (it can act as a D, or A (or even G, for HIS))
                                            unk_neighbors_in_config = np.array([[key[0], key[1]] for idx, key in enumerate(m) if type(key) == list])
                                            unk_neighbors_wholeres_dict = defaultdict(dict)

                                            bad_scenario_count = 0

                                            # I cant have a HIS ND atom be D while CE (which is supposed to be Grease) be A to another side of the residue. I have to have consistency in the assignment possibility
                                            for k_ndx, k in enumerate(unk_neighbors_in_config):
                                                if int(k[0]) not in unk_neighbors_wholeres_dict:
                                                    this_current_unk_resname = branch_name.topology.atom(int(k[0])).residue.name
                                                    this_current_unk_resindx = branch_name.topology.atom(int(k[0])).residue.index
                                                    if this_current_unk_resname in ['HIS', 'HIE', 'HID', 'HIP']:
                                                        this_current_unk_ND1 = branch_name.topology.select("resid {} and name ND1".format(this_current_unk_resindx))[0]
                                                        this_current_unk_CE1 = branch_name.topology.select("resid {} and name CE1".format(this_current_unk_resindx))[0]
                                                        this_current_unk_NE2 = branch_name.topology.select("resid {} and name NE2".format(this_current_unk_resindx))[0]
                                                        this_current_unk_CD2 = branch_name.topology.select("resid {} and name CD2".format(this_current_unk_resindx))[0]
                                                        if int(k[0]) == this_current_unk_ND1 or int(k[0]) == this_current_unk_NE2:
                                                            if k[1] in ['D', 'A']:
                                                                unk_neighbors_wholeres_dict[this_current_unk_ND1] = ['D', 'A']
                                                                unk_neighbors_wholeres_dict[this_current_unk_NE2] = ['D', 'A']
                                                                unk_neighbors_wholeres_dict[this_current_unk_CE1] = ['G']
                                                                unk_neighbors_wholeres_dict[this_current_unk_CD2] = ['G']
                                                            elif k[1] == 'G':
                                                                unk_neighbors_wholeres_dict[this_current_unk_ND1] = ['G']
                                                                unk_neighbors_wholeres_dict[this_current_unk_NE2] = ['G']
                                                                unk_neighbors_wholeres_dict[this_current_unk_CE1] = ['D', 'A']
                                                                unk_neighbors_wholeres_dict[this_current_unk_CD2] = ['D', 'A']
                                                        elif int(k[0]) == this_current_unk_CE1 or int(k[0]) == this_current_unk_CD2:
                                                            if k[1] in ['D', 'A']:
                                                                unk_neighbors_wholeres_dict[this_current_unk_CE1] = ['D', 'A']
                                                                unk_neighbors_wholeres_dict[this_current_unk_CD2] = ['D', 'A']
                                                                unk_neighbors_wholeres_dict[this_current_unk_ND1] = ['G']
                                                                unk_neighbors_wholeres_dict[this_current_unk_NE2] = ['G']
                                                            elif k[1] == 'G':
                                                                unk_neighbors_wholeres_dict[this_current_unk_CE1] = ['G']
                                                                unk_neighbors_wholeres_dict[this_current_unk_CD2] = ['G']
                                                                unk_neighbors_wholeres_dict[this_current_unk_ND1] = ['D', 'A']
                                                                unk_neighbors_wholeres_dict[this_current_unk_NE2] = ['D', 'A']

                                                    elif this_current_unk_resname in ['ASN', 'GLN']:
                                                        this_current_unk_o_side = branch_name.topology.select('((resname ASN and name OD1) or (resname GLN and name OE1)) and resid {}'.format(this_current_unk_resindx))[0]
                                                        this_current_unk_n_side = branch_name.topology.select('((resname ASN and name ND2) or (resname GLN and name NE2)) and resid {}'.format(this_current_unk_resindx))[0]
                                                        if int(k[0]) == this_current_unk_o_side:
                                                            if k[1] == 'D':
                                                                unk_neighbors_wholeres_dict[this_current_unk_o_side] = ['D']
                                                                unk_neighbors_wholeres_dict[this_current_unk_n_side] = ['A']
                                                            elif k[1] == 'A':
                                                                unk_neighbors_wholeres_dict[this_current_unk_o_side] = ['A']
                                                                unk_neighbors_wholeres_dict[this_current_unk_n_side] = ['D']
                                                        elif int(k[0]) == this_current_unk_n_side:
                                                            if k[1] == 'D':
                                                                unk_neighbors_wholeres_dict[this_current_unk_n_side] = ['D']
                                                                unk_neighbors_wholeres_dict[this_current_unk_o_side] = ['A']
                                                            elif k[1] == 'A':
                                                                unk_neighbors_wholeres_dict[this_current_unk_n_side] = ['A']
                                                                unk_neighbors_wholeres_dict[this_current_unk_o_side] = ['D']

                                                elif int(k[0]) in unk_neighbors_wholeres_dict:
                                                    if k[1] not in unk_neighbors_wholeres_dict[int(k[0])]:
                                                        bad_scenario_count += 1

                                            # An unknown atom can not be a feature to one atom of the looped residue and another feature with another atom of the looped residue
                                            # if this atom is neighbored with the O and N atoms of another ASN, I cant have a scenario where the neighbor is D to O and A to N. that atom will have to have the same feature.
                                            for k_ndx, k in enumerate(unk_neighbors_in_config):
                                                for l_ndx, l in enumerate(unk_neighbors_in_config):
                                                    if k_ndx != l_ndx:
                                                        if k[0] == l[0] and k[1] != l[1]:
                                                            bad_scenario_count += 1
                                            ############################################################################

                                            if bad_scenario_count == 0:
                                                print('Scenario {}:'.format(index_of_m))
                                                print('     Individual energies of 4 atoms of HIS ring of this arrangement : {}'.format(m))
                                                print('     HIS possible rotamers and their energies are {}.'.format(his_rotamers_ordered_in_energy))

                                                difference_in_energy_between_lowest_2_possible_rotamers = round(abs(his_rotamers_ordered_in_energy[0][1] - his_rotamers_ordered_in_energy[1][1]), 2)
                                                if difference_in_energy_between_lowest_2_possible_rotamers >= degenerate_states_e_cutoff:
                                                    all_possibilities_lowest_rotamers[index_of_m] = his_rotamers_ordered_in_energy[0]
                                                else:
                                                    all_possibilities_lowest_rotamers[index_of_m] = [his_rotamers_ordered_in_energy[0], his_rotamers_ordered_in_energy[1]]
                                            else:
                                                print('Scenario {}:'.format(index_of_m))
                                                print('     Individual energies of 4 atoms of HIS ring of this arrangement : {}'.format(m))
                                                print('     This neighbor configuration is NOT a good scenario due to having conflicting unknown atom(s) features.')
                                                print('     It will not be considered as a scenario')

                                        ############################################################
                                        # the point of this block is to extract the BEST scenario and its features so we can use it after finding the BEST winning configuration
                                        # the point of this his_config_scenarios_energies_edited dictionary is to basically make the dictionary symmetric.
                                        # meaning, if a configuration was to have more than one rotamer energy, I will just keep one
                                        his_config_scenarios_energies_edited = defaultdict(dict)
                                        for configuration in list(all_possibilities_lowest_rotamers.items()):
                                            if len(flatten(configuration[1])) == 2:
                                                his_config_scenarios_energies_edited[configuration[0]] = configuration[1]
                                            else:
                                                his_config_scenarios_energies_edited[configuration[0]] = configuration[1][0]

                                        his_config_scenarios_energies = sorted(list(his_config_scenarios_energies_edited.items()), key=lambda x: x[1][1])
                                        his_configuration_winning_scenario_index = his_config_scenarios_energies[0][0]
                                        his_configuration_winning_scenario_features_dict[j] = energies_of_4_atoms_of_his_mediatingH_neighbor[his_configuration_winning_scenario_index]
                                        ############################################################

                                        # Here I took each scenario's lowest rotamer (or rotamers if there are two) and put in one list
                                        all_scenarios_lowest_assignment = []
                                        for ele in list(all_possibilities_lowest_rotamers.items()):
                                            if len(flatten(ele[1])) == 2:
                                                all_scenarios_lowest_assignment.append(ele[1])
                                            else:
                                                for each_ele in ele[1]:
                                                    all_scenarios_lowest_assignment.append(each_ele)

                                        all_scenarios_lowest_assignment = sorted(all_scenarios_lowest_assignment, key=lambda x: x[1])
                                        print("\nList of scenario possibilities in this configuration is:")
                                        print(all_scenarios_lowest_assignment)

                                        lowest_energy_among_all_scenarios = min([ele[1] for ele in all_scenarios_lowest_assignment])

                                        # Here, I extract the scenario numbers for which its energies are within the degenerate states E cutoff from the lowest
                                        scenario_assignments_within_1kcalmol_from_lowest = []
                                        for scenario_assignment_energy in all_scenarios_lowest_assignment:
                                            if round(abs(scenario_assignment_energy[1] - lowest_energy_among_all_scenarios), 2) < degenerate_states_e_cutoff:
                                                scenario_assignments_within_1kcalmol_from_lowest.append(scenario_assignment_energy)

                                        # Here, I will only extract unique rotamer assignments
                                        unique_rotamers_in_this_configuration = []
                                        for scenario_assignment_energy in scenario_assignments_within_1kcalmol_from_lowest:
                                            if scenario_assignment_energy[0] not in flatten(unique_rotamers_in_this_configuration):
                                                unique_rotamers_in_this_configuration.append(scenario_assignment_energy)

                                        # # if there is only one unique assignment, remove the outer brackets
                                        # if len(flatten(unique_rotamers_in_this_configuration)) == 2:
                                        #     unique_rotamers_in_this_configuration = unique_rotamers_in_this_configuration[0]

                                        print("HIS assignments that are within the degenerate states E cutoff in the scenarios of this configuration ({}):\n{}".format(j, unique_rotamers_in_this_configuration))
                                        his_configuration_decisions_with_energies[j] = unique_rotamers_in_this_configuration

                                        print('########################')

                                print('\n########################')

                            ########################################################################
                            ########################################################################

                            print('List of all HIS configurations most stable rotamers and their lowest energies:')
                            for configuration in list(his_configuration_decisions_with_energies.items()):
                                print('     ' + str(configuration))
                            print('########################\n')

                            if len(his_neighbor_configurations_by_atoms) == 1:  # Only one neighbor configuration to this HIS residue
                                print('There is only one his neighbor configuration')

                                if (his_neighbor_configurations_by_features[j][0] == 'E' or his_neighbor_configurations_by_features[j][0] == 'G') and (his_neighbor_configurations_by_features[j][1] == 'E' or his_neighbor_configurations_by_features[j][1] == 'G') and (his_neighbor_configurations_by_features[j][2] == 'E' or his_neighbor_configurations_by_features[j][2] == 'G') and (his_neighbor_configurations_by_features[j][3] == 'E' or his_neighbor_configurations_by_features[j][3] == 'G'):
                                    his_rotamer_winner = 'HIE13'
                                    print('Since this HIS residue\'s 4 ring atoms neighbored with either grease or nothing, we will identify this residue as HIE13')
                                else:
                                    his_rotamers_ordered_in_energy = list(his_configuration_decisions_with_energies.items())[0][1]

                                    if len(flatten(his_rotamers_ordered_in_energy)) == 2:
                                        his_rotamer_winner = his_rotamers_ordered_in_energy[0][0]
                                    else:
                                        difference_in_energy_between_lowest_2_possible_rotamers = round(abs(his_rotamers_ordered_in_energy[0][1] - his_rotamers_ordered_in_energy[1][1]), 2)
                                        print('The difference in energy between the lowest two rotamers of this ONLY configuration is {}'.format(difference_in_energy_between_lowest_2_possible_rotamers))

                                        if difference_in_energy_between_lowest_2_possible_rotamers >= degenerate_states_e_cutoff:
                                            print('Since the difference in energy between the lowest two rotamers energetically is equal or above the degenerate states E cutoff.')
                                            his_rotamer_winner = his_rotamers_ordered_in_energy[0][0]
                                            print('     Keeping only the lowest rotamer energetically of this ONLY configuration of this HIS residue: {}'.format(his_rotamer_winner))

                                        else:
                                            print('Since the difference in energy between the lowest two rotamers energetically is within the degenerate states E cutoff, we will not identify this residue or make it known unless branching is necessary later on in the analysis!!')
                                            his_rotamer_winner = [his_rotamers_ordered_in_energy[0][0], his_rotamers_ordered_in_energy[1][0]]

                                            branch_name.his_with_splits[branch_name.topology.atom(his_ND[i]).residue.index] = his_rotamer_winner

                            else:  # When there is more than one neighbor configuration to this HIS residue
                                print('There are more than one his neighbor configuration:\n')

                                his_rotamers_ordered_in_energy = []
                                for configuration in list(his_configuration_decisions_with_energies.items()):
                                    for assignment in configuration[1]:
                                        his_rotamers_ordered_in_energy.append(assignment)

                                # Here, I extract the configs indices for which its energies are within the degenerate states E cutoff from the lowest
                                lowest_energy_among_all_configs = min([ele[1] for ele in his_rotamers_ordered_in_energy])
                                all_low_configs_assignments = []
                                for assignment_energy in his_rotamers_ordered_in_energy:
                                    if round(abs(assignment_energy[1] - lowest_energy_among_all_configs), 2) < degenerate_states_e_cutoff:
                                        all_low_configs_assignments.append(assignment_energy[0])

                                his_rotamer_winner = [list(np.unique(np.array(flatten(all_low_configs_assignments))))]

                                print("\nThese rotamer assignments are seen in the previous configs:")
                                print(his_rotamer_winner)

                                # if there is only one unique assignment, we can just set the residue to this assignment and we will consider the winning_config as the one lowest in E, that would be configs_indices_within_1kcalmol_from_lowest[0]
                                if len(flatten(his_rotamer_winner)) == 1:
                                    his_rotamer_winner = his_rotamer_winner[0][0]

                                    ############################################################
                                    # the point of this his_configurations_energies_edited dictionary is to basically make the dictionary symmetric.
                                    # meaning, if a configuration was to have more than one rotamer energy, I will just keep one so in the next step of the code I can compare one rotamer energy of each configuration without making errors in sorted() usage
                                    his_configurations_energies_edited = defaultdict(dict)
                                    for configuration in list(his_configuration_decisions_with_energies.items()):
                                        his_configurations_energies_edited[configuration[0]] = configuration[1][0]

                                    his_configs_rotamers_ordered_in_energy_edited = sorted(list(his_configurations_energies_edited.items()), key=lambda x: x[1][1])
                                    ############################################################
                                    winning_configuration_index = his_configs_rotamers_ordered_in_energy_edited[0][0]

                                else:
                                    print('his_splits of this residue is {}'.format(his_rotamer_winner))

                                    branch_name.his_with_splits[branch_name.topology.atom(his_ND[i]).residue.index] = his_rotamer_winner[0]

                            if isinstance(his_rotamer_winner, list) is False:

                                ####################################################
                                # This is very important because if I only have one configuration, the winning configuration index will always be "0". This is needed so we can set the HIS as HIE, HID, HIP (13/24).
                                if len(his_neighbor_configurations_by_atoms) == 1:
                                    winning_configuration_index = 0
                                ####################################################

                                ####################################################
                                # Now, that we know the winning_configuration_index, we can fetch the don neighbor H id which was used so we can append it to the branch_name.reduced_topology_not_available_donors list
                                for don_neighbor_id in his_configuration_donorneighbors_used[winning_configuration_index]:
                                    branch_name.reduced_topology_not_available_donors.append(don_neighbor_id)
                                    print('Atom {} just got added to reduced_topology_not_available_donors list. This ensures that this H can\'t be used later.'.format(don_neighbor_id))
                                ####################################################

                                if his_rotamer_winner[0:3] == 'HIE':
                                    print('Residue name before conversion is {}'.format(branch_name.topology.atom(his_ND[i]).residue))
                                    branch_name.topology.atom(his_ND[i]).residue.name = 'HIE'
                                    print('Residue name after conversion is {}'.format(branch_name.topology.atom(his_ND[i]).residue))

                                    if his_rotamer_winner[3:5] == '13':
                                        print('No atom flipping required!')

                                        if his_neighbor_configurations_by_features[winning_configuration_index][0] == 'D':
                                            branch_name.known_don = np.delete(branch_name.known_don, np.where(branch_name.known_don == his_neighbor_configurations_by_atoms[winning_configuration_index][0])[0][0])
                                            print('Atom {} just got deleted from known_don list ONCE because it was just used up in an H-bond interaction.'.format(his_neighbor_configurations_by_atoms[winning_configuration_index][0]))

                                            the_donor_neighbor = his_neighbor_configurations_by_atoms[winning_configuration_index][0]
                                            if branch_name.topology.atom(the_donor_neighbor).name in ['OH', 'OG', 'OG1'] and branch_name.topology.atom(the_donor_neighbor).residue.name in ['TYR', 'SER', 'THR']:
                                                print('       The donor atom {} ({}) which is used in this interaction happens to be SER/THR/TYR residue'.format(the_donor_neighbor, branch_name.topology.atom(the_donor_neighbor)))
                                                print('       This donor along with the acceptor will be added to donor_of_oxi_SER_THR_TYR_acc_neighbor_set list so the H can be printed in the output structure')
                                                branch_name.donor_of_oxi_SER_THR_TYR_acc_neighbor_set.append([[the_donor_neighbor, branch_name.topology.atom(the_donor_neighbor), branch_name.xyz[0][the_donor_neighbor]], [his_ND[i], branch_name.topology.atom(his_ND[i]), branch_name.xyz[0][his_ND[i]]]])

                                        elif his_neighbor_configurations_by_features[winning_configuration_index][0] == 'E' or his_neighbor_configurations_by_features[winning_configuration_index][0] == 'G' or his_neighbor_configurations_by_features[winning_configuration_index][0] == 'A':
                                            branch_name.known_acc = np.append(branch_name.known_acc, his_ND[i])
                                            print('Atom {} just got added to known_acc list ONCE. No complementary H-bond interaction.'.format(his_ND[i]))
                                        elif his_neighbor_configurations_by_features[winning_configuration_index][0] == 'D/A_known':
                                            if his_configuration_winning_scenario_features_dict[winning_configuration_index][0][1] == 'A':
                                                branch_name.known_acc = np.append(branch_name.known_acc, his_ND[i])
                                                print('Atom {} just got added to known_acc list ONCE. No complementary H-bond interaction.'.format(his_ND[i]))
                                            elif his_configuration_winning_scenario_features_dict[winning_configuration_index][0][1] == 'D':
                                                branch_name.known_don = np.delete(branch_name.known_don, np.where(branch_name.known_don == his_neighbor_configurations_by_atoms[winning_configuration_index][0])[0][0])
                                                print('Atom {} just got deleted from known_don list ONCE because it was just used up in an H-bond interaction.'.format(his_neighbor_configurations_by_atoms[winning_configuration_index][0]))
                                        else:
                                            branch_name.known_acc = np.append(branch_name.known_acc, his_ND[i])
                                            print('Atom {} just got added to known_acc list ONCE. No complementary H-bond interaction.'.format(his_ND[i]))

                                        if his_neighbor_configurations_by_features[winning_configuration_index][2] == 'A':
                                            branch_name.known_acc = np.delete(branch_name.known_acc, np.where(branch_name.known_acc == his_neighbor_configurations_by_atoms[winning_configuration_index][2])[0][0])
                                            print('Atom {} just got deleted from known_acc list ONCE because it was just used up in an H-bond interaction.'.format(his_neighbor_configurations_by_atoms[winning_configuration_index][2]))
                                        elif his_neighbor_configurations_by_features[winning_configuration_index][2] == 'E' or his_neighbor_configurations_by_features[winning_configuration_index][2] == 'G' or his_neighbor_configurations_by_features[winning_configuration_index][2] == 'D':
                                            branch_name.known_don = np.append(branch_name.known_don, his_NE[i])
                                            print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(his_NE[i]))
                                        elif his_neighbor_configurations_by_features[winning_configuration_index][2] == 'D/A_known':
                                            if his_configuration_winning_scenario_features_dict[winning_configuration_index][2][1] == 'A':
                                                branch_name.known_acc = np.delete(branch_name.known_acc, np.where(branch_name.known_acc == his_neighbor_configurations_by_atoms[winning_configuration_index][2])[0][0])
                                                print('Atom {} just got deleted from known_acc list ONCE because it was just used up in an H-bond interaction.'.format(his_neighbor_configurations_by_atoms[winning_configuration_index][2]))
                                            elif his_configuration_winning_scenario_features_dict[winning_configuration_index][2][1] == 'D':
                                                branch_name.known_don = np.append(branch_name.known_don, his_NE[i])
                                                print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(his_NE[i]))
                                        else:
                                            branch_name.known_don = np.append(branch_name.known_don, his_NE[i])
                                            print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(his_NE[i]))

                                        branch_name.allpos_prot_don_acc = np.delete(branch_name.allpos_prot_don_acc, np.where(branch_name.allpos_prot_don_acc == his_CE[i])[0][0])
                                        branch_name.allpos_prot_don_acc = np.delete(branch_name.allpos_prot_don_acc, np.where(branch_name.allpos_prot_don_acc == his_CD[i])[0][0])
                                        branch_name.grease = np.append(branch_name.grease, his_CE[i])
                                        branch_name.grease = np.append(branch_name.grease, his_CD[i])

                                    elif his_rotamer_winner[3:5] == '24':
                                        print('Atoms were:  {} {}, {} {}, {} {}, {} {}'.format(his_ND[i], branch_name.topology.atom(his_ND[i]).name, his_CE[i], branch_name.topology.atom(his_CE[i]).name, his_NE[i], branch_name.topology.atom(his_NE[i]).name, his_CD[i], branch_name.topology.atom(his_CD[i]).name))
                                        print('Atom names were just flipped')
                                        # We switch atom name and atom element from C to N and from N to C
                                        temp1 = branch_name.topology.atom(his_ND[i]).name
                                        temp2 = branch_name.topology.atom(his_CE[i]).name
                                        temp3 = branch_name.topology.atom(his_NE[i]).name
                                        temp4 = branch_name.topology.atom(his_CD[i]).name
                                        branch_name.topology.atom(his_ND[i]).name = temp4
                                        branch_name.topology.atom(his_CE[i]).name = temp3
                                        branch_name.topology.atom(his_NE[i]).name = temp2
                                        branch_name.topology.atom(his_CD[i]).name = temp1

                                        temp1_element = branch_name.topology.atom(his_ND[i]).element
                                        temp2_element = branch_name.topology.atom(his_CE[i]).element
                                        temp3_element = branch_name.topology.atom(his_NE[i]).element
                                        temp4_element = branch_name.topology.atom(his_CD[i]).element
                                        branch_name.topology.atom(his_ND[i]).element = temp4_element
                                        branch_name.topology.atom(his_CE[i]).element = temp3_element
                                        branch_name.topology.atom(his_NE[i]).element = temp2_element
                                        branch_name.topology.atom(his_CD[i]).element = temp1_element

                                        # We append those atoms that switch from C to N and from N to C
                                        branch_name.atm_name_change_to_N.append(his_CE[i])
                                        branch_name.atm_name_change_to_N.append(his_CD[i])
                                        branch_name.atm_name_change_to_C.append(his_ND[i])
                                        branch_name.atm_name_change_to_C.append(his_NE[i])

                                        # We remove the two atoms that were N from allpos_prot_don_acc list and we add to grease list
                                        branch_name.allpos_prot_don_acc = np.delete(branch_name.allpos_prot_don_acc, np.where(branch_name.allpos_prot_don_acc == his_ND[i])[0][0])
                                        branch_name.allpos_prot_don_acc = np.delete(branch_name.allpos_prot_don_acc, np.where(branch_name.allpos_prot_don_acc == his_NE[i])[0][0])
                                        branch_name.grease = np.append(branch_name.grease, his_ND[i])
                                        branch_name.grease = np.append(branch_name.grease, his_NE[i])

                                        print('Now, atoms are:  {} {}, {} {}, {} {}, {} {}'.format(his_ND[i], branch_name.topology.atom(his_ND[i]).name, his_CE[i], branch_name.topology.atom(his_CE[i]).name, his_NE[i], branch_name.topology.atom(his_NE[i]).name, his_CD[i], branch_name.topology.atom(his_CD[i]).name))
                                        if his_neighbor_configurations_by_features[winning_configuration_index][3] == 'D':
                                            branch_name.known_don = np.delete(branch_name.known_don, np.where(branch_name.known_don == his_neighbor_configurations_by_atoms[winning_configuration_index][3])[0][0])
                                            print('Atom {} just got deleted from known_don list ONCE because it was just used up in an H-bond interaction.'.format(his_neighbor_configurations_by_atoms[winning_configuration_index][3]))

                                            the_donor_neighbor = his_neighbor_configurations_by_atoms[winning_configuration_index][3]
                                            if branch_name.topology.atom(the_donor_neighbor).name in ['OH', 'OG', 'OG1'] and branch_name.topology.atom(the_donor_neighbor).residue.name in ['TYR', 'SER', 'THR']:
                                                print('       The donor atom {} ({}) which is used in this interaction happens to be SER/THR/TYR residue'.format(the_donor_neighbor, branch_name.topology.atom(the_donor_neighbor)))
                                                print('       This donor along with the acceptor will be added to donor_of_oxi_SER_THR_TYR_acc_neighbor_set list so the H can be printed in the output structure')
                                                branch_name.donor_of_oxi_SER_THR_TYR_acc_neighbor_set.append([[the_donor_neighbor, branch_name.topology.atom(the_donor_neighbor), branch_name.xyz[0][the_donor_neighbor]], [his_CD[i], branch_name.topology.atom(his_CD[i]), branch_name.xyz[0][his_CD[i]]]])

                                        elif his_neighbor_configurations_by_features[winning_configuration_index][3] == 'E' or his_neighbor_configurations_by_features[winning_configuration_index][3] == 'G' or his_neighbor_configurations_by_features[winning_configuration_index][3] == 'A':
                                            branch_name.known_acc = np.append(branch_name.known_acc, his_CD[i])
                                            print('Atom {} just got added to known_acc list ONCE. No complementary H-bond interaction.'.format(his_CD[i]))
                                        elif his_neighbor_configurations_by_features[winning_configuration_index][3] == 'D/A_known':
                                            if his_configuration_winning_scenario_features_dict[winning_configuration_index][3][1] == 'A':
                                                branch_name.known_acc = np.append(branch_name.known_acc, his_CD[i])
                                                print('Atom {} just got added to known_acc list ONCE. No complementary H-bond interaction.'.format(his_CD[i]))
                                            elif his_configuration_winning_scenario_features_dict[winning_configuration_index][3][1] == 'D':
                                                branch_name.known_don = np.delete(branch_name.known_don, np.where(branch_name.known_don == his_neighbor_configurations_by_atoms[winning_configuration_index][3])[0][0])
                                                print('Atom {} just got deleted from known_don list ONCE because it was just used up in an H-bond interaction.'.format(his_neighbor_configurations_by_atoms[winning_configuration_index][3]))
                                        else:
                                            branch_name.known_acc = np.append(branch_name.known_acc, his_CD[i])
                                            print('Atom {} just got added to known_acc list ONCE. No complementary H-bond interaction.'.format(his_CD[i]))

                                        if his_neighbor_configurations_by_features[winning_configuration_index][1] == 'A':
                                            branch_name.known_acc = np.delete(branch_name.known_acc, np.where(branch_name.known_acc == his_neighbor_configurations_by_atoms[winning_configuration_index][1])[0][0])
                                            print('Atom {} just got deleted from known_acc list ONCE because it was just used up in an H-bond interaction.'.format(his_neighbor_configurations_by_atoms[winning_configuration_index][1]))
                                        elif his_neighbor_configurations_by_features[winning_configuration_index][1] == 'E' or his_neighbor_configurations_by_features[winning_configuration_index][1] == 'G' or his_neighbor_configurations_by_features[winning_configuration_index][1] == 'D':
                                            branch_name.known_don = np.append(branch_name.known_don, his_CE[i])
                                            print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(his_CE[i]))
                                        elif his_neighbor_configurations_by_features[winning_configuration_index][1] == 'D/A_known':
                                            if his_configuration_winning_scenario_features_dict[winning_configuration_index][1][1] == 'A':
                                                branch_name.known_acc = np.delete(branch_name.known_acc, np.where(branch_name.known_acc == his_neighbor_configurations_by_atoms[winning_configuration_index][1])[0][0])
                                                print('Atom {} just got deleted from known_acc list ONCE because it was just used up in an H-bond interaction.'.format(his_neighbor_configurations_by_atoms[winning_configuration_index][1]))
                                            elif his_configuration_winning_scenario_features_dict[winning_configuration_index][1][1] == 'D':
                                                branch_name.known_don = np.append(branch_name.known_don, his_CE[i])
                                                print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(his_CE[i]))
                                        else:
                                            branch_name.known_don = np.append(branch_name.known_don, his_CE[i])
                                            print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(his_CE[i]))

                                elif his_rotamer_winner[0:3] == 'HID':
                                    print('Residue name before conversion is {}'.format(branch_name.topology.atom(his_ND[i]).residue))
                                    branch_name.topology.atom(his_ND[i]).residue.name = 'HID'
                                    print('Residue name after conversion is {}'.format(branch_name.topology.atom(his_ND[i]).residue))

                                    if his_rotamer_winner[3:5] == '13':
                                        print('No atom flipping required!')

                                        if his_neighbor_configurations_by_features[winning_configuration_index][0] == 'A':
                                            branch_name.known_acc = np.delete(branch_name.known_acc, np.where(branch_name.known_acc == his_neighbor_configurations_by_atoms[winning_configuration_index][0])[0][0])
                                            print('Atom {} just got deleted from known_acc list ONCE because it was just used up in an H-bond interaction.'.format(his_neighbor_configurations_by_atoms[winning_configuration_index][0]))
                                        elif his_neighbor_configurations_by_features[winning_configuration_index][0] == 'E' or his_neighbor_configurations_by_features[winning_configuration_index][0] == 'G' or his_neighbor_configurations_by_features[winning_configuration_index][0] == 'D':
                                            branch_name.known_don = np.append(branch_name.known_don, his_ND[i])
                                            print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(his_ND[i]))
                                        elif his_neighbor_configurations_by_features[winning_configuration_index][0] == 'D/A_known':
                                            if his_configuration_winning_scenario_features_dict[winning_configuration_index][0][1] == 'A':
                                                branch_name.known_acc = np.delete(branch_name.known_acc, np.where(branch_name.known_acc == his_neighbor_configurations_by_atoms[winning_configuration_index][0])[0][0])
                                                print('Atom {} just got deleted from known_acc list ONCE because it was just used up in an H-bond interaction.'.format(his_neighbor_configurations_by_atoms[winning_configuration_index][0]))
                                            elif his_configuration_winning_scenario_features_dict[winning_configuration_index][0][1] == 'D':
                                                branch_name.known_don = np.append(branch_name.known_don, his_ND[i])
                                                print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(his_ND[i]))
                                        else:
                                            branch_name.known_don = np.append(branch_name.known_don, his_ND[i])
                                            print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(his_ND[i]))

                                        if his_neighbor_configurations_by_features[winning_configuration_index][2] == 'D':
                                            branch_name.known_don = np.delete(branch_name.known_don, np.where(branch_name.known_don == his_neighbor_configurations_by_atoms[winning_configuration_index][2])[0][0])
                                            print('Atom {} just got deleted from known_don list ONCE because it was just used up in an H-bond interaction.'.format(his_neighbor_configurations_by_atoms[winning_configuration_index][2]))

                                            the_donor_neighbor = his_neighbor_configurations_by_atoms[winning_configuration_index][2]
                                            if branch_name.topology.atom(the_donor_neighbor).name in ['OH', 'OG', 'OG1'] and branch_name.topology.atom(the_donor_neighbor).residue.name in ['TYR', 'SER', 'THR']:
                                                print('       The donor atom {} ({}) which is used in this interaction happens to be SER/THR/TYR residue'.format(the_donor_neighbor, branch_name.topology.atom(the_donor_neighbor)))
                                                print('       This donor along with the acceptor will be added to donor_of_oxi_SER_THR_TYR_acc_neighbor_set list so the H can be printed in the output structure')
                                                branch_name.donor_of_oxi_SER_THR_TYR_acc_neighbor_set.append([[the_donor_neighbor, branch_name.topology.atom(the_donor_neighbor), branch_name.xyz[0][the_donor_neighbor]], [his_NE[i], branch_name.topology.atom(his_NE[i]), branch_name.xyz[0][his_NE[i]]]])

                                        elif his_neighbor_configurations_by_features[winning_configuration_index][2] == 'E' or his_neighbor_configurations_by_features[winning_configuration_index][2] == 'G' or his_neighbor_configurations_by_features[winning_configuration_index][2] == 'A':
                                            branch_name.known_acc = np.append(branch_name.known_acc, his_NE[i])
                                            print('Atom {} just got added to known_acc list ONCE. No complementary H-bond interaction.'.format(his_NE[i]))
                                        elif his_neighbor_configurations_by_features[winning_configuration_index][2] == 'D/A_known':
                                            if his_configuration_winning_scenario_features_dict[winning_configuration_index][2][1] == 'A':
                                                branch_name.known_acc = np.append(branch_name.known_acc, his_NE[i])
                                                print('Atom {} just got added to known_acc list ONCE. No complementary H-bond interaction.'.format(his_NE[i]))
                                            elif his_configuration_winning_scenario_features_dict[winning_configuration_index][2][1] == 'D':
                                                branch_name.known_don = np.delete(branch_name.known_don, np.where(branch_name.known_don == his_neighbor_configurations_by_atoms[winning_configuration_index][2])[0][0])
                                                print('Atom {} just got deleted from known_don list ONCE because it was just used up in an H-bond interaction.'.format(his_neighbor_configurations_by_atoms[winning_configuration_index][2]))
                                        else:
                                            branch_name.known_acc = np.append(branch_name.known_acc, his_NE[i])
                                            print('Atom {} just got added to known_acc list ONCE. No complementary H-bond interaction.'.format(his_NE[i]))

                                        branch_name.allpos_prot_don_acc = np.delete(branch_name.allpos_prot_don_acc, np.where(branch_name.allpos_prot_don_acc == his_CE[i])[0][0])
                                        branch_name.allpos_prot_don_acc = np.delete(branch_name.allpos_prot_don_acc, np.where(branch_name.allpos_prot_don_acc == his_CD[i])[0][0])
                                        branch_name.grease = np.append(branch_name.grease, his_CE[i])
                                        branch_name.grease = np.append(branch_name.grease, his_CD[i])

                                    elif his_rotamer_winner[3:5] == '24':
                                        print('Atoms were:  {} {}, {} {}, {} {}, {} {}'.format(his_ND[i], branch_name.topology.atom(his_ND[i]).name, his_CE[i], branch_name.topology.atom(his_CE[i]).name, his_NE[i], branch_name.topology.atom(his_NE[i]).name, his_CD[i], branch_name.topology.atom(his_CD[i]).name))
                                        print('Atom names were just flipped')
                                        # We switch atom name and atom element from C to N and from N to C
                                        temp1 = branch_name.topology.atom(his_ND[i]).name
                                        temp2 = branch_name.topology.atom(his_CE[i]).name
                                        temp3 = branch_name.topology.atom(his_NE[i]).name
                                        temp4 = branch_name.topology.atom(his_CD[i]).name
                                        branch_name.topology.atom(his_ND[i]).name = temp4
                                        branch_name.topology.atom(his_CE[i]).name = temp3
                                        branch_name.topology.atom(his_NE[i]).name = temp2
                                        branch_name.topology.atom(his_CD[i]).name = temp1

                                        temp1_element = branch_name.topology.atom(his_ND[i]).element
                                        temp2_element = branch_name.topology.atom(his_CE[i]).element
                                        temp3_element = branch_name.topology.atom(his_NE[i]).element
                                        temp4_element = branch_name.topology.atom(his_CD[i]).element
                                        branch_name.topology.atom(his_ND[i]).element = temp4_element
                                        branch_name.topology.atom(his_CE[i]).element = temp3_element
                                        branch_name.topology.atom(his_NE[i]).element = temp2_element
                                        branch_name.topology.atom(his_CD[i]).element = temp1_element

                                        # We append those atoms that switch from C to N and from N to C
                                        branch_name.atm_name_change_to_N.append(his_CE[i])
                                        branch_name.atm_name_change_to_N.append(his_CD[i])
                                        branch_name.atm_name_change_to_C.append(his_ND[i])
                                        branch_name.atm_name_change_to_C.append(his_NE[i])

                                        # We remove the two atoms that were N from allpos_prot_don_acc list and we add to grease list
                                        branch_name.allpos_prot_don_acc = np.delete(branch_name.allpos_prot_don_acc, np.where(branch_name.allpos_prot_don_acc == his_ND[i])[0][0])
                                        branch_name.allpos_prot_don_acc = np.delete(branch_name.allpos_prot_don_acc, np.where(branch_name.allpos_prot_don_acc == his_NE[i])[0][0])
                                        branch_name.grease = np.append(branch_name.grease, his_ND[i])
                                        branch_name.grease = np.append(branch_name.grease, his_NE[i])

                                        print('Now, atoms are:  {} {}, {} {}, {} {}, {} {}'.format(his_ND[i], branch_name.topology.atom(his_ND[i]).name, his_CE[i], branch_name.topology.atom(his_CE[i]).name, his_NE[i], branch_name.topology.atom(his_NE[i]).name, his_CD[i], branch_name.topology.atom(his_CD[i]).name))
                                        if his_neighbor_configurations_by_features[winning_configuration_index][3] == 'A':
                                            branch_name.known_acc = np.delete(branch_name.known_acc, np.where(branch_name.known_acc == his_neighbor_configurations_by_atoms[winning_configuration_index][3])[0][0])
                                            print('Atom {} just got deleted from known_acc list ONCE because it was just used up in an H-bond interaction.'.format(his_neighbor_configurations_by_atoms[winning_configuration_index][3]))
                                        elif his_neighbor_configurations_by_features[winning_configuration_index][3] == 'E' or his_neighbor_configurations_by_features[winning_configuration_index][3] == 'G' or his_neighbor_configurations_by_features[winning_configuration_index][3] == 'D':
                                            branch_name.known_don = np.append(branch_name.known_don, his_CD[i])
                                            print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(his_CD[i]))
                                        elif his_neighbor_configurations_by_features[winning_configuration_index][3] == 'D/A_known':
                                            if his_configuration_winning_scenario_features_dict[winning_configuration_index][3][1] == 'A':
                                                branch_name.known_acc = np.delete(branch_name.known_acc, np.where(branch_name.known_acc == his_neighbor_configurations_by_atoms[winning_configuration_index][3])[0][0])
                                                print('Atom {} just got deleted from known_acc list ONCE because it was just used up in an H-bond interaction.'.format(his_neighbor_configurations_by_atoms[winning_configuration_index][3]))
                                            elif his_configuration_winning_scenario_features_dict[winning_configuration_index][3][1] == 'D':
                                                branch_name.known_don = np.append(branch_name.known_don, his_CD[i])
                                                print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(his_CD[i]))
                                        else:
                                            branch_name.known_don = np.append(branch_name.known_don, his_CD[i])
                                            print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(his_CD[i]))

                                        if his_neighbor_configurations_by_features[winning_configuration_index][1] == 'D':
                                            branch_name.known_don = np.delete(branch_name.known_don, np.where(branch_name.known_don == his_neighbor_configurations_by_atoms[winning_configuration_index][1])[0][0])
                                            print('Atom {} just got deleted from known_don list ONCE because it was just used up in an H-bond interaction.'.format(his_neighbor_configurations_by_atoms[winning_configuration_index][1]))

                                            the_donor_neighbor = his_neighbor_configurations_by_atoms[winning_configuration_index][1]
                                            if branch_name.topology.atom(the_donor_neighbor).name in ['OH', 'OG', 'OG1'] and branch_name.topology.atom(the_donor_neighbor).residue.name in ['TYR', 'SER', 'THR']:
                                                print('       The donor atom {} ({}) which is used in this interaction happens to be SER/THR/TYR residue'.format(the_donor_neighbor, branch_name.topology.atom(the_donor_neighbor)))
                                                print('       This donor along with the acceptor will be added to donor_of_oxi_SER_THR_TYR_acc_neighbor_set list so the H can be printed in the output structure')
                                                branch_name.donor_of_oxi_SER_THR_TYR_acc_neighbor_set.append([[the_donor_neighbor, branch_name.topology.atom(the_donor_neighbor), branch_name.xyz[0][the_donor_neighbor]], [his_CE[i], branch_name.topology.atom(his_CE[i]), branch_name.xyz[0][his_CE[i]]]])

                                        elif his_neighbor_configurations_by_features[winning_configuration_index][1] == 'E' or his_neighbor_configurations_by_features[winning_configuration_index][1] == 'G' or his_neighbor_configurations_by_features[winning_configuration_index][1] == 'A':
                                            branch_name.known_acc = np.append(branch_name.known_acc, his_CE[i])
                                            print('Atom {} just got added to known_acc list ONCE. No complementary H-bond interaction.'.format(his_CE[i]))
                                        elif his_neighbor_configurations_by_features[winning_configuration_index][1] == 'D/A_known':
                                            if his_configuration_winning_scenario_features_dict[winning_configuration_index][1][1] == 'A':
                                                branch_name.known_acc = np.append(branch_name.known_acc, his_CE[i])
                                                print('Atom {} just got added to known_acc list ONCE. No complementary H-bond interaction.'.format(his_CE[i]))
                                            elif his_configuration_winning_scenario_features_dict[winning_configuration_index][1][1] == 'D':
                                                branch_name.known_don = np.delete(branch_name.known_don, np.where(branch_name.known_don == his_neighbor_configurations_by_atoms[winning_configuration_index][1])[0][0])
                                                print('Atom {} just got deleted from known_don list ONCE because it was just used up in an H-bond interaction.'.format(his_neighbor_configurations_by_atoms[winning_configuration_index][1]))
                                        else:
                                            branch_name.known_acc = np.append(branch_name.known_acc, his_CE[i])
                                            print('Atom {} just got added to known_acc list ONCE. No complementary H-bond interaction.'.format(his_CE[i]))

                                elif his_rotamer_winner[0:3] == 'HIP':
                                    print('Residue name before conversion is {}'.format(branch_name.topology.atom(his_ND[i]).residue))
                                    branch_name.topology.atom(his_ND[i]).residue.name = 'HIP'
                                    print('Residue name after conversion is {}'.format(branch_name.topology.atom(his_ND[i]).residue))

                                    if his_rotamer_winner[3:5] == '13':
                                        print('No atom flipping required!')

                                        if his_neighbor_configurations_by_features[winning_configuration_index][0] == 'A' or \
                                                (his_neighbor_configurations_by_features[winning_configuration_index][0] == 'D/A_known' and his_configuration_winning_scenario_features_dict[winning_configuration_index][0][1] == 'A'):
                                            branch_name.known_acc = np.delete(branch_name.known_acc, np.where(branch_name.known_acc == his_neighbor_configurations_by_atoms[winning_configuration_index][0])[0][0])
                                            print('Atom {} just got deleted from known_acc list ONCE because it was just used up in an H-bond interaction.'.format(his_neighbor_configurations_by_atoms[winning_configuration_index][0]))
                                        else:
                                            branch_name.known_don = np.append(branch_name.known_don, his_ND[i])

                                        if his_neighbor_configurations_by_features[winning_configuration_index][2] == 'A' or \
                                                (his_neighbor_configurations_by_features[winning_configuration_index][2] == 'D/A_known' and his_configuration_winning_scenario_features_dict[winning_configuration_index][2][1] == 'A'):
                                            branch_name.known_acc = np.delete(branch_name.known_acc, np.where(branch_name.known_acc == his_neighbor_configurations_by_atoms[winning_configuration_index][2])[0][0])
                                            print('Atom {} just got deleted from known_acc list ONCE because it was just used up in an H-bond interaction.'.format(his_neighbor_configurations_by_atoms[winning_configuration_index][2]))
                                        else:
                                            branch_name.known_don = np.append(branch_name.known_don, his_NE[i])

                                        branch_name.allpos_prot_don_acc = np.delete(branch_name.allpos_prot_don_acc, np.where(branch_name.allpos_prot_don_acc == his_CE[i])[0][0])
                                        branch_name.allpos_prot_don_acc = np.delete(branch_name.allpos_prot_don_acc, np.where(branch_name.allpos_prot_don_acc == his_CD[i])[0][0])
                                        branch_name.grease = np.append(branch_name.grease, his_CE[i])
                                        branch_name.grease = np.append(branch_name.grease, his_CD[i])

                                    elif his_rotamer_winner[3:5] == '24':
                                        print('Atoms were:  {} {}, {} {}, {} {}, {} {}'.format(his_ND[i], branch_name.topology.atom(his_ND[i]).name, his_CE[i], branch_name.topology.atom(his_CE[i]).name, his_NE[i], branch_name.topology.atom(his_NE[i]).name, his_CD[i], branch_name.topology.atom(his_CD[i]).name))
                                        print('Atom names were just flipped')
                                        # We switch atom name and atom element from C to N and from N to C
                                        temp1 = branch_name.topology.atom(his_ND[i]).name
                                        temp2 = branch_name.topology.atom(his_CE[i]).name
                                        temp3 = branch_name.topology.atom(his_NE[i]).name
                                        temp4 = branch_name.topology.atom(his_CD[i]).name
                                        branch_name.topology.atom(his_ND[i]).name = temp4
                                        branch_name.topology.atom(his_CE[i]).name = temp3
                                        branch_name.topology.atom(his_NE[i]).name = temp2
                                        branch_name.topology.atom(his_CD[i]).name = temp1

                                        temp1_element = branch_name.topology.atom(his_ND[i]).element
                                        temp2_element = branch_name.topology.atom(his_CE[i]).element
                                        temp3_element = branch_name.topology.atom(his_NE[i]).element
                                        temp4_element = branch_name.topology.atom(his_CD[i]).element
                                        branch_name.topology.atom(his_ND[i]).element = temp4_element
                                        branch_name.topology.atom(his_CE[i]).element = temp3_element
                                        branch_name.topology.atom(his_NE[i]).element = temp2_element
                                        branch_name.topology.atom(his_CD[i]).element = temp1_element

                                        # We append those atoms that switch from C to N and from N to C
                                        branch_name.atm_name_change_to_N.append(his_CE[i])
                                        branch_name.atm_name_change_to_N.append(his_CD[i])
                                        branch_name.atm_name_change_to_C.append(his_ND[i])
                                        branch_name.atm_name_change_to_C.append(his_NE[i])

                                        # We remove the two atoms that were N from allpos_prot_don_acc list and we add to grease list
                                        branch_name.allpos_prot_don_acc = np.delete(branch_name.allpos_prot_don_acc, np.where(branch_name.allpos_prot_don_acc == his_ND[i])[0][0])
                                        branch_name.allpos_prot_don_acc = np.delete(branch_name.allpos_prot_don_acc, np.where(branch_name.allpos_prot_don_acc == his_NE[i])[0][0])
                                        branch_name.grease = np.append(branch_name.grease, his_ND[i])
                                        branch_name.grease = np.append(branch_name.grease, his_NE[i])

                                        print('Now, atoms are:  {} {}, {} {}, {} {}, {} {}'.format(his_ND[i], branch_name.topology.atom(his_ND[i]).name, his_CE[i], branch_name.topology.atom(his_CE[i]).name, his_NE[i], branch_name.topology.atom(his_NE[i]).name, his_CD[i], branch_name.topology.atom(his_CD[i]).name))
                                        if his_neighbor_configurations_by_features[winning_configuration_index][3] == 'A' or \
                                                (his_neighbor_configurations_by_features[winning_configuration_index][3] == 'D/A_known' and his_configuration_winning_scenario_features_dict[winning_configuration_index][3][1] == 'A'):
                                            branch_name.known_acc = np.delete(branch_name.known_acc, np.where(branch_name.known_acc == his_neighbor_configurations_by_atoms[winning_configuration_index][3])[0][0])
                                            print('Atom {} just got deleted from known_acc list ONCE because it was just used up in an H-bond interaction.'.format(his_neighbor_configurations_by_atoms[winning_configuration_index][3]))
                                        else:
                                            branch_name.known_don = np.append(branch_name.known_don, his_CD[i])

                                        if his_neighbor_configurations_by_features[winning_configuration_index][1] == 'A' or \
                                                (his_neighbor_configurations_by_features[winning_configuration_index][1] == 'D/A_known' and his_configuration_winning_scenario_features_dict[winning_configuration_index][1][1] == 'A'):
                                            branch_name.known_acc = np.delete(branch_name.known_acc, np.where(branch_name.known_acc == his_neighbor_configurations_by_atoms[winning_configuration_index][1])[0][0])
                                            print('Atom {} just got deleted from known_acc list ONCE because it was just used up in an H-bond interaction.'.format(his_neighbor_configurations_by_atoms[winning_configuration_index][1]))
                                        else:
                                            branch_name.known_don = np.append(branch_name.known_don, his_CE[i])

                                # delete known residues from the unknown residues list
                                branch_name.unknown_residues = np.delete(branch_name.unknown_residues, np.where(branch_name.unknown_residues == branch_name.topology.atom(his_ND[i]).residue.index)[0][0])

                                # print('known_acc has {} atoms now'.format(len(branch_name.known_acc)))  # print('last 20 items in known_acc are {}'.format(branch_name.known_acc[-20:]))  # print('known_don has {} atoms now'.format(len(branch_name.known_don)))  # print('last 20 items in known_don are {}'.format(branch_name.known_don[-20:]))

                            print('\n############################################################')

                ########################  # END HIS  ########################

        final_number_of_unknowns_after_pass = len(branch_name.unknown_residues)

        print('#################################################################################################################################')


##############################################################


##############################################################
def branching_unknown_residue(branch_name):
    def his_branch_rotamer_assignment(his_rotamer):
        his_ND = branch_name.topology.select('name ND1 and resid {}'.format(res_indx))[0]  # 114
        his_CE = branch_name.topology.select('name CE1 and resid {}'.format(res_indx))[0]  # 116
        his_NE = branch_name.topology.select('name NE2 and resid {}'.format(res_indx))[0]  # 117
        his_CD = branch_name.topology.select('name CD2 and resid {}'.format(res_indx))[0]  # 115

        if his_rotamer[-5:-2] == 'HIE':
            print('Residue name before conversion is {}'.format(branch_name.topology.residue(res_indx)))
            branch_name.topology.residue(res_indx).name = 'HIE'
            print('Residue name after conversion is {}'.format(branch_name.topology.residue(res_indx)))

            if his_rotamer[-2:] == '13':
                print('No atom flipping required!')

                branch_name.known_don = np.append(branch_name.known_don, his_ND)
                print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(his_ND))

                branch_name.known_acc = np.append(branch_name.known_acc, his_NE)
                print('Atom {} just got added to known_acc list ONCE. No complementary H-bond interaction.'.format(his_NE))

                branch_name.allpos_prot_don_acc = np.delete(branch_name.allpos_prot_don_acc, np.where(branch_name.allpos_prot_don_acc == his_CE)[0][0])
                branch_name.allpos_prot_don_acc = np.delete(branch_name.allpos_prot_don_acc, np.where(branch_name.allpos_prot_don_acc == his_CD)[0][0])
                branch_name.grease = np.append(branch_name.grease, his_CE)
                branch_name.grease = np.append(branch_name.grease, his_CD)

            elif his_rotamer[-2:] == '24':
                print('Atoms were:  {} {}, {} {}, {} {}, {} {}'.format(his_ND, branch_name.topology.atom(his_ND).name, his_CE, branch_name.topology.atom(his_CE).name, his_NE, branch_name.topology.atom(his_NE).name, his_CD, branch_name.topology.atom(his_CD).name))
                print('Atom names were just flipped')
                # We switch atom name and atom element from C to N and from N to C
                temp1 = branch_name.topology.atom(his_ND).name
                temp2 = branch_name.topology.atom(his_CE).name
                temp3 = branch_name.topology.atom(his_NE).name
                temp4 = branch_name.topology.atom(his_CD).name
                branch_name.topology.atom(his_ND).name = temp4
                branch_name.topology.atom(his_CE).name = temp3
                branch_name.topology.atom(his_NE).name = temp2
                branch_name.topology.atom(his_CD).name = temp1

                temp1_element = branch_name.topology.atom(his_ND).element
                temp2_element = branch_name.topology.atom(his_CE).element
                temp3_element = branch_name.topology.atom(his_NE).element
                temp4_element = branch_name.topology.atom(his_CD).element
                branch_name.topology.atom(his_ND).element = temp4_element
                branch_name.topology.atom(his_CE).element = temp3_element
                branch_name.topology.atom(his_NE).element = temp2_element
                branch_name.topology.atom(his_CD).element = temp1_element

                # We append those atoms that switch from C to N and from N to C
                branch_name.atm_name_change_to_N.append(his_CE)
                branch_name.atm_name_change_to_N.append(his_CD)
                branch_name.atm_name_change_to_C.append(his_ND)
                branch_name.atm_name_change_to_C.append(his_NE)

                print('Now, atoms are:  {} {}, {} {}, {} {}, {} {}'.format(his_ND, branch_name.topology.atom(his_ND).name, his_CE, branch_name.topology.atom(his_CE).name, his_NE, branch_name.topology.atom(his_NE).name, his_CD, branch_name.topology.atom(his_CD).name))

                branch_name.known_acc = np.append(branch_name.known_acc, his_CD)
                print('Atom {} just got added to known_acc list ONCE. No complementary H-bond interaction.'.format(his_CD))

                branch_name.known_don = np.append(branch_name.known_don, his_CE)
                print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(his_CE))

                # We remove the two atoms that were N from allpos_prot_don_acc list and we add to grease list
                branch_name.allpos_prot_don_acc = np.delete(branch_name.allpos_prot_don_acc, np.where(branch_name.allpos_prot_don_acc == his_ND)[0][0])
                branch_name.allpos_prot_don_acc = np.delete(branch_name.allpos_prot_don_acc, np.where(branch_name.allpos_prot_don_acc == his_NE)[0][0])
                branch_name.grease = np.append(branch_name.grease, his_ND)
                branch_name.grease = np.append(branch_name.grease, his_NE)

        elif his_rotamer[-5:-2] == 'HID':
            print('Residue name before conversion is {}'.format(branch_name.topology.atom(his_ND).residue))
            branch_name.topology.atom(his_ND).residue.name = 'HID'
            print('Residue name after conversion is {}'.format(branch_name.topology.atom(his_ND).residue))

            if his_rotamer[-2:] == '13':
                print('No atom flipping required!')

                branch_name.known_don = np.append(branch_name.known_don, his_ND)
                print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(his_ND))

                branch_name.known_acc = np.append(branch_name.known_acc, his_NE)
                print('Atom {} just got added to known_acc list ONCE. No complementary H-bond interaction.'.format(his_NE))

                branch_name.allpos_prot_don_acc = np.delete(branch_name.allpos_prot_don_acc, np.where(branch_name.allpos_prot_don_acc == his_CE)[0][0])
                branch_name.allpos_prot_don_acc = np.delete(branch_name.allpos_prot_don_acc, np.where(branch_name.allpos_prot_don_acc == his_CD)[0][0])
                branch_name.grease = np.append(branch_name.grease, his_CE)
                branch_name.grease = np.append(branch_name.grease, his_CD)

            elif his_rotamer[-2:] == '24':
                print('Atoms were:  {} {}, {} {}, {} {}, {} {}'.format(his_ND, branch_name.topology.atom(his_ND).name, his_CE, branch_name.topology.atom(his_CE).name, his_NE, branch_name.topology.atom(his_NE).name, his_CD, branch_name.topology.atom(his_CD).name))
                print('Atom names were just flipped')
                # We switch atom name and atom element from C to N and from N to C
                temp1 = branch_name.topology.atom(his_ND).name
                temp2 = branch_name.topology.atom(his_CE).name
                temp3 = branch_name.topology.atom(his_NE).name
                temp4 = branch_name.topology.atom(his_CD).name
                branch_name.topology.atom(his_ND).name = temp4
                branch_name.topology.atom(his_CE).name = temp3
                branch_name.topology.atom(his_NE).name = temp2
                branch_name.topology.atom(his_CD).name = temp1

                temp1_element = branch_name.topology.atom(his_ND).element
                temp2_element = branch_name.topology.atom(his_CE).element
                temp3_element = branch_name.topology.atom(his_NE).element
                temp4_element = branch_name.topology.atom(his_CD).element
                branch_name.topology.atom(his_ND).element = temp4_element
                branch_name.topology.atom(his_CE).element = temp3_element
                branch_name.topology.atom(his_NE).element = temp2_element
                branch_name.topology.atom(his_CD).element = temp1_element

                # We append those atoms that switch from C to N and from N to C
                branch_name.atm_name_change_to_N.append(his_CE)
                branch_name.atm_name_change_to_N.append(his_CD)
                branch_name.atm_name_change_to_C.append(his_ND)
                branch_name.atm_name_change_to_C.append(his_NE)

                print('Now, atoms are:  {} {}, {} {}, {} {}, {} {}'.format(his_ND, branch_name.topology.atom(his_ND).name, his_CE, branch_name.topology.atom(his_CE).name, his_NE, branch_name.topology.atom(his_NE).name, his_CD, branch_name.topology.atom(his_CD).name))

                branch_name.known_don = np.append(branch_name.known_don, his_CD)
                print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(his_CD))

                branch_name.known_acc = np.append(branch_name.known_acc, his_CE)
                print('Atom {} just got added to known_acc list ONCE. No complementary H-bond interaction.'.format(his_CE))

                # We remove the two atoms that were N from allpos_prot_don_acc list and we add to grease list
                branch_name.allpos_prot_don_acc = np.delete(branch_name.allpos_prot_don_acc, np.where(branch_name.allpos_prot_don_acc == his_ND)[0][0])
                branch_name.allpos_prot_don_acc = np.delete(branch_name.allpos_prot_don_acc, np.where(branch_name.allpos_prot_don_acc == his_NE)[0][0])
                branch_name.grease = np.append(branch_name.grease, his_ND)
                branch_name.grease = np.append(branch_name.grease, his_NE)

        elif his_rotamer[-5:-2] == 'HIP':
            print('Residue name before conversion is {}'.format(branch_name.topology.atom(his_ND).residue))
            branch_name.topology.atom(his_ND).residue.name = 'HIP'
            print('Residue name after conversion is {}'.format(branch_name.topology.atom(his_ND).residue))

            if his_rotamer[-2:] == '13':
                print('No atom flipping required!')

                branch_name.known_don = np.append(branch_name.known_don, his_ND)
                print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(his_ND))

                branch_name.known_don = np.append(branch_name.known_don, his_NE)
                print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(his_NE))

                branch_name.allpos_prot_don_acc = np.delete(branch_name.allpos_prot_don_acc, np.where(branch_name.allpos_prot_don_acc == his_CE)[0][0])
                branch_name.allpos_prot_don_acc = np.delete(branch_name.allpos_prot_don_acc, np.where(branch_name.allpos_prot_don_acc == his_CD)[0][0])
                branch_name.grease = np.append(branch_name.grease, his_CE)
                branch_name.grease = np.append(branch_name.grease, his_CD)

            elif his_rotamer[-2:] == '24':
                print('Atoms were:  {} {}, {} {}, {} {}, {} {}'.format(his_ND, branch_name.topology.atom(his_ND).name, his_CE, branch_name.topology.atom(his_CE).name, his_NE, branch_name.topology.atom(his_NE).name, his_CD, branch_name.topology.atom(his_CD).name))
                print('Atom names were just flipped')
                # We switch atom name and atom element from C to N and from N to C
                temp1 = branch_name.topology.atom(his_ND).name
                temp2 = branch_name.topology.atom(his_CE).name
                temp3 = branch_name.topology.atom(his_NE).name
                temp4 = branch_name.topology.atom(his_CD).name
                branch_name.topology.atom(his_ND).name = temp4
                branch_name.topology.atom(his_CE).name = temp3
                branch_name.topology.atom(his_NE).name = temp2
                branch_name.topology.atom(his_CD).name = temp1

                temp1_element = branch_name.topology.atom(his_ND).element
                temp2_element = branch_name.topology.atom(his_CE).element
                temp3_element = branch_name.topology.atom(his_NE).element
                temp4_element = branch_name.topology.atom(his_CD).element
                branch_name.topology.atom(his_ND).element = temp4_element
                branch_name.topology.atom(his_CE).element = temp3_element
                branch_name.topology.atom(his_NE).element = temp2_element
                branch_name.topology.atom(his_CD).element = temp1_element

                # We append those atoms that switch from C to N and from N to C
                branch_name.atm_name_change_to_N.append(his_CE)
                branch_name.atm_name_change_to_N.append(his_CD)
                branch_name.atm_name_change_to_C.append(his_ND)
                branch_name.atm_name_change_to_C.append(his_NE)

                print('Now, atoms are:  {} {}, {} {}, {} {}, {} {}'.format(his_ND, branch_name.topology.atom(his_ND).name, his_CE, branch_name.topology.atom(his_CE).name, his_NE, branch_name.topology.atom(his_NE).name, his_CD, branch_name.topology.atom(his_CD).name))

                branch_name.known_don = np.append(branch_name.known_don, his_CD)
                print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(his_CD))

                branch_name.known_don = np.append(branch_name.known_don, his_CE)
                print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(his_CE))

                # We remove the two atoms that were N from allpos_prot_don_acc list and we add to grease list
                branch_name.allpos_prot_don_acc = np.delete(branch_name.allpos_prot_don_acc, np.where(branch_name.allpos_prot_don_acc == his_ND)[0][0])
                branch_name.allpos_prot_don_acc = np.delete(branch_name.allpos_prot_don_acc, np.where(branch_name.allpos_prot_don_acc == his_NE)[0][0])
                branch_name.grease = np.append(branch_name.grease, his_ND)
                branch_name.grease = np.append(branch_name.grease, his_NE)

        # delete known residues from the unknown residues list
        branch_name.unknown_residues = np.delete(branch_name.unknown_residues, np.where(branch_name.unknown_residues == branch_name.topology.atom(his_ND).residue.index)[0][0])

    branch_name.children_of_branch = []
    res_indx = branch_name.unknown_residues[0]
    if branch_name.topology.residue(res_indx).name in ['SER', 'THR', 'TYR']:
        ########################
        # Branching a SER/THR/TYR residue
        ########################
        print('Branching out a hydroxyl containing residue (SER, THR, or TYR):')
        print(branch_name.topology.residue(res_indx))

        oxi_atm_ndx = branch_name.topology.select('((resname SER and name OG) or (resname THR and name OG1) or (resname TYR and name OH)) and resid {}'.format(res_indx))

        oxi_bond = len(oxi_atm_ndx)

        ohRes = defaultdict(dict)

        for i in np.arange(oxi_bond):
            oxi_nn = md.compute_neighbors(branch_name.structure, h_bond_heavy_atm_cutoff, np.asarray([oxi_atm_ndx[i]]), branch_name.allpos_prot_don_acc, periodic=False)[0]

            print('Atom {} is the hydroxyl group oxygen atom of {}'.format(oxi_atm_ndx[i], branch_name.topology.atom(oxi_atm_ndx[i]).residue))

            oxi_Rid = branch_name.topology.atom(oxi_atm_ndx[i]).residue.index

            # Delete backbone O and N of same residue from neighbor list
            oxi_bb = [atom.index for atom in branch_name.topology.atoms if (((atom.element.symbol == 'N') or (atom.element.symbol == 'O')) and (atom.residue.index == oxi_Rid) and atom.is_backbone)]  # list of backbone O and N of the looped residue
            for j in oxi_bb:
                if j in oxi_nn:
                    oxi_nn = np.delete(oxi_nn, np.where(oxi_nn == j)[0])

            # Delete from neighbor lists of side chain O and N of unknown_residues or any neighbor that is not available anymore to H-bond. those can be:
            # 1 - backbone N or O that is not available anymore for H-bond
            # OR 2 - side chain atoms whose residue is not in unknown_residues list (means they were already identified but no longer able to H-bond as they were used already).
            for k in oxi_nn:
                if k not in branch_name.known_don and k not in branch_name.known_acc and ((branch_name.topology.atom(k).name == 'N' or branch_name.topology.atom(k).name == 'O') or ((branch_name.topology.atom(k).name != 'N' or branch_name.topology.atom(k).name != 'O') and branch_name.topology.atom(k).residue.index not in branch_name.unknown_residues)):
                    oxi_nn = np.delete(oxi_nn, np.where(oxi_nn == k)[0][0])

            print('List of neighbors of the side chain O atom of {}:'.format(branch_name.topology.atom(oxi_atm_ndx[i]).residue))
            print(oxi_nn)

            # Here we filter out non Hbond neighbors that are having bad angles
            oxi_nn = filter_out_non_hbond_neighbors(branch_name, oxi_nn, oxi_atm_ndx[i])
            print('List of neighbors of the side chain O atom of {} after filtering out non H-bond neighbors:'.format(branch_name.topology.atom(oxi_atm_ndx[i]).residue))
            print(oxi_nn)

            oxi_nn_don = []
            oxi_nn_acc = []
            oxi_nn_unknown = []
            if len(oxi_nn) != 0:
                for k in oxi_nn:
                    if k in branch_name.known_don and k not in branch_name.known_acc:
                        oxi_nn_don.append(k)
                    elif k in branch_name.known_acc and k not in branch_name.known_don:
                        oxi_nn_acc.append(k)
                    else:
                        oxi_nn_unknown.append(k)

            if len(oxi_nn_don) == 0 and len(oxi_nn_acc) == 0:
                if len(oxi_nn_unknown) == 1:
                    oxi_nn_unknown = oxi_nn_unknown[0]
                    print('Neighbor atom {} ({}) is the only neighbored unknown.'.format(oxi_nn_unknown, branch_name.topology.atom(oxi_nn_unknown)))

                elif len(oxi_nn_unknown) > 1:
                    print('There are more than one unknown neighbor acting as H-bond acceptors: {},\n     I will only keep the closest unknown neighbor to set up the two branches.'.format(oxi_nn_unknown))
                    dist_dic = {}
                    for l in oxi_nn_unknown:
                        datoms = np.asarray([oxi_atm_ndx[i], l], dtype="int").reshape(1, 2)
                        dist = md.compute_distances(branch_name.structure, datoms)[0][0]
                        dist_dic[l] = dist
                    oxi_nn_unknown = sorted(dist_dic.items(), key=operator.itemgetter(1))[0][0]
                    print('Neighbor atom {} ({}) is the closest unknown.'.format(oxi_nn_unknown, branch_name.topology.atom(oxi_nn_unknown)))

            if branch_name.topology.atom(oxi_nn_unknown).residue.name in ['SER', 'THR', 'TYR']:
                print('The coupled residues are {} and {}.'.format(branch_name.topology.atom(oxi_atm_ndx[i]).residue, branch_name.topology.atom(oxi_nn_unknown).residue))

                name1 = str(branch_name.topology.atom(oxi_atm_ndx[i]).residue)
                name2 = str(branch_name.topology.atom(oxi_nn_unknown).residue)
                res1 = str(branch_name.topology.atom(oxi_atm_ndx[i]).residue.index)
                res2 = str(branch_name.topology.atom(oxi_nn_unknown).residue.index)

                ##########################################################################
                # CHILD 1
                globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'] = copy.deepcopy(branch_name)
                globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].name = 'res_indx_{}_{}_acc_res_indx_{}_{}_don'.format(res1, name1, res2, name2)
                globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].path.append(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].name)

                branch_name.children_of_branch.append(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].name)

                globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_don = np.append(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_don, oxi_atm_ndx[i])
                globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_acc = np.append(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_acc, oxi_nn_unknown)
                globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].unknown_residues = np.delete(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].unknown_residues, np.where(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].unknown_residues == globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_atm_ndx[i]).residue.index)[0][0])
                globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].unknown_residues = np.delete(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].unknown_residues, np.where(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].unknown_residues == globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_nn_unknown).residue.index)[0][0])
                globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].directly_set_donor_of_oxi_SER_THR_TYR.append([[oxi_nn_unknown, globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_nn_unknown), globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].xyz[0][oxi_nn_unknown]], [oxi_atm_ndx[i], globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_atm_ndx[i]), globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].xyz[0][oxi_atm_ndx[i]]]])
                ##########################################################################

                ##########################################################################
                # CHILD 2
                globals()['res_indx_' + res2 + '_' + name2 + '_acc_' + 'res_indx_' + res1 + '_' + name1 + '_don'] = copy.deepcopy(branch_name)
                globals()['res_indx_' + res2 + '_' + name2 + '_acc_' + 'res_indx_' + res1 + '_' + name1 + '_don'].name = 'res_indx_{}_{}_acc_res_indx_{}_{}_don'.format(res2, name2, res1, name1)
                globals()['res_indx_' + res2 + '_' + name2 + '_acc_' + 'res_indx_' + res1 + '_' + name1 + '_don'].path.append(globals()['res_indx_' + res2 + '_' + name2 + '_acc_' + 'res_indx_' + res1 + '_' + name1 + '_don'].name)

                branch_name.children_of_branch.append(globals()['res_indx_' + res2 + '_' + name2 + '_acc_' + 'res_indx_' + res1 + '_' + name1 + '_don'].name)

                globals()['res_indx_' + res2 + '_' + name2 + '_acc_' + 'res_indx_' + res1 + '_' + name1 + '_don'].known_don = np.append(globals()['res_indx_' + res2 + '_' + name2 + '_acc_' + 'res_indx_' + res1 + '_' + name1 + '_don'].known_don, oxi_nn_unknown)
                globals()['res_indx_' + res2 + '_' + name2 + '_acc_' + 'res_indx_' + res1 + '_' + name1 + '_don'].known_acc = np.append(globals()['res_indx_' + res2 + '_' + name2 + '_acc_' + 'res_indx_' + res1 + '_' + name1 + '_don'].known_acc, oxi_atm_ndx[i])
                globals()['res_indx_' + res2 + '_' + name2 + '_acc_' + 'res_indx_' + res1 + '_' + name1 + '_don'].unknown_residues = np.delete(globals()['res_indx_' + res2 + '_' + name2 + '_acc_' + 'res_indx_' + res1 + '_' + name1 + '_don'].unknown_residues, np.where(globals()['res_indx_' + res2 + '_' + name2 + '_acc_' + 'res_indx_' + res1 + '_' + name1 + '_don'].unknown_residues == globals()['res_indx_' + res2 + '_' + name2 + '_acc_' + 'res_indx_' + res1 + '_' + name1 + '_don'].topology.atom(oxi_atm_ndx[i]).residue.index)[0][0])
                globals()['res_indx_' + res2 + '_' + name2 + '_acc_' + 'res_indx_' + res1 + '_' + name1 + '_don'].unknown_residues = np.delete(globals()['res_indx_' + res2 + '_' + name2 + '_acc_' + 'res_indx_' + res1 + '_' + name1 + '_don'].unknown_residues, np.where(globals()['res_indx_' + res2 + '_' + name2 + '_acc_' + 'res_indx_' + res1 + '_' + name1 + '_don'].unknown_residues == globals()['res_indx_' + res2 + '_' + name2 + '_acc_' + 'res_indx_' + res1 + '_' + name1 + '_don'].topology.atom(oxi_nn_unknown).residue.index)[0][0])
                globals()['res_indx_' + res2 + '_' + name2 + '_acc_' + 'res_indx_' + res1 + '_' + name1 + '_don'].directly_set_donor_of_oxi_SER_THR_TYR.append([[oxi_atm_ndx[i], globals()['res_indx_' + res2 + '_' + name2 + '_acc_' + 'res_indx_' + res1 + '_' + name1 + '_don'].topology.atom(oxi_atm_ndx[i]), globals()['res_indx_' + res2 + '_' + name2 + '_acc_' + 'res_indx_' + res1 + '_' + name1 + '_don'].xyz[0][oxi_atm_ndx[i]]], [oxi_nn_unknown, globals()['res_indx_' + res2 + '_' + name2 + '_acc_' + 'res_indx_' + res1 + '_' + name1 + '_don'].topology.atom(oxi_nn_unknown), globals()['res_indx_' + res2 + '_' + name2 + '_acc_' + 'res_indx_' + res1 + '_' + name1 + '_don'].xyz[0][oxi_nn_unknown]]])
                ##########################################################################

                print('There will be two child branches here:\n1) {} - in which {} accepts H-bond with {}. This will make {} appended to known_don list once and {} appended to known_acc list once. \n2) {} - in which {} accepts H-bond with {}. This will make {} appended to known_don list once and {} appended to known_acc list once.'.format(
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].name,
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_atm_ndx[i]),
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_nn_unknown),
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_atm_ndx[i]),
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_nn_unknown), globals()['res_indx_' + res2 + '_' + name2 + '_acc_' + 'res_indx_' + res1 + '_' + name1 + '_don'].name, globals()['res_indx_' + res2 + '_' + name2 + '_acc_' + 'res_indx_' + res1 + '_' + name1 + '_don'].topology.atom(oxi_nn_unknown), globals()['res_indx_' + res2 + '_' + name2 + '_acc_' + 'res_indx_' + res1 + '_' + name1 + '_don'].topology.atom(oxi_atm_ndx[i]), globals()['res_indx_' + res2 + '_' + name2 + '_acc_' + 'res_indx_' + res1 + '_' + name1 + '_don'].topology.atom(oxi_nn_unknown), globals()['res_indx_' + res2 + '_' + name2 + '_acc_' + 'res_indx_' + res1 + '_' + name1 + '_don'].topology.atom(oxi_atm_ndx[i])))

            elif branch_name.topology.atom(oxi_nn_unknown).residue.name in ['ASN', 'GLN']:
                print('The coupled residues are {} and {}.'.format(branch_name.topology.atom(oxi_atm_ndx[i]).residue, branch_name.topology.atom(oxi_nn_unknown).residue))

                if branch_name.topology.atom(oxi_nn_unknown).name in ['ND2', 'NE2']:

                    oxi_nn_unknown_oxygen_end = branch_name.topology.select('((resname ASN and name OD1) or (resname GLN and name OE1)) and resid {}'.format(branch_name.topology.atom(oxi_nn_unknown).residue.index))[0]
                    oxi_nn_unknown_nitrogen_end = oxi_nn_unknown

                    ##########################################################################
                    # CHILD 1
                    name1 = str(branch_name.topology.atom(oxi_atm_ndx[i]).residue)
                    name2 = str(branch_name.topology.atom(oxi_nn_unknown_nitrogen_end).residue)
                    res1 = str(branch_name.topology.atom(oxi_atm_ndx[i]).residue.index)
                    res2 = str(branch_name.topology.atom(oxi_nn_unknown_nitrogen_end).residue.index)

                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'] = copy.deepcopy(branch_name)
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].name = 'res_indx_{}_{}_acc_res_indx_{}_{}_don'.format(res1, name1, res2, name2)
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].path.append(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].name)

                    print('The first child branch is {}'.format(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].name))

                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_don = np.append(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_don, oxi_atm_ndx[i])
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_don = np.append(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_don, oxi_nn_unknown_nitrogen_end)

                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_acc = np.append(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_acc, oxi_nn_unknown_oxygen_end)
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_acc = np.append(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_acc, oxi_nn_unknown_oxygen_end)

                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].unknown_residues = np.delete(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].unknown_residues, np.where(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].unknown_residues == globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_atm_ndx[i]).residue.index)[0][0])
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].unknown_residues = np.delete(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].unknown_residues, np.where(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].unknown_residues == globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_nn_unknown_nitrogen_end).residue.index)[0][0])

                    branch_name.children_of_branch.append(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].name)

                    print('#######')
                    ##########################################################################

                    ##########################################################################
                    # CHILD 2

                    name1 = str(branch_name.topology.atom(oxi_nn_unknown_oxygen_end).residue)
                    name2 = str(branch_name.topology.atom(oxi_atm_ndx[i]).residue)
                    res1 = str(branch_name.topology.atom(oxi_nn_unknown_oxygen_end).residue.index)
                    res2 = str(branch_name.topology.atom(oxi_atm_ndx[i]).residue.index)

                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'] = copy.deepcopy(branch_name)
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].name = 'res_indx_{}_{}_acc_res_indx_{}_{}_don'.format(res1, name1, res2, name2)
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].path.append(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].name)

                    print('The second child branch is {}'.format(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].name))

                    print('Atoms were {} ({}) and {} ({})'.format(oxi_nn_unknown_oxygen_end, globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_nn_unknown_oxygen_end).name, oxi_nn_unknown_nitrogen_end, globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_nn_unknown_nitrogen_end).name))
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].switched_atm_name_uncharged.append((oxi_nn_unknown, oxi_nn_unknown_nitrogen_end))
                    tmp_O = globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_nn_unknown_oxygen_end).name
                    tmp_N = globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_nn_unknown_nitrogen_end).name
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_nn_unknown_oxygen_end).name = tmp_N
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_nn_unknown_nitrogen_end).name = tmp_O

                    tmp_O_element = globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_nn_unknown_oxygen_end).element
                    tmp_N_element = globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_nn_unknown_nitrogen_end).element
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_nn_unknown_oxygen_end).element = tmp_N_element
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_nn_unknown_nitrogen_end).element = tmp_O_element

                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].atm_name_change_to_N.append(oxi_nn_unknown_oxygen_end)
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].atm_name_change_to_O.append(oxi_nn_unknown_nitrogen_end)
                    print('Atoms just flipped to accommodate complementarity for H-bonding')
                    print('Atoms are now {} ({}) and {} ({})'.format(oxi_nn_unknown_oxygen_end, globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_nn_unknown_oxygen_end).name, oxi_nn_unknown_nitrogen_end, globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_nn_unknown_nitrogen_end).name))

                    # oxi_nn_unknown_nitrogen_end is now the oxygen end
                    # oxi_nn_unknown_oxygen_end is now the nitrogen end

                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_acc = np.append(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_acc, oxi_atm_ndx[i])
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_acc = np.append(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_acc, oxi_nn_unknown_nitrogen_end)

                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_don = np.append(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_don, oxi_nn_unknown_oxygen_end)
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_don = np.append(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_don, oxi_nn_unknown_oxygen_end)

                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].directly_set_donor_of_oxi_SER_THR_TYR.append([[oxi_atm_ndx[i], globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_atm_ndx[i]), globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].xyz[0][oxi_atm_ndx[i]]], [oxi_nn_unknown_nitrogen_end, globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_nn_unknown_nitrogen_end), globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].xyz[0][oxi_nn_unknown_nitrogen_end]]])

                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].unknown_residues = np.delete(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].unknown_residues, np.where(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].unknown_residues == globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_atm_ndx[i]).residue.index)[0][0])
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].unknown_residues = np.delete(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].unknown_residues, np.where(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].unknown_residues == globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_nn_unknown_nitrogen_end).residue.index)[0][0])

                    branch_name.children_of_branch.append(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].name)

                    print('#######')
                    ##########################################################################

                elif branch_name.topology.atom(oxi_nn_unknown).name in ['OD1', 'OE1']:

                    oxi_nn_unknown_oxygen_end = oxi_nn_unknown
                    oxi_nn_unknown_nitrogen_end = branch_name.topology.select('((resname ASN and name ND2) or (resname GLN and name NE2)) and resid {}'.format(branch_name.topology.atom(oxi_nn_unknown).residue.index))[0]

                    ##########################################################################
                    # CHILD 1

                    name1 = str(branch_name.topology.atom(oxi_atm_ndx[i]).residue)
                    name2 = str(branch_name.topology.atom(oxi_nn_unknown_nitrogen_end).residue)
                    res1 = str(branch_name.topology.atom(oxi_atm_ndx[i]).residue.index)
                    res2 = str(branch_name.topology.atom(oxi_nn_unknown_nitrogen_end).residue.index)

                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'] = copy.deepcopy(branch_name)
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].name = 'res_indx_{}_{}_acc_res_indx_{}_{}_don'.format(res1, name1, res2, name2)
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].path.append(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].name)

                    print('The first child branch is {}'.format(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].name))

                    print('Atoms were {} ({}) and {} ({})'.format(oxi_nn_unknown_oxygen_end, globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_nn_unknown_oxygen_end).name, oxi_nn_unknown_nitrogen_end, globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_nn_unknown_nitrogen_end).name))
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].switched_atm_name_uncharged.append((oxi_nn_unknown, oxi_nn_unknown_nitrogen_end))
                    tmp_O = globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_nn_unknown_oxygen_end).name
                    tmp_N = globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_nn_unknown_nitrogen_end).name
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_nn_unknown_oxygen_end).name = tmp_N
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_nn_unknown_nitrogen_end).name = tmp_O

                    tmp_O_element = globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_nn_unknown_oxygen_end).element
                    tmp_N_element = globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_nn_unknown_nitrogen_end).element
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_nn_unknown_oxygen_end).element = tmp_N_element
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_nn_unknown_nitrogen_end).element = tmp_O_element

                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].atm_name_change_to_N.append(oxi_nn_unknown_oxygen_end)
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].atm_name_change_to_O.append(oxi_nn_unknown_nitrogen_end)
                    print('Atoms just flipped to accommodate complementarity for H-bonding')
                    print('Atoms are now {} ({}) and {} ({})'.format(oxi_nn_unknown_oxygen_end, globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_nn_unknown_oxygen_end).name, oxi_nn_unknown_nitrogen_end, globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_nn_unknown_nitrogen_end).name))

                    # oxi_nn_unknown_nitrogen_end is now the oxygen end
                    # oxi_nn_unknown_oxygen_end is now the nitrogen end

                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_don = np.append(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_don, oxi_atm_ndx[i])
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_don = np.append(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_don, oxi_nn_unknown_oxygen_end)

                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_acc = np.append(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_acc, oxi_nn_unknown_nitrogen_end)
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_acc = np.append(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_acc, oxi_nn_unknown_nitrogen_end)

                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].unknown_residues = np.delete(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].unknown_residues, np.where(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].unknown_residues == globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_atm_ndx[i]).residue.index)[0][0])
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].unknown_residues = np.delete(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].unknown_residues, np.where(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].unknown_residues == globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_nn_unknown_nitrogen_end).residue.index)[0][0])

                    branch_name.children_of_branch.append(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].name)

                    print('#######')
                    ##########################################################################

                    ##########################################################################
                    # CHILD 2

                    name1 = str(branch_name.topology.atom(oxi_nn_unknown_oxygen_end).residue)
                    name2 = str(branch_name.topology.atom(oxi_atm_ndx[i]).residue)
                    res1 = str(branch_name.topology.atom(oxi_nn_unknown_oxygen_end).residue.index)
                    res2 = str(branch_name.topology.atom(oxi_atm_ndx[i]).residue.index)

                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'] = copy.deepcopy(branch_name)
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].name = 'res_indx_{}_{}_acc_res_indx_{}_{}_don'.format(res1, name1, res2, name2)
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].path.append(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].name)

                    print('The second child branch is {}'.format(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].name))

                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_acc = np.append(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_acc, oxi_atm_ndx[i])
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_acc = np.append(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_acc, oxi_nn_unknown_oxygen_end)

                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_don = np.append(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_don, oxi_nn_unknown_nitrogen_end)
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_don = np.append(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].known_don, oxi_nn_unknown_nitrogen_end)

                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].directly_set_donor_of_oxi_SER_THR_TYR.append([[oxi_atm_ndx[i], globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_atm_ndx[i]), globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].xyz[0][oxi_atm_ndx[i]]], [oxi_nn_unknown_oxygen_end, globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_nn_unknown_oxygen_end), globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].xyz[0][oxi_nn_unknown_oxygen_end]]])

                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].unknown_residues = np.delete(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].unknown_residues, np.where(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].unknown_residues == globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_atm_ndx[i]).residue.index)[0][0])
                    globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].unknown_residues = np.delete(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].unknown_residues, np.where(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].unknown_residues == globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].topology.atom(oxi_nn_unknown_oxygen_end).residue.index)[0][0])

                    branch_name.children_of_branch.append(globals()['res_indx_' + res1 + '_' + name1 + '_acc_' + 'res_indx_' + res2 + '_' + name2 + '_don'].name)

                    print('#######')  ##########################################################################

            elif branch_name.topology.atom(oxi_nn_unknown).residue.name in ['HIS', 'HIE', 'HID', 'HIP']:
                print("The coupled residues are SER/TYR/THR and HIS. The easiest way to handle this is to move the index of the HIS residue to the front of the unknown_residue list so it can be branched out and that would set the SER/TYR/THR in reverse.")
                # user_decision = input("This pause was to ensure bringing this to your attention. To continue with the analysis, press any letter then ENTER.")
                # pass
                print("unknown_residues list order WAS {}.".format(branch_name.unknown_residues))
                his_residue_index_that_needs_branching_instead = branch_name.topology.atom(oxi_nn_unknown).residue.index
                branch_name.unknown_residues = np.delete(branch_name.unknown_residues, list(branch_name.unknown_residues).index(his_residue_index_that_needs_branching_instead))
                branch_name.unknown_residues = np.insert(branch_name.unknown_residues, 0, his_residue_index_that_needs_branching_instead)
                print("unknown_residues list order IS NOW {}.".format(branch_name.unknown_residues))
                branch_name.children_of_branch = []

    elif branch_name.topology.residue(res_indx).name in ['ASN', 'GLN']:
        ########################
        # Branching a ASN/GLN residue
        ########################
        print('Branching out an ASN/GLN residue:')
        print(branch_name.topology.residue(res_indx))

        unknown_asn_gln_oxygen_end = branch_name.topology.select('((resname ASN and name OD1) or (resname GLN and name OE1)) and resid {}'.format(res_indx))[0]
        unknown_asn_gln_nitrogen_end = branch_name.topology.select('((resname ASN and name ND2) or (resname GLN and name NE2)) and resid {}'.format(res_indx))[0]

        name1 = str(branch_name.topology.residue(res_indx))
        res1 = str(res_indx)

        ##########################################################################
        # CHILD 1

        globals()['res_indx_' + res1 + '_' + name1 + '_flip_a'] = copy.deepcopy(branch_name)
        globals()['res_indx_' + res1 + '_' + name1 + '_flip_a'].name = 'res_indx_{}_{}_flip_a'.format(res1, name1)
        globals()['res_indx_' + res1 + '_' + name1 + '_flip_a'].path.append(globals()['res_indx_' + res1 + '_' + name1 + '_flip_a'].name)

        print('The first child branch is {}'.format(globals()['res_indx_' + res1 + '_' + name1 + '_flip_a'].name))

        globals()['res_indx_' + res1 + '_' + name1 + '_flip_a'].known_don = np.append(globals()['res_indx_' + res1 + '_' + name1 + '_flip_a'].known_don, unknown_asn_gln_nitrogen_end)
        globals()['res_indx_' + res1 + '_' + name1 + '_flip_a'].known_don = np.append(globals()['res_indx_' + res1 + '_' + name1 + '_flip_a'].known_don, unknown_asn_gln_nitrogen_end)

        globals()['res_indx_' + res1 + '_' + name1 + '_flip_a'].known_acc = np.append(globals()['res_indx_' + res1 + '_' + name1 + '_flip_a'].known_acc, unknown_asn_gln_oxygen_end)
        globals()['res_indx_' + res1 + '_' + name1 + '_flip_a'].known_acc = np.append(globals()['res_indx_' + res1 + '_' + name1 + '_flip_a'].known_acc, unknown_asn_gln_oxygen_end)

        globals()['res_indx_' + res1 + '_' + name1 + '_flip_a'].unknown_residues = np.delete(globals()['res_indx_' + res1 + '_' + name1 + '_flip_a'].unknown_residues, np.where(globals()['res_indx_' + res1 + '_' + name1 + '_flip_a'].unknown_residues == res_indx)[0][0])

        branch_name.children_of_branch.append(globals()['res_indx_' + res1 + '_' + name1 + '_flip_a'].name)

        print('#######')
        ##########################################################################

        ##########################################################################
        # CHILD 2

        name1 = str(branch_name.topology.residue(res_indx))

        globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'] = copy.deepcopy(branch_name)
        globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'].name = 'res_indx_{}_{}_flip_b'.format(res1, name1)
        globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'].path.append(globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'].name)

        print('The second child branch is {}'.format(globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'].name))

        print('Atoms were {} ({}) and {} ({})'.format(unknown_asn_gln_oxygen_end, globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'].topology.atom(unknown_asn_gln_oxygen_end).name, unknown_asn_gln_nitrogen_end, globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'].topology.atom(unknown_asn_gln_nitrogen_end).name))
        globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'].switched_atm_name_uncharged.append((unknown_asn_gln_oxygen_end, unknown_asn_gln_nitrogen_end))
        tmp_O = globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'].topology.atom(unknown_asn_gln_oxygen_end).name
        tmp_N = globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'].topology.atom(unknown_asn_gln_nitrogen_end).name
        globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'].topology.atom(unknown_asn_gln_oxygen_end).name = tmp_N
        globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'].topology.atom(unknown_asn_gln_nitrogen_end).name = tmp_O

        tmp_O_element = globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'].topology.atom(unknown_asn_gln_oxygen_end).element
        tmp_N_element = globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'].topology.atom(unknown_asn_gln_nitrogen_end).element
        globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'].topology.atom(unknown_asn_gln_oxygen_end).element = tmp_N_element
        globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'].topology.atom(unknown_asn_gln_nitrogen_end).element = tmp_O_element

        globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'].atm_name_change_to_N.append(unknown_asn_gln_oxygen_end)
        globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'].atm_name_change_to_O.append(unknown_asn_gln_nitrogen_end)
        print('Atoms just flipped to accommodate complementarity for H-bonding')
        print('Atoms are now {} ({}) and {} ({})'.format(unknown_asn_gln_oxygen_end, globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'].topology.atom(unknown_asn_gln_oxygen_end).name, unknown_asn_gln_nitrogen_end, globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'].topology.atom(unknown_asn_gln_nitrogen_end).name))

        # unknown_asn_gln_nitrogen_end is now the oxygen end
        # unknown_asn_gln_oxygen_end is now the nitrogen end

        globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'].known_acc = np.append(globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'].known_acc, unknown_asn_gln_nitrogen_end)
        globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'].known_acc = np.append(globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'].known_acc, unknown_asn_gln_nitrogen_end)

        globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'].known_don = np.append(globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'].known_don, unknown_asn_gln_oxygen_end)
        globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'].known_don = np.append(globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'].known_don, unknown_asn_gln_oxygen_end)

        globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'].unknown_residues = np.delete(globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'].unknown_residues, np.where(globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'].unknown_residues == res_indx)[0][0])

        branch_name.children_of_branch.append(globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'].name)

        print('#######')
        ##########################################################################

        print('There will be two child branches here:\n1) {} . This will make {} ({}) appended to known_don list twice, {} ({}) appended to known_acc list twice.\n2) {} . This will make {} ({}) appended to known_don list twice, {} ({}) appended to known_acc list twice.'.format(
            globals()['res_indx_' + res1 + '_' + name1 + '_flip_a'].name, unknown_asn_gln_nitrogen_end, globals()['res_indx_' + res1 + '_' + name1 + '_flip_a'].topology.atom(unknown_asn_gln_nitrogen_end), unknown_asn_gln_oxygen_end, globals()['res_indx_' + res1 + '_' + name1 + '_flip_a'].topology.atom(unknown_asn_gln_oxygen_end), globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'].name, unknown_asn_gln_oxygen_end, globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'].topology.atom(unknown_asn_gln_oxygen_end), unknown_asn_gln_nitrogen_end, globals()['res_indx_' + res1 + '_' + name1 + '_flip_b'].topology.atom(unknown_asn_gln_nitrogen_end)))

        print('#######')

    elif branch_name.topology.residue(res_indx).name in ['HIS', 'HIE', 'HID', 'HIP']:
        ########################
        # Branching a HIS residue
        ########################
        print('Branching out a HIS residue:')
        print(branch_name.topology.residue(res_indx))

        if res_indx in branch_name.his_with_splits:
            print('Residue index {} - {} - splits into two or more rotamer possibilities.'.format(res_indx, branch_name.topology.residue(res_indx)))

            his_rotamer_winners = branch_name.his_with_splits[res_indx]

            # # locating the list which contain the res_index residue and the rotamers of this branching in the branch_name.his_with_splits list
            # # https: // www.geeksforgeeks.org / python - key - index - in -dictionary /
            # # we need to know the arrangement to see from the known_donor or the known_acceptor list I shall remove the neighbor that's is being used in the interaction so it can't be used again.
            # temp = branch_name.his_with_splits
            # res = [idx for idx, key in enumerate(temp) if key[0] == res_indx]
            #
            # his_rotamer_winners = branch_name.his_with_splits[res[0]][1:]

        elif res_indx not in branch_name.his_with_splits:  # If a HIS residue is unknown and isn't an obvious split between just two rotamers, are we considering the 6 HIS rotamer possibilities?
            # print('Residue index {} - {} - can be one of the 6 rotamer possibilities.'.format(res_indx, branch_name.topology.residue(res_indx)))
            # his_rotamer_winners = ['HIE13', 'HIE24', 'HID13', 'HID24', 'HIP13', 'HIP24']
            print('Residue index {} - {} - can be one of those 4 rotamer possibilities? [HIE13, HIE24, HID13, HID24]'.format(res_indx, branch_name.topology.residue(res_indx)))
            his_rotamer_winners = ['HIE13', 'HIE24', 'HID13', 'HID24']
        ##########################################################################
        print('There will be {} child branches from this unknown residue.'.format(len(his_rotamer_winners)))

        print('#######')

        for rotamer_index in np.arange(len(his_rotamer_winners)):
            # CHILD {rotamer_index}
            name1 = str(branch_name.topology.residue(res_indx))
            res1 = str(res_indx)
            name2 = his_rotamer_winners[rotamer_index]
            globals()['res_indx_' + res1 + '_' + name1 + '_' + name2] = copy.deepcopy(branch_name)
            globals()['res_indx_' + res1 + '_' + name1 + '_' + name2].name = 'res_indx_{}_{}_{}'.format(res1, name1, name2)
            globals()['res_indx_' + res1 + '_' + name1 + '_' + name2].path.append(globals()['res_indx_' + res1 + '_' + name1 + '_' + name2].name)

            print('The child branch {} is {}'.format(rotamer_index + 1, globals()['res_indx_' + res1 + '_' + name1 + '_' + name2].name))

            print('#######')

            branch_name.children_of_branch.append(globals()['res_indx_' + res1 + '_' + name1 + '_' + name2].name)
        ##########################################################################

        for his_rotamer in branch_name.children_of_branch:
            his_ND = globals()[his_rotamer].topology.select('name ND1 and resid {}'.format(res_indx))[0]  # 114
            his_CE = globals()[his_rotamer].topology.select('name CE1 and resid {}'.format(res_indx))[0]  # 116
            his_NE = globals()[his_rotamer].topology.select('name NE2 and resid {}'.format(res_indx))[0]  # 117
            his_CD = globals()[his_rotamer].topology.select('name CD2 and resid {}'.format(res_indx))[0]  # 115

            if his_rotamer[-5:-2] == 'HIE':
                print('Residue name before conversion is {}'.format(globals()[his_rotamer].topology.residue(res_indx)))
                globals()[his_rotamer].topology.residue(res_indx).name = 'HIE'
                print('Residue name after conversion is {}'.format(globals()[his_rotamer].topology.residue(res_indx)))

                if his_rotamer[-2:] == '13':
                    print('No atom flipping required!')

                    globals()[his_rotamer].known_don = np.append(globals()[his_rotamer].known_don, his_ND)
                    print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(his_ND))

                    globals()[his_rotamer].known_acc = np.append(globals()[his_rotamer].known_acc, his_NE)
                    print('Atom {} just got added to known_acc list ONCE. No complementary H-bond interaction.'.format(his_NE))

                    globals()[his_rotamer].allpos_prot_don_acc = np.delete(globals()[his_rotamer].allpos_prot_don_acc, np.where(globals()[his_rotamer].allpos_prot_don_acc == his_CE)[0][0])
                    globals()[his_rotamer].allpos_prot_don_acc = np.delete(globals()[his_rotamer].allpos_prot_don_acc, np.where(globals()[his_rotamer].allpos_prot_don_acc == his_CD)[0][0])
                    globals()[his_rotamer].grease = np.append(globals()[his_rotamer].grease, his_CE)
                    globals()[his_rotamer].grease = np.append(globals()[his_rotamer].grease, his_CD)

                elif his_rotamer[-2:] == '24':
                    print('Atoms were:  {} {}, {} {}, {} {}, {} {}'.format(his_ND, globals()[his_rotamer].topology.atom(his_ND).name, his_CE, globals()[his_rotamer].topology.atom(his_CE).name, his_NE, globals()[his_rotamer].topology.atom(his_NE).name, his_CD, globals()[his_rotamer].topology.atom(his_CD).name))
                    print('Atom names were just flipped')
                    # We switch atom name and atom element from C to N and from N to C
                    temp1 = globals()[his_rotamer].topology.atom(his_ND).name
                    temp2 = globals()[his_rotamer].topology.atom(his_CE).name
                    temp3 = globals()[his_rotamer].topology.atom(his_NE).name
                    temp4 = globals()[his_rotamer].topology.atom(his_CD).name
                    globals()[his_rotamer].topology.atom(his_ND).name = temp4
                    globals()[his_rotamer].topology.atom(his_CE).name = temp3
                    globals()[his_rotamer].topology.atom(his_NE).name = temp2
                    globals()[his_rotamer].topology.atom(his_CD).name = temp1

                    temp1_element = globals()[his_rotamer].topology.atom(his_ND).element
                    temp2_element = globals()[his_rotamer].topology.atom(his_CE).element
                    temp3_element = globals()[his_rotamer].topology.atom(his_NE).element
                    temp4_element = globals()[his_rotamer].topology.atom(his_CD).element
                    globals()[his_rotamer].topology.atom(his_ND).element = temp4_element
                    globals()[his_rotamer].topology.atom(his_CE).element = temp3_element
                    globals()[his_rotamer].topology.atom(his_NE).element = temp2_element
                    globals()[his_rotamer].topology.atom(his_CD).element = temp1_element

                    # We append those atoms that switch from C to N and from N to C
                    globals()[his_rotamer].atm_name_change_to_N.append(his_CE)
                    globals()[his_rotamer].atm_name_change_to_N.append(his_CD)
                    globals()[his_rotamer].atm_name_change_to_C.append(his_ND)
                    globals()[his_rotamer].atm_name_change_to_C.append(his_NE)

                    print('Now, atoms are:  {} {}, {} {}, {} {}, {} {}'.format(his_ND, globals()[his_rotamer].topology.atom(his_ND).name, his_CE, globals()[his_rotamer].topology.atom(his_CE).name, his_NE, globals()[his_rotamer].topology.atom(his_NE).name, his_CD, globals()[his_rotamer].topology.atom(his_CD).name))

                    globals()[his_rotamer].known_acc = np.append(globals()[his_rotamer].known_acc, his_CD)
                    print('Atom {} just got added to known_acc list ONCE. No complementary H-bond interaction.'.format(his_CD))

                    globals()[his_rotamer].known_don = np.append(globals()[his_rotamer].known_don, his_CE)
                    print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(his_CE))

                    # We remove the two atoms that were N from allpos_prot_don_acc list and we add to grease list
                    globals()[his_rotamer].allpos_prot_don_acc = np.delete(globals()[his_rotamer].allpos_prot_don_acc, np.where(globals()[his_rotamer].allpos_prot_don_acc == his_ND)[0][0])
                    globals()[his_rotamer].allpos_prot_don_acc = np.delete(globals()[his_rotamer].allpos_prot_don_acc, np.where(globals()[his_rotamer].allpos_prot_don_acc == his_NE)[0][0])
                    globals()[his_rotamer].grease = np.append(globals()[his_rotamer].grease, his_ND)
                    globals()[his_rotamer].grease = np.append(globals()[his_rotamer].grease, his_NE)

            elif his_rotamer[-5:-2] == 'HID':
                print('Residue name before conversion is {}'.format(globals()[his_rotamer].topology.atom(his_ND).residue))
                globals()[his_rotamer].topology.atom(his_ND).residue.name = 'HID'
                print('Residue name after conversion is {}'.format(globals()[his_rotamer].topology.atom(his_ND).residue))

                if his_rotamer[-2:] == '13':
                    print('No atom flipping required!')

                    globals()[his_rotamer].known_don = np.append(globals()[his_rotamer].known_don, his_ND)
                    print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(his_ND))

                    globals()[his_rotamer].known_acc = np.append(globals()[his_rotamer].known_acc, his_NE)
                    print('Atom {} just got added to known_acc list ONCE. No complementary H-bond interaction.'.format(his_NE))

                    globals()[his_rotamer].allpos_prot_don_acc = np.delete(globals()[his_rotamer].allpos_prot_don_acc, np.where(globals()[his_rotamer].allpos_prot_don_acc == his_CE)[0][0])
                    globals()[his_rotamer].allpos_prot_don_acc = np.delete(globals()[his_rotamer].allpos_prot_don_acc, np.where(globals()[his_rotamer].allpos_prot_don_acc == his_CD)[0][0])
                    globals()[his_rotamer].grease = np.append(globals()[his_rotamer].grease, his_CE)
                    globals()[his_rotamer].grease = np.append(globals()[his_rotamer].grease, his_CD)

                elif his_rotamer[-2:] == '24':
                    print('Atoms were:  {} {}, {} {}, {} {}, {} {}'.format(his_ND, globals()[his_rotamer].topology.atom(his_ND).name, his_CE, globals()[his_rotamer].topology.atom(his_CE).name, his_NE, globals()[his_rotamer].topology.atom(his_NE).name, his_CD, globals()[his_rotamer].topology.atom(his_CD).name))
                    print('Atom names were just flipped')
                    # We switch atom name and atom element from C to N and from N to C
                    temp1 = globals()[his_rotamer].topology.atom(his_ND).name
                    temp2 = globals()[his_rotamer].topology.atom(his_CE).name
                    temp3 = globals()[his_rotamer].topology.atom(his_NE).name
                    temp4 = globals()[his_rotamer].topology.atom(his_CD).name
                    globals()[his_rotamer].topology.atom(his_ND).name = temp4
                    globals()[his_rotamer].topology.atom(his_CE).name = temp3
                    globals()[his_rotamer].topology.atom(his_NE).name = temp2
                    globals()[his_rotamer].topology.atom(his_CD).name = temp1

                    temp1_element = globals()[his_rotamer].topology.atom(his_ND).element
                    temp2_element = globals()[his_rotamer].topology.atom(his_CE).element
                    temp3_element = globals()[his_rotamer].topology.atom(his_NE).element
                    temp4_element = globals()[his_rotamer].topology.atom(his_CD).element
                    globals()[his_rotamer].topology.atom(his_ND).element = temp4_element
                    globals()[his_rotamer].topology.atom(his_CE).element = temp3_element
                    globals()[his_rotamer].topology.atom(his_NE).element = temp2_element
                    globals()[his_rotamer].topology.atom(his_CD).element = temp1_element

                    # We append those atoms that switch from C to N and from N to C
                    globals()[his_rotamer].atm_name_change_to_N.append(his_CE)
                    globals()[his_rotamer].atm_name_change_to_N.append(his_CD)
                    globals()[his_rotamer].atm_name_change_to_C.append(his_ND)
                    globals()[his_rotamer].atm_name_change_to_C.append(his_NE)

                    print('Now, atoms are:  {} {}, {} {}, {} {}, {} {}'.format(his_ND, globals()[his_rotamer].topology.atom(his_ND).name, his_CE, globals()[his_rotamer].topology.atom(his_CE).name, his_NE, globals()[his_rotamer].topology.atom(his_NE).name, his_CD, globals()[his_rotamer].topology.atom(his_CD).name))

                    globals()[his_rotamer].known_don = np.append(globals()[his_rotamer].known_don, his_CD)
                    print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(his_CD))

                    globals()[his_rotamer].known_acc = np.append(globals()[his_rotamer].known_acc, his_CE)
                    print('Atom {} just got added to known_acc list ONCE. No complementary H-bond interaction.'.format(his_CE))

                    # We remove the two atoms that were N from allpos_prot_don_acc list and we add to grease list
                    globals()[his_rotamer].allpos_prot_don_acc = np.delete(globals()[his_rotamer].allpos_prot_don_acc, np.where(globals()[his_rotamer].allpos_prot_don_acc == his_ND)[0][0])
                    globals()[his_rotamer].allpos_prot_don_acc = np.delete(globals()[his_rotamer].allpos_prot_don_acc, np.where(globals()[his_rotamer].allpos_prot_don_acc == his_NE)[0][0])
                    globals()[his_rotamer].grease = np.append(globals()[his_rotamer].grease, his_ND)
                    globals()[his_rotamer].grease = np.append(globals()[his_rotamer].grease, his_NE)

            elif his_rotamer[-5:-2] == 'HIP':
                print('Residue name before conversion is {}'.format(globals()[his_rotamer].topology.atom(his_ND).residue))
                globals()[his_rotamer].topology.atom(his_ND).residue.name = 'HIP'
                print('Residue name after conversion is {}'.format(globals()[his_rotamer].topology.atom(his_ND).residue))

                if his_rotamer[-2:] == '13':
                    print('No atom flipping required!')

                    globals()[his_rotamer].known_don = np.append(globals()[his_rotamer].known_don, his_ND)
                    print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(his_ND))

                    globals()[his_rotamer].known_don = np.append(globals()[his_rotamer].known_don, his_NE)
                    print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(his_NE))

                    globals()[his_rotamer].allpos_prot_don_acc = np.delete(globals()[his_rotamer].allpos_prot_don_acc, np.where(globals()[his_rotamer].allpos_prot_don_acc == his_CE)[0][0])
                    globals()[his_rotamer].allpos_prot_don_acc = np.delete(globals()[his_rotamer].allpos_prot_don_acc, np.where(globals()[his_rotamer].allpos_prot_don_acc == his_CD)[0][0])
                    globals()[his_rotamer].grease = np.append(globals()[his_rotamer].grease, his_CE)
                    globals()[his_rotamer].grease = np.append(globals()[his_rotamer].grease, his_CD)

                elif his_rotamer[-2:] == '24':
                    print('Atoms were:  {} {}, {} {}, {} {}, {} {}'.format(his_ND, globals()[his_rotamer].topology.atom(his_ND).name, his_CE, globals()[his_rotamer].topology.atom(his_CE).name, his_NE, globals()[his_rotamer].topology.atom(his_NE).name, his_CD, globals()[his_rotamer].topology.atom(his_CD).name))
                    print('Atom names were just flipped')
                    # We switch atom name and atom element from C to N and from N to C
                    temp1 = globals()[his_rotamer].topology.atom(his_ND).name
                    temp2 = globals()[his_rotamer].topology.atom(his_CE).name
                    temp3 = globals()[his_rotamer].topology.atom(his_NE).name
                    temp4 = globals()[his_rotamer].topology.atom(his_CD).name
                    globals()[his_rotamer].topology.atom(his_ND).name = temp4
                    globals()[his_rotamer].topology.atom(his_CE).name = temp3
                    globals()[his_rotamer].topology.atom(his_NE).name = temp2
                    globals()[his_rotamer].topology.atom(his_CD).name = temp1

                    temp1_element = globals()[his_rotamer].topology.atom(his_ND).element
                    temp2_element = globals()[his_rotamer].topology.atom(his_CE).element
                    temp3_element = globals()[his_rotamer].topology.atom(his_NE).element
                    temp4_element = globals()[his_rotamer].topology.atom(his_CD).element
                    globals()[his_rotamer].topology.atom(his_ND).element = temp4_element
                    globals()[his_rotamer].topology.atom(his_CE).element = temp3_element
                    globals()[his_rotamer].topology.atom(his_NE).element = temp2_element
                    globals()[his_rotamer].topology.atom(his_CD).element = temp1_element

                    # We append those atoms that switch from C to N and from N to C
                    globals()[his_rotamer].atm_name_change_to_N.append(his_CE)
                    globals()[his_rotamer].atm_name_change_to_N.append(his_CD)
                    globals()[his_rotamer].atm_name_change_to_C.append(his_ND)
                    globals()[his_rotamer].atm_name_change_to_C.append(his_NE)

                    print('Now, atoms are:  {} {}, {} {}, {} {}, {} {}'.format(his_ND, globals()[his_rotamer].topology.atom(his_ND).name, his_CE, globals()[his_rotamer].topology.atom(his_CE).name, his_NE, globals()[his_rotamer].topology.atom(his_NE).name, his_CD, globals()[his_rotamer].topology.atom(his_CD).name))

                    globals()[his_rotamer].known_don = np.append(globals()[his_rotamer].known_don, his_CD)
                    print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(his_CD))

                    globals()[his_rotamer].known_don = np.append(globals()[his_rotamer].known_don, his_CE)
                    print('Atom {} just got added to known_don list ONCE. No complementary H-bond interaction.'.format(his_CE))

                    # We remove the two atoms that were N from allpos_prot_don_acc list and we add to grease list
                    globals()[his_rotamer].allpos_prot_don_acc = np.delete(globals()[his_rotamer].allpos_prot_don_acc, np.where(globals()[his_rotamer].allpos_prot_don_acc == his_ND)[0][0])
                    globals()[his_rotamer].allpos_prot_don_acc = np.delete(globals()[his_rotamer].allpos_prot_don_acc, np.where(globals()[his_rotamer].allpos_prot_don_acc == his_NE)[0][0])
                    globals()[his_rotamer].grease = np.append(globals()[his_rotamer].grease, his_ND)
                    globals()[his_rotamer].grease = np.append(globals()[his_rotamer].grease, his_NE)

            # delete known residues from the unknown residues list
            globals()[his_rotamer].unknown_residues = np.delete(globals()[his_rotamer].unknown_residues, np.where(globals()[his_rotamer].unknown_residues == globals()[his_rotamer].topology.atom(his_ND).residue.index)[0][0])

        ##########################################################################


##############################################################


##############################################################
def protocol_of_branching(branch_name, outfile_prefix):
    print("branch_name is {}".format(branch_name.name))
    local_outfile_prefix = outfile_prefix
    if len(branch_name.unknown_residues) > 0:
        run_second_and_subsequent_passes_on_unknowns(branch_name)
        if len(branch_name.unknown_residues) > 0:
            print('{} still has unknown residues which will not get resolved unless branching takes place.'.format(branch_name))
            branching_unknown_residue(branch_name)  # branching the first residue in unknown_residues list (first come first serve)
            if len(branch_name.children_of_branch) == 0:
                protocol_of_branching(branch_name, local_outfile_prefix)
            else:
                ###########################################################################################
                # checking memory usage
                print("Memory usage:")
                memory_used_in_mb = round(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, 2)
                memory_used_in_gb = round(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3, 4)
                if memory_used_in_mb > 1000:
                    print("The process is taking {} GB right now.".format(memory_used_in_gb))
                else:
                    print("The process is taking {} MB right now.".format(memory_used_in_mb))
                ###########################################################################################

                for branch_object in branch_name.children_of_branch:
                    print("branch_object is {}".format(branch_object))
                    protocol_of_branching(globals()[branch_object], local_outfile_prefix)
        else:
            protocol_of_branching(branch_name, local_outfile_prefix)
    else:
        print("OUTPUTTING!!")
        checking_for_acid_dyad_branching(branch_name, branch_name.acid_dyad_branches, branch_name.acidic_residues_branching_by_atoms, local_outfile_prefix)

##############################################################
