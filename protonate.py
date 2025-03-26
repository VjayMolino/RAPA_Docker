# ===============================================================================
#          FILE: protonate.py
#
#         USAGE:
#
#   DESCRIPTION: protonate.py is the protonation state protein prep code developed by Mossa Ghattas in Prof. Thomas Kurtzman's lab at Lehman College CUNY
#                along with collaborations with Daniel Mckay and Anthony B. Cruz in Ventus Therapeutics. This code uses the necessary_functions.py as a module
#                that's imported at the end of the script and run.
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


from logging import raiseExceptions
import os
import sys
import argparse
from tkinter.tix import Tree
import numpy as np
import time
import mdtraj as md
import shutil
import operator
from collections import defaultdict
from difflib import SequenceMatcher
import itertools
import math
import necessary_functions
import timeit
import subprocess as subp
import shlex
import glob
import csv

os.chdir("/app/output") # change the directory to the dockerimages input/output directory
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, help="Input PDB file", required=True)
parser.add_argument("-o", "--output", type=str, help="Output prefix", required=True)
parser.add_argument("-c","--cutoff", type=float, help="Energy cutoff, default=2.0", required=False )
parser.add_argument("-u", "--use", type=str, help="HETATM file", required=False)
args = parser.parse_args()
inifile = args.input
out_prefix = args.output
het_atms_to_be_used = args.use
if args.cutoff:
    degenerate_states_e_cutoff = args.cutoff
else:
    degenerate_states_e_cutoff = 2.0

necessary_functions.degenerate_states_e_cutoff = degenerate_states_e_cutoff

# Using colors from https://stackoverflow.com/questions/39473297/how-do-i-print-colored-output-with-python-3
if not os.path.exists(inifile):
    sys.stderr.write("\n\033[1;31;48mRuntimeError\033[1;37;0m: {0} is not accessible. Verify that {0} is in the current directory\n".format(inifile))
    sys.exit()

if args.use is not None:
    if not os.path.exists(het_atms_to_be_used):
        sys.stderr.write("\n\033[1;31;48mRuntimeError\033[1;37;0m: {0} is not accessible. Verify that {0} is in the current directory\n".format(het_atms_to_be_used))
        sys.exit()

#######################################################################################################

initial_time = timeit.default_timer()

parent_directory = os.path.abspath(os.getcwd())
target_directory = inifile.split('.pdb')[0]
target_directory_path = os.path.join(parent_directory, target_directory)
#base_path = os.path.split(os.path.abspath(__file__))[0]
base_path = "/app/rapa/" #base path containing the src in the docker image
reduce_hetatm_db_filepath = os.path.join(base_path, "reduce_wwPDB_het_dict.txt")
PES_data_filepath = os.path.join(base_path, "OOdist_OHOangle_E_smooth_interpolate.csv")

if os.path.exists(target_directory_path):
    shutil.rmtree(target_directory_path)
os.mkdir(target_directory_path)

# Copy inifile file using subprocess
copy_command = "cp {} {}".format(inifile, target_directory_path)
copy_command_line = shlex.split(copy_command)
copy_proc = subp.Popen(copy_command_line)
copy_proc_ex_code = copy_proc.wait()

# Copy hetatms file using subprocess
if args.use is not None:
    copy_command = "cp {} {}".format(het_atms_to_be_used, target_directory_path)
    copy_command_line = shlex.split(copy_command)
    copy_proc = subp.Popen(copy_command_line)
    copy_proc_ex_code = copy_proc.wait()

os.chdir(target_directory_path)

# Check out_prefix is different
out_prefix_check = out_prefix.strip()
if out_prefix_check == target_directory:
    out_prefix = "{}_PREP".format(out_prefix_check)
    sys.stdout.write("\n\033[1;33;48mRuntimeWarning\033[1;37;0m: Output prefix ({}) should be different than the pdb basename ({}).\nChanging the output prefix to {}.\n\n".format(out_prefix_check, target_directory, out_prefix))
    time.sleep(2)

# # Warn the user to make sure that there are no missing heavy atoms of this protein and that there should be 'TER' line between different protein monomers/units
# repeat_question = True
# while repeat_question:
#     print("Before the analysis begins:")
#     print("It's the user's responsibility to make sure there are no missing heavy atoms from the protein structure.")
#     print("It's also very important that 'TER' lines are separating between different protein chains and ions/solvents/ligands.")
#     user_decision = input("With that being said, would you like to proceed with the analysis? (y/n)")
#     if user_decision.lower() not in ('y', 'n'):
#         print("Please, respond to the following question with 'y' or 'n'")
#         repeat_question = True
#     else:
#         repeat_question = False
#         if user_decision == 'y':
#             pass
#         elif user_decision == 'n':
#             print("Terminating the analysis...")
#             sys.exit()

# # for making a log file of the analysis, uncomment those next three lines
# name_of_log_file = inifile.split('.pdb')[0] + '_protein_prep.log'
# log = open(name_of_log_file, "a")
# sys.stdout = log

print('#########################################################################################################################################')
print('Analyzing the input structure to only keep the first alternate location it encounters for residues with more than one alternate location')
print('#########################################################################################################################################')

with open(inifile, "r") as file:
    in_file = file.readlines()
    atom_lines_of_multi_locations = []
    residues_seqnmbr_of_multi_locations = []
    for line in in_file:
        if line[0:4] == 'ATOM' or line[0:6] == 'HETATM':
            if line[16:17] != ' ':
                atom_lines_of_multi_locations.append(line)
                if int(line[22:26]) not in residues_seqnmbr_of_multi_locations:
                    residues_seqnmbr_of_multi_locations.append(int(line[22:26]))
#######################################################################################################

sentence_list = atom_lines_of_multi_locations
result = []  # grouping atoms lines by chain and residue sequence number.
for sentence in sentence_list:
    if len(result) == 0:
        result.append([sentence])
    else:
        for i in range(0, len(result)):
            score = SequenceMatcher(None, sentence[21:26], result[i][0][21:26]).ratio()  # grouping atoms lines by residue sequence number.
            if score != 1:
                if i == len(result) - 1:
                    result.append([sentence])
            else:
                if score == 1:
                    result[i].append(sentence)
                    break

atom_lines_of_multi_locations_grouped_by_chain_and_resseq = result
#######################################################################################################

# alternate_location_residues_decisions = defaultdict(dict)
#
# for i in atom_lines_of_multi_locations_grouped_by_chain_and_resseq:
#     possible_alternate_location_indicators_and_occupancies = defaultdict(dict)
#
#     for j in i:
#         alternate_location_indicator_and_resname = str(j[16:20])
#         occupancy = float(j[54:60])
#         if alternate_location_indicator_and_resname not in possible_alternate_location_indicators_and_occupancies.keys():
#             possible_alternate_location_indicators_and_occupancies[alternate_location_indicator_and_resname] = occupancy
#
#     print('Residue {} (residue sequence number {}) has {} alternate locations.'.format(str(i[0][17:20]), int(i[0][22:26]), len(possible_alternate_location_indicators_and_occupancies.keys())))
#
#     atom_total_occupancies = sum(possible_alternate_location_indicators_and_occupancies.values())
#     possible_alternate_location_indicators_and_occupancies_ordered = sorted(possible_alternate_location_indicators_and_occupancies.items(), key=lambda x: x[1], reverse=True)
#
#     print('The ordered alternate locations based on occupancy are {}.'.format(possible_alternate_location_indicators_and_occupancies_ordered))
#     print('The {} alternate locations of this residue\'s atoms have occupancy totaling to {}'.format(len(possible_alternate_location_indicators_and_occupancies.keys()), atom_total_occupancies))
#
#     if atom_total_occupancies >= 0.95:
#         print('The total occupancy is >= 0.95 .......')
#     else:
#         print('The total occupancy DOES NOT add to 0.95 or more.......')
#         print('       This might mean that there are some alternate locations that are left unknown and not added in the crystal structure.')
#
#     if len(possible_alternate_location_indicators_and_occupancies_ordered) == 1:
#         # only one alternate location is present
#         print('There is only one alternate location present for this residue, we will consider this alternate location')
#         alternate_location_residues_decisions['chain' + str(i[0][21:22]) + '_res' + str(int(i[0][22:26]))] = [possible_alternate_location_indicators_and_occupancies_ordered[0][0]]
#     elif len(possible_alternate_location_indicators_and_occupancies_ordered) != 1:
#         if abs(possible_alternate_location_indicators_and_occupancies_ordered[0][1] - possible_alternate_location_indicators_and_occupancies_ordered[1][1]) >= 0.20:
#             # no split, consider the alternate location with the higher occupancy only
#             print('Since the difference in occupancy between the highest two alternate locations in occupancy is equal or more than 0.20, we will only consider the alternate location with the highest occupancy')
#             alternate_location_residues_decisions['chain'+str(i[0][21:22])+'_res'+str(int(i[0][22:26]))] = [possible_alternate_location_indicators_and_occupancies_ordered[0][0]]
#         else:
#             # split between the two alternate locations
#             print('Since the difference in occupancy between the highest two alternate locations in occupancy is less than 0.20, we will consider all alternate locations of this residue.')
#             alternate_location_residues_decisions['chain'+str(i[0][21:22])+'_res'+str(int(i[0][22:26]))] = list(map(operator.itemgetter(0), possible_alternate_location_indicators_and_occupancies_ordered))  # makes a list of first item in each sublist
#
#     print('################################################################')

#######################################################################################################
# This can be used to replace the previous block of code. This can be used such that only the first alternate location encountered gets stored as the residue's location
alternate_location_residues_decisions = defaultdict(dict)

for i in atom_lines_of_multi_locations_grouped_by_chain_and_resseq:
    possible_alternate_location_indicators_and_occupancies = defaultdict(dict)

    list_of_alt_loc_resname = []
    for j in i:
        alternate_location_indicator_and_resname = str(j[16:20])
        list_of_alt_loc_resname.append(alternate_location_indicator_and_resname)

    print(list_of_alt_loc_resname)
    alt_loc_decision = list_of_alt_loc_resname[0]
    print('Residue {} (residue sequence number {}) will have only 1 alternate locations (whichever came first): {}.'.format(str(i[0][17:20]), int(i[0][22:26]), alt_loc_decision))

    alternate_location_residues_decisions['chain' + str(i[0][21:22]) + '_res' + str(int(i[0][22:26]))] = [alt_loc_decision]
    print('################################################################')

#######################################################################################################

print('################################################################################')

print('alternate_location_residues_decisions is {}.'.format(list(alternate_location_residues_decisions.items())))
branching_configurations_of_residues_with_alternate_locations = list(itertools.product(*list(alternate_location_residues_decisions.values())))
print('There will be {} pdb output files that represent the branching of the initial pdb structure based on the alternate location residues decisions'.format(len(branching_configurations_of_residues_with_alternate_locations)))
print('branching_configurations_of_residues_with_alternate_locations is {}'.format(branching_configurations_of_residues_with_alternate_locations))
print('Outputting initial structure branches based on decision made on alternate location branching:')

init_structure_branch_output_files = []
for branch_index, multi_occupancy_residues_structure_branch in enumerate(branching_configurations_of_residues_with_alternate_locations):
    initial_occupancy_branchings_time = timeit.default_timer()

    target_occu_branch_directory_path = os.path.join(target_directory_path, inifile.split('.pdb')[0] + "_" + str(branch_index + 1))
    os.mkdir(target_occu_branch_directory_path)

    # Copy inifile file using subprocess
    copy_command = "cp {} {}".format(inifile, target_occu_branch_directory_path)
    copy_command_line = shlex.split(copy_command)
    copy_proc = subp.Popen(copy_command_line)
    copy_proc_ex_code = copy_proc.wait()

    # Copy hetatms file using subprocess
    if args.use is not None:
        copy_command = "cp {} {}".format(het_atms_to_be_used, target_occu_branch_directory_path)
        copy_command_line = shlex.split(copy_command)
        copy_proc = subp.Popen(copy_command_line)
        copy_proc_ex_code = copy_proc.wait()

    os.chdir(target_occu_branch_directory_path)

    structure_branch_output = inifile.split('.pdb')[0] + '_' + str(branch_index+1) + '.pdb'
    structure_branch_output_mdtraj_Ready = inifile.split('.pdb')[0] + '_' + str(branch_index+1) + '_mdtraj_ready.pdb'
    structure_branch_output_REDUCE_Ready = inifile.split('.pdb')[0] + '_' + str(branch_index+1) + '_REDUCE_ready.pdb'

    init_structure_branch_output_files.append(structure_branch_output)
    print('Names of output files of initial structure Branch {} is:'.format(branch_index+1))
    print('       1) "{}" is the file that represents a branch of the initial structure. It also fixes atom indices and removes CONECT lines.'.format(structure_branch_output))
    print('       2) "{}" is the file that will be used for the analysis as the input structure. '
          '                 It\'s the same file as file (1) but residue sequence column is fixed.'.format(structure_branch_output_mdtraj_Ready))
    print('       3) "{}" is the same file as 2) except that we have removed TER lines and renumbered the residue seq nmbr'
          '                 such that its continuous in case there is more than one chain in the structure. This file will be used by REDUCE shortly.'.format(structure_branch_output_REDUCE_Ready))

    branching_configurations_of_residues_with_alternate_locations_dict = defaultdict(dict)
    for i in range(0, len(multi_occupancy_residues_structure_branch)):
        branching_configurations_of_residues_with_alternate_locations_dict[list(alternate_location_residues_decisions.keys())[i]] = list(multi_occupancy_residues_structure_branch)[i]

    ################################################################################################################################
    # Outputting file (1). This file was edited such that atom indices are fixed, CONECT lines were removed
    ################################################################################################################################
    with open(inifile, "r") as file:
        in_file = file.readlines()
        with open(structure_branch_output, "w") as outfile:
            atm_ndx = 1
            for line in in_file:
                if line[0:6] == 'CONECT':
                    continue
                elif line[0:4] != 'ATOM' and line[0:6] != 'HETATM':
                    outfile.write(line)
                else:
                    alternate_location_indicator = line[16:17]
                    chain = str(line[21:22])
                    residue_seq_nmbr = int(line[22:26])
                    L1 = line[0:6]
                    # atm_ndx is taking 5 indices locations after L1 which means it takes cells [6:11] ~ {:5d} just means 5 consecutive indices
                    L2 = line[12:16]
                    L3 = line[17:78]

                    if alternate_location_indicator == ' ':
                        if L3.endswith("\n"):
                            outfile.write('{}{:5d} {} {}'.format(L1, atm_ndx, L2, L3))
                        else:
                            outfile.write('{}{:5d} {} {}\n'.format(L1, atm_ndx, L2, L3))
                        atm_ndx += 1
                    else:
                        line_chain_and_resseqnmbr = 'chain' + chain + '_res' + str(residue_seq_nmbr)
                        if line_chain_and_resseqnmbr in list(branching_configurations_of_residues_with_alternate_locations_dict.keys()):
                            winning_alternate_location_indicator = branching_configurations_of_residues_with_alternate_locations_dict.get(line_chain_and_resseqnmbr)[0]
                            if alternate_location_indicator == winning_alternate_location_indicator:  # if the atom's line alt loc indicator is the same as the winning alt loc indicator, we will print the line out
                                if L3.endswith("\n"):
                                    outfile.write('{}{:5d} {} {}'.format(L1, atm_ndx, L2, L3))
                                else:
                                    outfile.write('{}{:5d} {} {}\n'.format(L1, atm_ndx, L2, L3))
                                atm_ndx += 1
                        else:
                            continue
    ################################################################################################################################

    ################################################################################################################################
    # Outputting file (2). This file was edited from file (1) such that it fixes the residue sequence number
    ################################################################################################################################
    with open(structure_branch_output, "r") as file:
        in_file = file.readlines()
        with open(structure_branch_output_mdtraj_Ready, "w") as outfile:
            previously_assigned_residue_seq_nmbr = 0
            newly_assigned_residue_seq_nmbr = 0
            previous_code_insertion_residues = ' '
            for line in in_file:
                if line[0:3] == 'END':
                    continue
                elif line[0:3] == 'TER':
                    L1 = line[0:22]

                    L2 = line[27:78]
                    #outfile.write('{}{:4d}{}\n'.format(L1, newly_assigned_residue_seq_nmbr, L2))
                    outfile.write(line)
                    previously_assigned_residue_seq_nmbr = 0
                    newly_assigned_residue_seq_nmbr = 0
                elif line[0:4] != 'ATOM' and line[0:6] != 'HETATM':
                    outfile.write(line)
                elif line[0:4] == 'ATOM' or line[0:6] == 'HETATM':

                    L1 = line[0:22]
                    # newly_assigned_residue_seq_nmbr is taking 4 indices locations after L1 which means it takes cells [22:26] ~ {:4d} just means 4 consecutive indices
                    code_insertion_residues = ' '
                    L2 = line[27:78]

                    if int(line[22:26]) != previously_assigned_residue_seq_nmbr or line[26:27] != previous_code_insertion_residues:
                        newly_assigned_residue_seq_nmbr += 1
                        previously_assigned_residue_seq_nmbr = int(line[22:26])
                        previous_code_insertion_residues = line[26:27]

                    if L2.endswith("\n"):
                        outfile.write('{}{:4d}{}{}'.format(L1, newly_assigned_residue_seq_nmbr, code_insertion_residues, L2))
                    else:
                        outfile.write('{}{:4d}{}{}\n'.format(L1, newly_assigned_residue_seq_nmbr, code_insertion_residues, L2))
    ################################################################################################################################

    ################################################################################################################################
    # Outputting file (3). This file was edited such that atom indices are fixed, CONECT lines were removed, TER lines were removed, residue sequence number are continuous. REDUCE breaks if the res seq nmbr are not continuous and TER lines exist.
    ################################################################################################################################
    with open(structure_branch_output_mdtraj_Ready, "r") as file:
        in_file = file.readlines()
        with open(structure_branch_output_REDUCE_Ready, "w") as outfile:
            atm_ndx = 1
            residue_seq_nmbr_of_previous_residue = 0
            residues_count_of_all_chains = 0
            for line in in_file:
                if line[0:3] == 'TER':
                    continue
                elif line[0:6] == 'CONECT':
                    continue
                elif line[0:4] != 'ATOM' and line[0:6] != 'HETATM':
                    outfile.write(line)
                else:
                    alternate_location_indicator = line[16:17]
                    chain = str(line[21:22])

                    L1 = line[0:6]
                    L2 = line[12:16]
                    L3 = line[17:22]

                    residue_given_seq_nmbr_in_line = int(line[22:26])

                    if residue_given_seq_nmbr_in_line != residue_seq_nmbr_of_previous_residue:
                        residues_count_of_all_chains += 1
                        residue_seq_nmbr_new_assigned = residues_count_of_all_chains
                        residue_seq_nmbr_of_previous_residue = residue_given_seq_nmbr_in_line
                    else:
                        residue_seq_nmbr_new_assigned = residues_count_of_all_chains
                        residue_seq_nmbr_of_previous_residue = residue_given_seq_nmbr_in_line

                    L4 = line[26:78]

                    if alternate_location_indicator == ' ':
                        if L4.endswith("\n"):
                            outfile.write('{}{:5d} {} {}{:4d}{}'.format(L1, atm_ndx, L2, L3, residue_seq_nmbr_new_assigned, L4))
                        else:
                            outfile.write('{}{:5d} {} {}{:4d}{}\n'.format(L1, atm_ndx, L2, L3, residue_seq_nmbr_new_assigned, L4))
                        atm_ndx += 1
                    else:
                        line_chain_and_resseqnmbr = 'chain'+chain+'_res'+str(residue_given_seq_nmbr_in_line)
                        if line_chain_and_resseqnmbr in list(branching_configurations_of_residues_with_alternate_locations_dict.keys()):
                            winning_alternate_location_indicator = branching_configurations_of_residues_with_alternate_locations_dict.get(line_chain_and_resseqnmbr)[0]
                            if alternate_location_indicator == winning_alternate_location_indicator:  # if the atom's line alt loc indicator is the same as the winning alt loc indicator, we will print the line out
                                if L4.endswith("\n"):
                                    outfile.write('{}{:5d} {} {}{:4d}{}'.format(L1, atm_ndx, L2, L3, residue_seq_nmbr_new_assigned, L4))
                                else:
                                    outfile.write('{}{:5d} {} {}{:4d}{}\n'.format(L1, atm_ndx, L2, L3, residue_seq_nmbr_new_assigned, L4))
                                atm_ndx += 1
                        else:
                            continue
    ################################################################################################################################
    print('################################################################################')

    ################################################################################################################################
    # Here, there are two files produced:
    # the first file is the raw REDUCE output file produced by REDUCE. It has the same name as the input file ("......_REDUCE_ready.pdb") except it has ".h" added at the end of the name of the file
    # the second file is the "processed" file of the raw REDUCE output. In it, we fixed the missing chain ID and atom serial number
    ################################################################################################################################

    # generate output filename
    REDUCE_protonated_file = structure_branch_output_REDUCE_Ready+'.h'

    # use reduce
    reduce_command = "reduce -NOADJust -NOFLIP {}".format(structure_branch_output_REDUCE_Ready)
    reduce_command_line = shlex.split(reduce_command)
    with open(REDUCE_protonated_file, "w") as reduce_out:
        reduce_proc = subp.Popen(reduce_command_line, stdout=reduce_out)
    reduce_proc_ex_code = reduce_proc.wait()

    print('Executed in terminal: {}'.format(reduce_command))
    print('This command produced {}: The file that has the all atoms plus all Hydrogen atoms added by REDUCE'.format(REDUCE_protonated_file))

    fixed_atm_SN_chain_ID_of_REDUCE_protonated_file = 'fixed_atm_SN_chain_ID_REDUCE_output_of_'+structure_branch_output

    with open(REDUCE_protonated_file, "r") as file:
        in_file = file.readlines()
        with open(fixed_atm_SN_chain_ID_of_REDUCE_protonated_file, "w") as outfile:
            atm_ndx = 1
            for line in in_file:
                if line[0:3] == 'TER':
                    #outfile.write(line)
                    pass
                elif line[0:4] == 'ATOM' or line[0:6] == 'HETATM':
                    L1 = line[:6]
                    L2 = line[12:20]
                    if line[13] != 'H' and line[13:16] != 'OXT':
                        chain = line[21:22]
                    L3 = line[22:79]

                    if L3.endswith("\n"):
                        outfile.write('{}{:5d} {} {:1s}{}'.format(L1, atm_ndx, L2, chain, L3))
                    else:
                        outfile.write('{}{:5d} {} {:1s}{}\n'.format(L1, atm_ndx, L2, chain, L3))

                    atm_ndx += 1
                else:
                    outfile.write(line)

    print('Additionally, {} is the fixed atm SN and chain ID REDUCE output file'.format(fixed_atm_SN_chain_ID_of_REDUCE_protonated_file))
    ################################################################################################################################

    class MotherOfAll:
        def __init__(self):
            pass


    motherstructure = MotherOfAll()
    motherstructure.name = structure_branch_output.split('.pdb')[0]
    motherstructure.path = [motherstructure.name]

    # Load the structure and generate the topology
    motherstructure.structure = md.load_pdb(structure_branch_output_mdtraj_Ready)  # This is the pdb input
    motherstructure.topology = motherstructure.structure.topology
    motherstructure.xyz = motherstructure.structure.xyz

    # Load the protonated form of the structure and generate the topology.
    motherstructure.h_structure = md.load_pdb(fixed_atm_SN_chain_ID_of_REDUCE_protonated_file)  # This is the pdb input with lower occupancy atoms removed and with hydrogen atoms added by REDUCE and atm SN and chain ID fixed to be used in Histidine block analysis
    motherstructure.h_topology = motherstructure.h_structure.topology
    motherstructure.h_xyz = motherstructure.h_structure.xyz

    ################################################################################################################################

    ###########################################################################
    # all_n_o_elements list will have all N and O atoms in the whole proteins including ligands, non-standard residues. WATER ISNT included
    motherstructure.all_n_o_elements = motherstructure.topology.select('element N O and not water')
    ###########################################################################

    ###########################################################################
    # Using 'element N O and protein' in the *.topology.select (mdtraj atom selection function) is not accurate. It misses some ASH/GLH residues and it misses all other non-standard residues/ligands.
    # Dan wants the user to be informed of all non-standard residues/ligands side chain atoms and wants the code to output all those O and N atoms
    # Therefore, we will do this by specifying atom name specifically. We will need to check if there are ligand atoms with those atom names, in the case there is, we can give the user the options to input them.
    # allpos_prot_don_acc list will contain O and N atoms in just standard amino acids (no ligands, no water, no non-standard residues):
    # backbone N and O [N, O] (except proline? cuz it cant donate - no H)
    # ASP/ASH [OD1, OD2]
    # GLU/GLH [OE1, OE2]
    # ASN [OD1, ND2]
    # GLN [OE1, NE2]
    # ARG [NE, NH1, NH2]
    # TRP [NE1]
    # LYS [NZ]
    # HIS/HIP/HIE/HID [ND1, NE2]
    # SER [OG]
    # THR [OG1]
    # TYR [OH]
    # HIS/HIP/HIE/HID [CE1, CD2]  # CE1 CD2 atoms of all HIS residues (and its ionization states) will be potential donors as they could be flipped.
    motherstructure.allpos_prot_don_acc = motherstructure.topology.select('name N O and not water')  # all backbone atoms of residues (including non-standard residues backbone)
    motherstructure.allpos_prot_don_acc = np.append(motherstructure.allpos_prot_don_acc, motherstructure.topology.select('name OD1 OD2 OE1 OE2 ND2 NE2 NE NH1 NH2 NE1 NZ ND1 OG OG1 OH and resname ASP ASH GLU GLH ARG ARN TRP LYS LYN CYS CYX ASN GLN SER THR TYR HIS HIE HID HIP'))
    if len(motherstructure.topology.select('resname HIS HIE HID HIP and name CE1 CD2')) != 0:  # this ensures that it wont add if it's empty. empty array would be floats and would change the atom indices into floats which will break the analysis
        motherstructure.allpos_prot_don_acc = np.sort(np.append(motherstructure.allpos_prot_don_acc, motherstructure.topology.select('resname HIS HIE HID HIP and name CE1 CD2')))
    ###########################################################################

    ###########################################################################
    # Here, we see what N/O atoms that are in the structure file and not in the allpos_prot_don_acc list.
    # Then, we will inform the user of the N and O elements that could be known donors or acceptors but aren't used in the analysis.
    # we will also output a .txt file at the end of those hetero atoms so that the user can choose to use as a parameter if he chooses to go for another run of the code
    motherstructure.o_n_atoms_of_nonstd_residues_or_ligands = np.empty((0,), dtype=int)
    for i in motherstructure.all_n_o_elements:
        if i not in motherstructure.allpos_prot_don_acc:
            motherstructure.o_n_atoms_of_nonstd_residues_or_ligands = np.append(motherstructure.o_n_atoms_of_nonstd_residues_or_ligands, i)
    ###########################################################################

    ###########################################################################
    # Here we make sure we have a dictionary of HETATMs that have H atoms attached to it using the database provided in reduce github. we exclude hydrogens which even though they are connected to heavy N or O atom, they start their own line with xxCONECT. those H are commented with xx by reduce because those H are not physiological.
    with open(reduce_hetatm_db_filepath, "r") as het_atm_db_file:
        het_atm_db_lines = het_atm_db_file.read().splitlines()

        # creating dictionary with rename as the key and list of all CONECT atom names (excluding xxCONECT)
        res_dict = {}
        for line_ndx, line in enumerate(het_atm_db_lines):
            if len(line) != 0:
                line_list = line.split()
                if line_list[0] == 'RESIDUE':
                    resname = line_list[1]
                    res_atm_list = []
                elif line_list[0] == 'CONECT':
                    res_atm_list.append(line_list[1])
                elif line_list[0] == 'END':
                    res_dict[resname] = res_atm_list

        # Here, we create the dictionary which will have keys as "resname_O/Nname" format with values of the hydrogens attached to it (excluding xxCONECT atoms)
        het_atm_dict = {}
        for line in het_atm_db_lines:
            if len(line) != 0:
                line_list = line.split()
                if line_list[0] == 'RESIDUE':
                    resname = line_list[1]
                elif line_list[0] == 'CONECT':
                    if line_list[1][0] == 'O' or line_list[1][0] == 'N':
                        o_n_atom_name = line_list[1]
                        bondsnum = int(line[19])
                        bonded_atoms = line_list[-bondsnum:]

                        if len(line_list) == bondsnum + 2:  # this means that index 20 was not empty space because the bonded atom had a long name.
                            bonded_atoms[0] = bonded_atoms[0][1:]  # Here, I just got rid of the first index of the first bonded atom because it was concatenated because the atom name is long
                        elif len(line_list) != bondsnum + 3 and len(line_list) != bondsnum + 2:
                            print('The line syntax is weirdly put')
                            print('{}'.format(line))
                            print('################################################################################')
                            continue

                        H_bonded_atoms = []
                        for bonded in bonded_atoms:
                            if bonded[0] == 'H' and bonded in res_dict[resname]:
                                H_bonded_atoms.append(bonded)

                        if len(H_bonded_atoms) != 0:
                            het_atm_dict[resname + '_' + o_n_atom_name] = H_bonded_atoms
        motherstructure.het_atm_dict = het_atm_dict
    ###########################################################################

    ###########################################################################
    # OUTPUTTING a file that contains all the Hetatm the script recognized, which the script will not use in this run
    # we won't output the file with the hetatms of this structure if the user already inputted this file as an argument for the script or there arent any hetatms seen
    if len(motherstructure.o_n_atoms_of_nonstd_residues_or_ligands) != 0 and args.use is None:
        print('################################################################################')
        print("The tool identified O/N atoms of ligands or non-standard residues.\nThose atoms will not be used in this analysis run.")
        print("At the end of this run, a .txt file containing all those atoms will be outputted.")
        print("It's the user's responsibility to edit the file to confirm if the atom can indeed be 'ACC', or 'DON' and how many possible donors/acceptors it can be.")
        print("More instructions on the usage of this file will be in the REMARKS header of the file")
        het_atms_file_name = out_prefix + '_het_atms.txt'
        with open(het_atms_file_name, "w") as het_atms_file:
            het_atms_file.write("REMARKS:\nThis file has the HETATM (O and N atoms) of any non-standard residues and ligands found in this structure.\n"
                                "The tool will not use those atoms in this run. If the user wishes to use those atoms in another run, the .txt file will need to be added to the command as a parameter.\n"
                                "To use those atoms as potential acceptors and/or donors in the analysis, you will need manually examine these atoms below to decide which atom is 'ACC', 'DON', and how many it can act as 'DON' or 'ACC'\n"
                                "FOR DONORS: The tool can already conclude how many possible donors on the heavy atom because it scans the bond connectivity\n"
                                "            and finds out how many H atoms are attached to it.\n"
                                "            The number at the end of the line will indicate the number of times the heavy atom can donate an H-bond.\n"
                                "FOR ACCEPTORS: The tool will put 1 as the number of possible acceptors as the default.\n"
                                "            The user has to either delete that line if they see that atom can not act as an acceptor,\n"
                                "            or replace the '1' with number of possible acceptors on this atom that can act as known acc\n"
                                "If the user sees an atom that is in the .txt file that shouldn't be used in the run, they should delete the line from the text file\n\n"
                                "Format of the het atom lines:\n"
                                "'HETATM     atom_index     atm_name     residue_name     H_bond_feature     number_of_feature'\n\n")

            for i in motherstructure.o_n_atoms_of_nonstd_residues_or_ligands:
                het_atm_name = motherstructure.topology.atom(i).name
                het_atm_index = motherstructure.topology.atom(i).index
                het_atm_residue_name = motherstructure.topology.atom(i).residue.name
                het_atm_residue_index = motherstructure.topology.atom(i).residue.index
                het_dict_key_name_check = het_atm_residue_name + '_' + het_atm_name

                # the hetatm being an acceptor
                number_of_acceptors = 1  # As default we will consider one acceptor. user will have to change this number inside the .txt file to indicate how many acceptors can accept H-bonds
                het_atms_file.write('HETATM     {:5d}     {:^5s}     {:^5s}     ACC     {:2d}\n'.format(het_atm_index, het_atm_name, het_atm_residue_name, number_of_acceptors))

                # the hetatm being an donor
                if het_dict_key_name_check in het_atm_dict:  # remember het_atm_dict only has atms that has h atoms. this if condition just means that it has H attached to it in the reduce database
                    number_of_donors = 0
                    het_atm_h_atms = het_atm_dict[het_dict_key_name_check]
                    for j in het_atm_h_atms:
                        het_atm_h_check = motherstructure.h_topology.select('name "{}" and resid {}'.format(j, het_atm_residue_index))  # using "" for the name is SO important otherwise the script will crash if the atom name has prime as part of the name such as seen in ADP residues
                        if len(necessary_functions.flatten(het_atm_h_check)) == 1:
                            number_of_donors += 1

                    if number_of_donors != 0:
                        het_atms_file.write('HETATM     {:5d}     {:^5s}     {:^5s}     DON     {:2d}\n'.format(het_atm_index, het_atm_name, het_atm_residue_name, number_of_donors))
                    else:
                        het_atms_file.write('HETATM     {:5d}     {:^5s}     {:^5s}     DON     {:2d}     #reduce_didnt_add_H\n'.format(het_atm_index, het_atm_name, het_atm_residue_name, number_of_donors))

        print("'{}' is the file that contains the HETATM of non-standard residues/ligands found in the structure.".format(het_atms_file_name))
        print('################################################################################')
    ###########################################################################

    ###########################################################################
    # Storing to memory the PES data points
    motherstructure.OO_distances = [str(round(i, 1)) for i in np.arange(2.5, 5.1, 0.1)]
    motherstructure.OHO_angles = [str(round(i, 1)) for i in np.arange(90, 180.1, 0.1)]

    motherstructure.PES_energies = []
    with open(PES_data_filepath, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        csv_list = list(csv_reader)
        for csv_row_indx, csv_row in enumerate(csv_list):
            if csv_row_indx != 0:
                motherstructure.PES_energies.append(csv_row[4])
    ###########################################################################

    ################################################################################################################################

    # Identifying all grease side chain C atoms
    motherstructure.grease = motherstructure.topology.select('element C and not backbone and resname GLY ALA VAL LEU ILE PHE PRO TRP MET TYR')

    # Identifying all metal cations
    motherstructure.cations = motherstructure.topology.select('element Li Na K Rb Cs Fr Be Mg Ca Sr Ba Ra Sc Ti Cr Mn Fe Co Ni Cu Zn Mo Yb')

    # Identifying all 'unknown_residues' of this protein.
    # We don't include residues such as 'GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'PRO', 'TRP', 'MET', 'ACE', 'NME' as 'unknown'
    # because there are no ionization states for them, they are always known.
    motherstructure.unknown_residues = np.empty((0,), dtype=int)
    for i in motherstructure.allpos_prot_don_acc:
        if motherstructure.topology.atom(i).residue.index not in motherstructure.unknown_residues and motherstructure.topology.atom(i).residue.name in ['ASN', 'GLN', 'SER', 'THR', 'TYR', 'HIS', 'HIE', 'HID', 'HIP']:  # I think its best not to make those residues ('ASP', 'ASH', 'GLU', 'GLH', 'ARG', 'ARN', 'LYS', 'LYN', 'CYS', 'CYX') unknown to begin with. thats because if one is missing side chain heavy atoms, the script will not exit because it wont make it known thus we want to avoid seeing them as remain unknown in case the residue has missing sidechains
            motherstructure.unknown_residues = np.append(motherstructure.unknown_residues, motherstructure.topology.atom(i).residue.index)

    ################################################################################################################################
    # Generating known don & acc (these lists have available donor and acceptor atoms to set other residues and interactions.)
    motherstructure.known_don = np.empty((0,), dtype=int)
    motherstructure.known_acc = np.empty((0,), dtype=int)

    for i in motherstructure.allpos_prot_don_acc:
        atom_name = motherstructure.topology.atom(i).name
        residue_name = motherstructure.topology.atom(i).residue.name
        if atom_name == 'N' and residue_name != 'PRO':  # backbone N (acts as one donor). Proline backbone N doesnt have an H on it
            motherstructure.known_don = np.append(motherstructure.known_don, i)
        elif atom_name == 'O':  # backbone O (acts as two acceptors)
            motherstructure.known_acc = np.append(motherstructure.known_acc, i)
            motherstructure.known_acc = np.append(motherstructure.known_acc, i)
        elif atom_name == 'OXT':  # C-terminus OXT atom
            motherstructure.known_acc = np.append(motherstructure.known_acc, i)
            motherstructure.known_acc = np.append(motherstructure.known_acc, i)
        elif atom_name == 'NE1' and residue_name == 'TRP':  # TRP(NE1) is always one donor. The lone pair on NE1 is fully delocalized.
            motherstructure.known_don = np.append(motherstructure.known_don, i)
        elif atom_name == 'NE' and residue_name in ['ARG', 'ARN']:  # ARG(NE) is always one donor. The lone pair on NE is fully delocalized.
            motherstructure.known_don = np.append(motherstructure.known_don, i)

    # adding those hetatms of non-standard residues and ligands into allpos_prot_don_acc, known_don, and known_acc lists
    if args.use is not None:
        print("################################################################################")
        print("\nReading the atoms that are input in the parameter file {}".format(het_atms_to_be_used))
        with open(het_atms_to_be_used, "r") as het_atms_to_be_used_file:
            het_atms_to_be_used_lines = het_atms_to_be_used_file.read().splitlines()
            for line in het_atms_to_be_used_lines:
                if len(line) != 0:
                    line_list = line.split()
                    if line_list[0] == 'HETATM':
                        print(line_list)
                        het_atm_index = int(line_list[1])
                        het_atm_feature = line_list[4]
                        het_atm_full_name = motherstructure.topology.atom(het_atm_index)
                        het_atm_feature_count = int(line_list[5])

                        if het_atm_feature == 'DON':
                            for i in range(het_atm_feature_count):
                                motherstructure.known_don = np.append(motherstructure.known_don, het_atm_index)
                                if het_atm_index not in motherstructure.allpos_prot_don_acc:  # This makes sure I dont append it more than once since it can be written in the file twice for having possibility of being a donor and acceptor
                                    motherstructure.allpos_prot_don_acc = np.append(motherstructure.allpos_prot_don_acc, het_atm_index)

                            print('Atom {} ({}) is added to known_don list {} time(s)'.format(het_atm_index, het_atm_full_name, het_atm_feature_count))
                        elif het_atm_feature == 'ACC':
                            for i in range(het_atm_feature_count):
                                motherstructure.known_acc = np.append(motherstructure.known_acc, het_atm_index)
                                if het_atm_index not in motherstructure.allpos_prot_don_acc:  # This makes sure I dont append it more than once since it can be written in the file twice for having possibility of being a donor and acceptor
                                    motherstructure.allpos_prot_don_acc = np.append(motherstructure.allpos_prot_don_acc, het_atm_index)

                            print('Atom {} ({}) is added to known_acc list {} time(s)'.format(het_atm_index, het_atm_full_name, het_atm_feature_count))
                        print('#######################################')
        print("################################################################################")

    ################################################################################################################################

    ##########################################################################################################################################################################################################################################################################################################################################################################################
    # Restructuring ASP/GLU
    # Neg residues Check
    ########################
    # Check ASP and GLU
    ########################
    print('\nDealing with Negative residues, ASP/ASH and GLU/GLH:')
    print('###########################################\n\n')

    asp_glu_sc_O_atms_analyzed = []
    acidic_atom_names_that_were_flipped = []
    negative_residues_to_be_changed_to_donor_form = []
    acidic_residues_branching_by_atoms = []
    acidic_residues_branching_by_residues = []

    # All ASP and GLU residues' indices of the protein
    asp_od1_glu_oe1_atoms = np.append(motherstructure.topology.select('name OD1 and resname ASP ASH'), motherstructure.topology.select('name OE1 and resname GLU GLH'))
    asp_glu_residues = np.empty((0,), dtype=int)
    for i in asp_od1_glu_oe1_atoms:
        asp_glu_residues = np.append(asp_glu_residues, motherstructure.topology.atom(i).residue.index)

    if len(asp_glu_residues) > 0:  # if there are ASP/GLU/ASH/GLH residues in this protein

        ##################################################################################################################
        # Here, we start off by making all ASH/GLH residue names into their unprotonated form (ASP/GLU).
        for i in asp_glu_residues:
            if motherstructure.topology.residue(i).name in ['ASH', 'GLH']:
                if motherstructure.topology.residue(i).name == 'ASH':
                    motherstructure.topology.residue(i).name = 'ASP'
                elif motherstructure.topology.residue(i).name == 'GLH':
                    motherstructure.topology.residue(i).name = 'GLU'
        ##################################################################################################################

        neg_atm_ndx_1 = np.sort(np.append(motherstructure.topology.select('name OD1 and resname ASP ASH'), motherstructure.topology.select('name OE1 and resname GLU GLH')))
        neg_atm_ndx_2 = np.sort(np.append(motherstructure.topology.select('name OD2 and resname ASP ASH'), motherstructure.topology.select('name OE2 and resname GLU GLH')))

        # neg_bond2 is the list that contains the atoms that can be neighbored to ASP/GLU side chain oxygen atoms.
        neg_bond2 = np.sort(np.append(neg_atm_ndx_1, neg_atm_ndx_2))

        neg_bond = len(neg_atm_ndx_1)

        #########################################################################################################################################
        # Finding ASP/ASH/GLU/GLH residues that are neighbored to the cations. any residue neighbored to a cation will be left unprotonated.
        neg_res_atoms_neighbored_to_cations = np.empty((0,), dtype=int)
        for i in motherstructure.cations:
            tmp_neighbor_list = md.compute_neighbors(motherstructure.structure, necessary_functions.h_bond_heavy_atm_cutoff, np.asarray([i]), neg_bond2, periodic=False)[0]
            neg_res_atoms_neighbored_to_cations = np.sort(np.unique(np.append(neg_res_atoms_neighbored_to_cations, tmp_neighbor_list)))

        neg_res_indices_neighbored_to_cations = np.empty((0,), dtype=int)
        for i in neg_res_atoms_neighbored_to_cations:
            neg_res_indices_neighbored_to_cations = np.unique(np.append(neg_res_indices_neighbored_to_cations, motherstructure.topology.atom(i).residue.index))
        print("Residues {} are neighbored with cations. They will be left unprotonated.".format(neg_res_indices_neighbored_to_cations))
        for i in neg_res_indices_neighbored_to_cations:
            print(motherstructure.topology.residue(i))

        for i in neg_res_indices_neighbored_to_cations:
            # here I'm appending each atom of the two atoms of each residue TWICE as known_acc
            sidechain_atoms = motherstructure.topology.select('name OD1 OE1 OD2 OE2 and resname ASP ASH GLU GLH and resid {}'.format(i))
            for j in sidechain_atoms:
                motherstructure.known_acc = np.append(motherstructure.known_acc, j)
                motherstructure.known_acc = np.append(motherstructure.known_acc, j)
                asp_glu_sc_O_atms_analyzed.append(j)
        #########################################################################################################################################

        for i in np.arange(neg_bond):
            print("i is {}".format(i))
            if neg_atm_ndx_1[i] not in asp_glu_sc_O_atms_analyzed and neg_atm_ndx_2[i] not in asp_glu_sc_O_atms_analyzed:

                ############################################
                # if the looped residue is neither previously protonated nor set in branching (not acidic_residues_branching_by_atoms)
                print('Atoms {} and {} are the two side chain oxygen atoms of {}.'.format(neg_atm_ndx_1[i], neg_atm_ndx_2[i], motherstructure.topology.atom(neg_atm_ndx_1[i]).residue))
                negndx_O1 = md.compute_neighbors(motherstructure.structure, necessary_functions.h_bond_heavy_atm_cutoff, np.asarray([neg_atm_ndx_1[i]]), neg_bond2, periodic=False)
                negndx_O2 = md.compute_neighbors(motherstructure.structure, necessary_functions.h_bond_heavy_atm_cutoff, np.asarray([neg_atm_ndx_2[i]]), neg_bond2, periodic=False)

                negndx_O1 = list(negndx_O1[0])
                negndx_O2 = list(negndx_O2[0])
                ############################################

                ############################################
                # Here, we delete OD1/OE1 and OD2/OE2 atoms from each other's neighbor lists.
                neg_atm_ndx_0102 = np.empty((0,), dtype=int)
                neg_atm_ndx_0102 = np.append(neg_atm_ndx_0102, (neg_atm_ndx_1[i], neg_atm_ndx_2[i]))
                for j in neg_atm_ndx_0102:
                    if j in negndx_O1:
                        negndx_O1.remove(j)
                    if j in negndx_O2:
                        negndx_O2.remove(j)

                print('Neighbors of atom {} are {}.'.format(neg_atm_ndx_1[i], negndx_O1))
                print('Neighbors of atom {} are {}.'.format(neg_atm_ndx_2[i], negndx_O2))
                ############################################

                if len(negndx_O1) == 0 and len(negndx_O2) == 0:
                    # For isolated ASP/GLU, OE2/OD2 will act as two acceptors (two lone pairs), the third lone pair on OE2/OD2 will be delocalized. OE1/OD1 will act as two acceptors (two lone pairs).
                    motherstructure.known_acc = np.append(motherstructure.known_acc, neg_atm_ndx_1[i])
                    motherstructure.known_acc = np.append(motherstructure.known_acc, neg_atm_ndx_2[i])
                    motherstructure.known_acc = np.append(motherstructure.known_acc, neg_atm_ndx_1[i])
                    motherstructure.known_acc = np.append(motherstructure.known_acc, neg_atm_ndx_2[i])
                    asp_glu_sc_O_atms_analyzed.append(neg_atm_ndx_1[i])
                    asp_glu_sc_O_atms_analyzed.append(neg_atm_ndx_2[i])
                    print('{} is an isolated residue that doesn\'t have any neighbors to its two side chain oxygen atoms within the heavy-heavy atom cutoff'.format(motherstructure.topology.atom(neg_atm_ndx_1[i]).residue))

                else:
                    ############################################
                    # Here we will create list of heavy to heavy atm pair interactions.
                    # For each pair, the first index will be the atom (to which we will find the 2 possible H positions, later) and the second index is the atom to which the H will donate the H-bond.
                    # For each pair in this pair interactions list, we will make sure the first index (the donor) is not involved in a pair where the donor's residue has been protonated or is involved in being branched out in ACID DYAD.
                    pair_interactions = []
                    for j in negndx_O1:
                        # we make sure that the acceptor CAN accept H-bond. it can accept if it has been analyzed before and its acc is still available (which means its acc is in known_acc)
                        if (j in asp_glu_sc_O_atms_analyzed and j in motherstructure.known_acc) or j not in asp_glu_sc_O_atms_analyzed:
                            pair_interactions.append([neg_atm_ndx_1[i], j])
                        else:
                            print("{} (atom {}) donating H-bond to ----- {} (atom {}) is NOT possible. We won't consider this pair interaction.".format(motherstructure.topology.atom(neg_atm_ndx_1[i]), neg_atm_ndx_1[i], motherstructure.topology.atom(j), j))

                        # we make sure that the donor cant be in acid dyad, protonated residues, or cation-interacted residue
                        if motherstructure.topology.atom(j).residue.index not in negative_residues_to_be_changed_to_donor_form and \
                                motherstructure.topology.atom(j).residue.index not in necessary_functions.flatten(acidic_residues_branching_by_residues) and \
                                motherstructure.topology.atom(j).residue.index not in neg_res_indices_neighbored_to_cations:
                            pair_interactions.append([j, neg_atm_ndx_1[i]])
                        else:
                            print("{} (atom {}) donating H-bond to ----- {} (atom {}) is NOT possible. We won't consider this pair interaction.".format(motherstructure.topology.atom(j), j, motherstructure.topology.atom(neg_atm_ndx_1[i]), neg_atm_ndx_1[i]))

                    for j in negndx_O2:
                        # we make sure that the acceptor CAN accept H-bond. it can accept if it has been analyzed before and its acc is still available (which means its acc is in known_acc)
                        if (j in asp_glu_sc_O_atms_analyzed and j in motherstructure.known_acc) or j not in asp_glu_sc_O_atms_analyzed:
                            pair_interactions.append([neg_atm_ndx_2[i], j])
                        else:
                            print("{} (atom {}) donating H-bond to ----- {} (atom {}) is NOT possible. We won't consider this pair interaction.".format(motherstructure.topology.atom(neg_atm_ndx_2[i]), neg_atm_ndx_2[i], motherstructure.topology.atom(j), j))

                        # we make sure that the donor cant be in acid dyad, protonated residues, or cation-interacted residue
                        if motherstructure.topology.atom(j).residue.index not in negative_residues_to_be_changed_to_donor_form and \
                                motherstructure.topology.atom(j).residue.index not in necessary_functions.flatten(acidic_residues_branching_by_residues) and \
                                motherstructure.topology.atom(j).residue.index not in neg_res_indices_neighbored_to_cations:
                            pair_interactions.append([j, neg_atm_ndx_2[i]])
                        else:
                            print("{} (atom {}) donating H-bond to ----- {} (atom {}) is NOT possible. We won't consider this pair interaction.".format(motherstructure.topology.atom(j), j, motherstructure.topology.atom(neg_atm_ndx_2[i]), neg_atm_ndx_2[i]))

                    ############################################

                    ############################################
                    # Here we want to filter the pair interactions that cant have an H-bond due to bad angles.
                    filtered_pair_interactions = []
                    for index, pair in enumerate(pair_interactions):
                        print(pair)
                        if necessary_functions.filter_out_non_hbond_asp_glu_pair_interaction(motherstructure, pair) is not True:  # this pair interaction should be kept and not filtered out/deleted
                            filtered_pair_interactions.append(pair)
                    pair_interactions = filtered_pair_interactions
                    ############################################

                    if len(pair_interactions) != 0:
                        ############################################
                        # This tells us how many ASP/GLU residues interacting in this network.
                        residues_indices_involved = []
                        for j in necessary_functions.flatten(pair_interactions):
                            if motherstructure.topology.atom(j).residue.index not in residues_indices_involved:
                                residues_indices_involved.append(motherstructure.topology.atom(j).residue.index)

                        # informs the user that there is more than two residues interacting - Acid Triad??
                        if len(residues_indices_involved) > 2:
                            print("####################################")
                            print("Warning: A rare occasion of an acid triad (or more than a triad) is observed?")
                            print("{} are the residues indices which are interacting in the network.".format(residues_indices_involved))
                            print("The residues of this network are:")
                            for j in residues_indices_involved:
                                print(motherstructure.topology.residue(j))
                            print("####################################")
                        ############################################

                        ############################################
                        # Here, I find the list of the two side oxygen atoms that are involved in the network
                        all_sc_o_atms_of_the_residues_involved = []
                        for j in pair_interactions:
                            for k in j:
                                sc_o_atms = motherstructure.topology.select('name OD1 OD2 OE1 OE2 and resname ASP ASH GLU GLH and resid {}'.format(motherstructure.topology.atom(k).residue.index))
                                for l in sc_o_atms:
                                    if l not in all_sc_o_atms_of_the_residues_involved:
                                        all_sc_o_atms_of_the_residues_involved.append(l)
                        ############################################

                        ############################################
                        # calculating energies of different pair interactions seen
                        print('####################################')
                        print('\nThe pair interactions are:')
                        pair_interactions_energy_dic = {}
                        pair_interactions_min_E_kept_dic = {}
                        for interaction_index, j in enumerate(pair_interactions):
                            print('####################################')
                            print('This pair interaction index is {}'.format(interaction_index))
                            print('This pair interaction is {}'.format(j))
                            print("{}'s donor is interacting with {}'s acceptor.".format(motherstructure.topology.atom(j[0]), motherstructure.topology.atom(j[1])))
                            mediating_h = necessary_functions.coords_of_don_neighbor_atom(motherstructure, j[0], j[1])

                            datoms = np.asarray([j[0], j[1]], dtype="int").reshape(1, 2)
                            dist = md.compute_distances(motherstructure.structure, datoms)[0][0]
                            print('The distance between the two heavy atoms is {} A ({} nm)'.format(round(dist*10, 3), round(dist, 4)))

                            pair_interaction_energies = []
                            for k in mediating_h:
                                vector_1 = motherstructure.xyz[0][j[0]] - k
                                vector_2 = motherstructure.xyz[0][j[1]] - k
                                vector1_vector_2_cosine_angle = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
                                acc_H_don_angle_in_radians = np.arccos(vector1_vector_2_cosine_angle)
                                acc_H_don_angle_in_degrees = np.degrees(acc_H_don_angle_in_radians)
                                print('The O-H---O angle is {} degrees.'.format(round(acc_H_don_angle_in_degrees, 2)))

                                r_and_theta = np.array([dist, acc_H_don_angle_in_degrees])
                                pair_interaction_energy = necessary_functions.PES_lookup_table(motherstructure, dist, acc_H_don_angle_in_degrees)

                                print('       The H-bond energy is {} kcal/mol.'.format(round(pair_interaction_energy, 3)))

                                pair_interaction_energies.append(pair_interaction_energy)

                            # Here is a trick, the value of the key in the dictionary will be a list that contains the two energies in the list and the index of the minimum E so we can know if that donor atom is interacting with different neighbors using one location of two locations. if its one location then we will append one acc in knowns and if its using two location in the most minimum pairs then we might end up not having any of those in the known acc because they are sterically hindered
                            pair_interactions_energy_dic[interaction_index] = [pair_interaction_energies, pair_interaction_energies.index(min(pair_interaction_energies))]

                            pair_interactions_min_E_kept_dic[interaction_index] = sorted(pair_interaction_energies)[0]
                            print('####################################')
                        print('####################################################')

                        # ordering the interactions based on energy
                        pair_interactions_ordered_in_energy = sorted(pair_interactions_min_E_kept_dic.items(), key=lambda x: x[1])
                        pair_interactions_ordered_in_energy = [list(ele) for ele in pair_interactions_ordered_in_energy]
                        print('\nASP/GLU pair interactions energies are:\n')
                        for j in pair_interactions_ordered_in_energy:
                            print("Pair interaction index {} has energy of ---------> {} kcal/mol.".format(j[0], round(j[1], 2)))
                        print('\n####################################################')

                        most_favorable_interaction_E = pair_interactions_ordered_in_energy[0][1]
                        most_favorable_interaction_pairs_by_indices = []
                        for j in pair_interactions_ordered_in_energy:
                            if round(abs(j[1] - most_favorable_interaction_E), 2) < degenerate_states_e_cutoff:  # here, we will keep the indices of the pair interactions that are within the degenerate states E cutoff from the most stable (lowest energetically)
                                most_favorable_interaction_pairs_by_indices.append(j[0])
                        ############################################

                        ############################################
                        asp_glu_o_atm_to_donate_and_Hloc_string = []
                        sc_O_atms_that_win_H_in_the_network = []
                        asp_glu_donating_res_indices = []
                        asp_glu_o_atm_to_donate = []
                        asp_glu_o_atm_to_accept = []
                        scanned_pairs = []
                        print("\nBelow are the pair interactions that are within the degenerate states E cutoff from the lowest energetically:")
                        for pair_index in most_favorable_interaction_pairs_by_indices:
                            print(pair_interactions[pair_index])
                            pair_interaction_donor = pair_interactions[pair_index][0]
                            pair_interaction_acceptor = pair_interactions[pair_index][1]

                            if pair_interaction_donor not in sc_O_atms_that_win_H_in_the_network:
                                sc_O_atms_that_win_H_in_the_network.append(pair_interaction_donor)

                            my_current_pair = [pair_interactions[pair_index]]
                            if any(item in my_current_pair for item in scanned_pairs) is False:
                                asp_glu_o_atm_to_donate_and_Hloc_string.append(str(pair_interaction_donor) + '_' + str(pair_interactions_energy_dic[pair_index][1]))
                                asp_glu_o_atm_to_accept.append(pair_interaction_acceptor)
                                asp_glu_donating_res_indices.append(motherstructure.topology.atom(pair_interaction_donor).residue.index)

                                scanned_pairs.append([pair_interaction_donor, pair_interaction_acceptor])
                                scanned_pairs.append([pair_interaction_acceptor, pair_interaction_donor])

                            print("{} can donate an H-bond to {}. Energy of the H-bond ---------> {} kcal/mol.".format(motherstructure.topology.atom(pair_interaction_donor), motherstructure.topology.atom(pair_interaction_acceptor), round(pair_interactions_min_E_kept_dic[pair_index], 2)))

                        print('####################################')

                        # find out how many donor atoms present. if you see two instances of the same item, this means I should not add it at all in known_acc because the atom is found to have two possible hydrogen bonds where it can be donor
                        asp_glu_o_atm_to_donate_and_Hloc_string_uniques = np.unique(np.array(asp_glu_o_atm_to_donate_and_Hloc_string))
                        for j in asp_glu_o_atm_to_donate_and_Hloc_string_uniques:
                            asp_glu_o_atm_to_donate.append(int(j.split('_')[0]))

                        asp_glu_o_atm_don_and_acc = list(np.append(asp_glu_o_atm_to_donate, asp_glu_o_atm_to_accept))
                        ############################################

                        ############################################
                        if len(sc_O_atms_that_win_H_in_the_network) == 1:
                            print('There is only one donor atom - No proton traps seen.')
                            the_winner_atom_of_the_H_between_the_acid_pair = sc_O_atms_that_win_H_in_the_network[0]
                            negative_residues_to_be_changed_to_donor_form.append(motherstructure.topology.atom(the_winner_atom_of_the_H_between_the_acid_pair).residue.index)
                            print("The winning residue (the residue to be protonated) is {}".format(motherstructure.topology.atom(the_winner_atom_of_the_H_between_the_acid_pair).residue))

                            # Checking if I need to flip atom names cuz OD1/OE1 can't have the H. Only OD2/OE2 can.
                            if (motherstructure.topology.atom(the_winner_atom_of_the_H_between_the_acid_pair).name == 'OD2') or (motherstructure.topology.atom(the_winner_atom_of_the_H_between_the_acid_pair).name == 'OE2'):
                                print('No atom names of the winning residue need to be flipped!!')
                            elif (motherstructure.topology.atom(the_winner_atom_of_the_H_between_the_acid_pair).name == 'OD1') or (motherstructure.topology.atom(the_winner_atom_of_the_H_between_the_acid_pair).name == 'OE1'):
                                print('Atoms were {} ({}) and {} ({}) '.format(the_winner_atom_of_the_H_between_the_acid_pair, motherstructure.topology.atom(the_winner_atom_of_the_H_between_the_acid_pair).name, the_winner_atom_of_the_H_between_the_acid_pair + 1, motherstructure.topology.atom(the_winner_atom_of_the_H_between_the_acid_pair + 1).name))
                                if motherstructure.topology.atom(the_winner_atom_of_the_H_between_the_acid_pair).name == 'OD1':
                                    motherstructure.topology.atom(the_winner_atom_of_the_H_between_the_acid_pair).name = 'OD2'
                                    motherstructure.topology.atom(the_winner_atom_of_the_H_between_the_acid_pair + 1).name = 'OD1'
                                elif motherstructure.topology.atom(the_winner_atom_of_the_H_between_the_acid_pair).name == 'OE1':
                                    motherstructure.topology.atom(the_winner_atom_of_the_H_between_the_acid_pair).name = 'OE2'
                                    motherstructure.topology.atom(the_winner_atom_of_the_H_between_the_acid_pair + 1).name = 'OE1'
                                acidic_atom_names_that_were_flipped.append([the_winner_atom_of_the_H_between_the_acid_pair, the_winner_atom_of_the_H_between_the_acid_pair + 1])
                                print('Atoms are now {} ({}) and {} ({})'.format(the_winner_atom_of_the_H_between_the_acid_pair, motherstructure.topology.atom(the_winner_atom_of_the_H_between_the_acid_pair).name, the_winner_atom_of_the_H_between_the_acid_pair + 1, motherstructure.topology.atom(the_winner_atom_of_the_H_between_the_acid_pair + 1).name))
                                print('Atoms {} \'s names of the winning residue were just flipped'.format([the_winner_atom_of_the_H_between_the_acid_pair, the_winner_atom_of_the_H_between_the_acid_pair + 1]))

                        elif len(sc_O_atms_that_win_H_in_the_network) > 1:
                            print("There is a proton trap - Acid Dyad!!")
                            print("This acid dyad is {}".format(sc_O_atms_that_win_H_in_the_network))
                            acidic_residues_branching_by_atoms.append(sc_O_atms_that_win_H_in_the_network)
                            acidic_branching_residues = []
                            for j in sc_O_atms_that_win_H_in_the_network:
                                if motherstructure.topology.atom(j).residue.index not in acidic_branching_residues:
                                    acidic_branching_residues.append(motherstructure.topology.atom(j).residue.index)
                            acidic_residues_branching_by_residues.append(acidic_branching_residues)
                        ############################################

                        for j in all_sc_o_atms_of_the_residues_involved:
                            if j not in asp_glu_sc_O_atms_analyzed:
                                if j in asp_glu_o_atm_don_and_acc:
                                    accepting_counter = 2 - asp_glu_o_atm_don_and_acc.count(j)  # 2 because each O atom can accept twice but we subtract from it the times the atom is accepting in the favorable H bond interaction
                                elif j not in asp_glu_o_atm_don_and_acc:
                                    accepting_counter = 2
                                print("Atom {} ({}) will be added to known_acc list {} times.".format(j, motherstructure.topology.atom(j), accepting_counter))
                                for k in range(accepting_counter):
                                    motherstructure.known_acc = np.append(motherstructure.known_acc, j)
                                asp_glu_sc_O_atms_analyzed.append(j)
                            else:
                                if j in asp_glu_o_atm_to_accept:
                                    print("Atom {} ({}) will not be added to known-acc list. In fact, we will delete one of its acceptor from known_acc.".format(j, motherstructure.topology.atom(j)))
                                    motherstructure.known_acc = np.delete(motherstructure.known_acc, np.where(motherstructure.known_acc == j)[0][0])

                    else:
                        # For isolated ASP/GLU, OE2/OD2 will act as two acceptors (two lone pairs), the third lone pair on OE2/OD2 will be delocalized. OE1/OD1 will act as two acceptors (two lone pairs).
                        motherstructure.known_acc = np.append(motherstructure.known_acc, neg_atm_ndx_1[i])
                        motherstructure.known_acc = np.append(motherstructure.known_acc, neg_atm_ndx_2[i])
                        motherstructure.known_acc = np.append(motherstructure.known_acc, neg_atm_ndx_1[i])
                        motherstructure.known_acc = np.append(motherstructure.known_acc, neg_atm_ndx_2[i])
                        asp_glu_sc_O_atms_analyzed.append(neg_atm_ndx_1[i])
                        asp_glu_sc_O_atms_analyzed.append(neg_atm_ndx_2[i])
                        print('{} initially had neighbors but the pair interactions were all filtered out.\nThe residue is being treated as an isolated residue that doesn\'t have any neighbors to its two side chain oxygen atoms within the heavy-heavy atom cutoff'.format(motherstructure.topology.atom(neg_atm_ndx_1[i]).residue))

            else:
                print('Atoms {} and {} are the two oxygen atoms of {}.'.format(neg_atm_ndx_1[i], neg_atm_ndx_2[i], motherstructure.topology.atom(neg_atm_ndx_1[i]).residue))
                print('This residue has been previously identified. We will not touch it anymore.')
                if motherstructure.topology.atom(neg_atm_ndx_1[i]).residue.index in neg_res_indices_neighbored_to_cations:
                    print("This residue was analyzed as a cation-neighbored residue earlier in the script and it can't be re-analyzed")
                if motherstructure.topology.atom(neg_atm_ndx_1[i]).residue.index in necessary_functions.flatten(negative_residues_to_be_changed_to_donor_form):
                    print("This residue was analyzed as a protonated residue earlier in the script and it can't be re-analyzed")
                if motherstructure.topology.atom(neg_atm_ndx_1[i]).residue.index in necessary_functions.flatten(acidic_residues_branching_by_residues):
                    print("This residue was analyzed as part of a proton trap (ACID DYAD) earlier in the script and it can't be re-analyzed")

            # print('known_acc has {} atoms now'.format(len(motherstructure.known_acc)))
            # print('last 20 items in known_acc are {}'.format(motherstructure.known_acc[-20:]))
            # print('known_don has {} atoms now'.format(len(motherstructure.known_don)))
            # print('last 20 items in known_don are {}'.format(motherstructure.known_don[-20:]))

            print('\n############################################################\n')

    else:
        print('It seems like no ASP/ASH/GLU/GLH residues are found in this protein structure\n')

    ####################################

    print('\n############################################################\n')
    print('acidic_residues_branching_by_atoms is {}.'.format(acidic_residues_branching_by_atoms))
    print('acidic_residues_branching_by_residues is {}.'.format(acidic_residues_branching_by_residues))
    print('negative_residues_to_be_changed_to_donor_form is {}.'.format(negative_residues_to_be_changed_to_donor_form))
    print('acidic_atom_names_that_were_flipped is {}.'.format(acidic_atom_names_that_were_flipped))
    print('\n############################################################\n')

    print('Converting the negatively charged residues into their neutral donor form .........\n')

    if len(negative_residues_to_be_changed_to_donor_form) > 0:
        for i in negative_residues_to_be_changed_to_donor_form:
            print('{} is {} before conversion'.format(i, motherstructure.topology.residue(i).name))
            if motherstructure.topology.residue(i).name == 'ASP':
                motherstructure.topology.residue(i).name = 'ASH'
            elif motherstructure.topology.residue(i).name == 'GLU':
                motherstructure.topology.residue(i).name = 'GLH'
            print('{} is {} after conversion to donor form'.format(i, motherstructure.topology.residue(i).name))
            print('############')
    else:
        print('There are no negative residues to be changed to the donor form')

    ################################################################################################################################

    # Getting all the possible combinations
    if len(acidic_residues_branching_by_atoms) == 0:
        acid_dyad_branches = []
    else:
        number_of_acid_dyad_branches = 1
        for branch_set in acidic_residues_branching_by_atoms:
            number_of_acid_dyad_branches = number_of_acid_dyad_branches * len(branch_set)

        acid_dyad_branches = list(itertools.product(*acidic_residues_branching_by_atoms))

    motherstructure.acid_dyad_branches = acid_dyad_branches
    motherstructure.acidic_residues_branching_by_atoms = acidic_residues_branching_by_atoms

    print('\n####################################################################################################################################################################################\n')

    #####################
    # END ASP and GLU
    #####################

    ##########################################################################################################################################################################################################################################################################################################################################################################################

    # Pos residues Check
    ########################
    # Check LYS and ARG
    ########################
    print('Dealing with Positive residues ARG and LYS:')
    print('###########################################\n\n')

    motherstructure.reduced_topology_not_available_donors = []
    positive_residues_to_be_changed_to_neutral_form = []  # This list will have the indices of the LYS residues that will be changed to 'LYN'.
    previously_identified_arg_atoms = []  # This list will have arg NH1/NH2 atoms that get identified/appended to known_don list to be used in H-bonds.
    lys_arg_sc_N_atms_analyzed = []

    # All ASP and GLU residues' indices of the protein
    arg_nh1_nh2_lys_nz_atoms = np.append(motherstructure.topology.select('name NH1 NH2 and resname ARG ARN'), motherstructure.topology.select('name NZ and resname LYS LYN'))
    arg_lys_residues = np.empty((0,), dtype=int)
    for i in arg_nh1_nh2_lys_nz_atoms:
        arg_lys_residues = np.append(arg_lys_residues, motherstructure.topology.atom(i).residue.index)

    if len(arg_lys_residues) > 0:  # if there are ARG/ARN/LYS/LYN residues in this protein

        ##################################################################################################################
        # Here, we start off by making all ARN/LYN residue names into their charged form (ARG/LYS).
        for i in arg_lys_residues:
            if motherstructure.topology.residue(i).name in ['ARN', 'LYN']:
                if motherstructure.topology.residue(i).name == 'ARN':
                    motherstructure.topology.residue(i).name = 'ARG'
                elif motherstructure.topology.residue(i).name == 'LYN':
                    motherstructure.topology.residue(i).name = 'LYS'
        ##################################################################################################################

        ############################################
        # I figured it might be a lot easier to already append all ARG/ARN (NH1 NH2) and all LYS/LYN (NZ) to known don list
        for i in motherstructure.allpos_prot_don_acc:
            atom_name = motherstructure.topology.atom(i).name
            residue_name = motherstructure.topology.atom(i).residue.name
            if atom_name == 'NH1' and residue_name in ['ARG', 'ARN']:  # ARG(NH1) is always two donors. The lone pair is fully delocalized.
                motherstructure.known_don = np.append(motherstructure.known_don, i)
                motherstructure.known_don = np.append(motherstructure.known_don, i)
                lys_arg_sc_N_atms_analyzed.append(i)
            elif atom_name == 'NH2' and residue_name in ['ARG', 'ARN']:  # ARG(NH2) is always two donors. The lone pair is fully delocalized.
                motherstructure.known_don = np.append(motherstructure.known_don, i)
                motherstructure.known_don = np.append(motherstructure.known_don, i)
                lys_arg_sc_N_atms_analyzed.append(i)
            elif atom_name == 'NZ' and residue_name in ['LYS', 'LYN']:  # LYS(NZ) is always three donors. LYN(NZ) is always two donors. We will treat all lysines as LYS (3 donors) until proven otherwise in the residue by residue analysis in the positive block
                motherstructure.known_don = np.append(motherstructure.known_don, i)
                motherstructure.known_don = np.append(motherstructure.known_don, i)
                motherstructure.known_don = np.append(motherstructure.known_don, i)
        ############################################

        pos_atm_ndx_1 = motherstructure.topology.select('name NH1 and resname ARG ARN')
        pos_atm_ndx_2 = motherstructure.topology.select('name NH2 and resname ARG ARN')
        pos_atm_ndx_3 = motherstructure.topology.select('name NZ and resname LYS LYN')

        pos_atm_ndx = np.append(pos_atm_ndx_1, pos_atm_ndx_2)
        pos_atm_ndx = np.sort(np.append(pos_atm_ndx, pos_atm_ndx_3))  # List of NH1 NH2 NZ atoms of ARG and LYS

        # pos_bond2 = np.sort(np.append(pos_atm_ndx, backbone_amide_O_N))  # List of the heavy atoms that are going to be checked if neighbors.
        pos_bond2 = pos_atm_ndx

        for i in np.arange(len(pos_atm_ndx_3)):
            print("i is {}".format(i))
            print('Atom {} ({}) is the side chain nitrogen atom of {}.'.format(pos_atm_ndx_3[i], motherstructure.topology.atom(pos_atm_ndx_3[i]).name, motherstructure.topology.atom(pos_atm_ndx_3[i]).residue))

            # getting the neighbors' list of the residue
            posndx = md.compute_neighbors(motherstructure.structure, necessary_functions.h_bond_heavy_atm_cutoff, np.asarray([pos_atm_ndx_3[i]]), pos_bond2, periodic=False)[0]
            posndx = list(posndx)
            pos_res_neighbors_count = len(posndx)
            print('Neighbors of atom {} are {}.'.format(pos_atm_ndx_3[i], posndx))

            if pos_atm_ndx_3[i] not in lys_arg_sc_N_atms_analyzed and motherstructure.topology.atom(pos_atm_ndx_3[i]).residue.index not in positive_residues_to_be_changed_to_neutral_form:
                if pos_res_neighbors_count == 0:
                    print('{} is an isolated residue that doesn\'t have any neighbors (within the heavy-heavy atom cutoff) to its side chain nitrogen(s).'.format(motherstructure.topology.atom(pos_atm_ndx_3[i]).residue))
                    print('Atom {} will NOT get added to known_don list.\n     It was added at the beginning of the analysis of the positive residues block.'.format(pos_atm_ndx_3[i]))
                    lys_arg_sc_N_atms_analyzed.append(pos_atm_ndx_3[i])

                else:
                    ############################################
                    # Here we will create list of heavy to heavy atm pair interactions.
                    # For each pair, the first index will be the atom (to which we will find the possible H positions, later) and the second index is H bond acceptor - it's the atom to which the H will donate the H-bond.
                    # For each pair in this pair interactions list, we will make sure the second index (the acceptor on a potential LYN) is not involved an H-bond that made the residue a LYN instead of LYS
                    pair_interactions = []
                    for j in posndx:
                        if j in motherstructure.known_don:
                            if motherstructure.topology.atom(j).residue.name == 'ARG':
                                pair_interactions.append([j, pos_atm_ndx_3[i]])
                            elif motherstructure.topology.atom(j).residue.name == 'LYS':
                                pair_interactions.append([j, pos_atm_ndx_3[i]])
                                if motherstructure.topology.atom(j).residue.index not in positive_residues_to_be_changed_to_neutral_form:
                                    pair_interactions.append([pos_atm_ndx_3[i], j])
                    print("pair interactions before filtering: {}.".format(pair_interactions))
                    ############################################

                    ############################################
                    # Here we want to filter the pair interactions that cant have an H-bond due to bad angles.
                    filtered_pair_interactions = []
                    for index, pair in enumerate(pair_interactions):
                        print(pair)
                        if necessary_functions.filter_out_non_hbond_lys_arg_pair_interaction(motherstructure, pair) is not True:  # this pair interaction should be kept and not filtered/deleted
                            filtered_pair_interactions.append(pair)
                    pair_interactions = filtered_pair_interactions
                    print("pair interactions after filtering: {}.".format(pair_interactions))
                    ############################################

                    if len(pair_interactions) != 0:
                        ############################################
                        # This tells us how many LYS/ARG residues interacting in this network.
                        residues_indices_involved = []
                        for j in necessary_functions.flatten(pair_interactions):
                            if motherstructure.topology.atom(j).residue.index not in residues_indices_involved:
                                residues_indices_involved.append(motherstructure.topology.atom(j).residue.index)

                        # informs the user that there is more than two residues interacting
                        if len(residues_indices_involved) > 2:
                            print("####################################")
                            print("Warning: A rare occasion of more than two positive residues is observed?")
                            print("{} are the residues indices which are interacting in the network.".format(residues_indices_involved))
                            print("The residues of this network are:")
                            for j in residues_indices_involved:
                                print(motherstructure.topology.residue(j))
                            print("####################################")
                        ############################################

                        ############################################
                        # Here, I find the list of the two side oxygen atoms that are involved in the network
                        all_sc_n_atms_of_the_residues_involved = []
                        for j in pair_interactions:
                            for k in j:
                                sc_n_atms = motherstructure.topology.select('name NH1 NH2 NZ and resname LYS LYN ARG and resid {}'.format(motherstructure.topology.atom(k).residue.index))
                                for l in sc_n_atms:
                                    if l not in all_sc_n_atms_of_the_residues_involved:
                                        all_sc_n_atms_of_the_residues_involved.append(l)
                        ############################################

                        ############################################
                        # calculating energies of different interactions seen
                        print('####################################')
                        print('\nThe pair interactions are:')
                        pair_interactions_energy_dic = {}
                        pair_interactions_min_E_kept_dic = {}
                        pair_interactions_mediating_h = {}  # This will let me have the H used in each of the pair interaction so when I settle and find the winning pair interaction I can get the H index which was used and append it to reduced list which have donors that cant be reused.
                        for interaction_index, j in enumerate(pair_interactions):
                            print('####################################')
                            print('This pair interaction index is {}'.format(interaction_index))
                            print('This pair interaction is {}'.format(j))
                            print("{}'s donor is interacting with {}'s acceptor.".format(motherstructure.topology.atom(j[0]), motherstructure.topology.atom(j[1])))
                            mediating_h = necessary_functions.coords_of_don_neighbor_atom(motherstructure, j[0], j[1])
                            pair_interactions_mediating_h[interaction_index] = mediating_h  # This will let me have the H used in each of the pair interaction so when I settle and find the winning pair interaction I can get the H index which was used and append it to reduced list which have donors that cant be reused.

                            mediating_h = [item[1] for item in mediating_h]

                            datoms = np.asarray([j[0], j[1]], dtype="int").reshape(1, 2)
                            dist = md.compute_distances(motherstructure.structure, datoms)[0][0]
                            print('The distance between the two heavy atoms is {} A ({} nm)'.format(round(dist*10, 3), round(dist, 4)))

                            pair_interaction_energies = []
                            for k in mediating_h:
                                vector_1 = motherstructure.xyz[0][j[0]] - k
                                vector_2 = motherstructure.xyz[0][j[1]] - k
                                vector1_vector_2_cosine_angle = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
                                acc_H_don_angle_in_radians = np.arccos(vector1_vector_2_cosine_angle)
                                acc_H_don_angle_in_degrees = np.degrees(acc_H_don_angle_in_radians)
                                print('The N-H---N angle is {} degrees.'.format(round(acc_H_don_angle_in_degrees, 2)))

                                pair_interaction_energy = necessary_functions.PES_lookup_table(motherstructure, dist, acc_H_don_angle_in_degrees)

                                print('The H-bond energy is {} kcal/mol.'.format(round(pair_interaction_energy, 3)))

                                pair_interaction_energies.append(pair_interaction_energy)

                            # Here is a trick, the value of the key in the dictionary will be a list that contains the two energies in the list and the index of the minimum E so we can know if that donor atom is interacting with different neighbors using one location of two locations.
                            pair_interactions_energy_dic[interaction_index] = [pair_interaction_energies, pair_interaction_energies.index(min(pair_interaction_energies))]

                            pair_interactions_min_E_kept_dic[interaction_index] = sorted(pair_interaction_energies)[0]
                            print('####################################')
                        print('####################################################')

                        # ordering the interactions based on energy
                        pair_interactions_ordered_in_energy = sorted(pair_interactions_min_E_kept_dic.items(), key=lambda x: x[1])
                        pair_interactions_ordered_in_energy = [list(ele) for ele in pair_interactions_ordered_in_energy]
                        print('\nARG/LYS pair interactions energies are:\n')
                        for j in pair_interactions_ordered_in_energy:
                            print("Pair interaction index {} has energy of ---------> {} kcal/mol.".format(j[0], round(j[1], 2)))
                        print('\n####################################################')

                        most_favorable_interaction_E = pair_interactions_ordered_in_energy[0][1]
                        most_favorable_interaction_pairs_by_indices = []
                        for j in pair_interactions_ordered_in_energy:
                            if round(abs(j[1] - most_favorable_interaction_E), 2) < degenerate_states_e_cutoff:  # here, we will keep the indices of the pair interactions that are within the degenerate states E cutoff from the most stable (lowest energetically)
                                most_favorable_interaction_pairs_by_indices.append(j[0])
                        pair_interaction_index_winner = most_favorable_interaction_pairs_by_indices[0]
                        ############################################

                        ############################################
                        print('The winning pair interaction index is {}'.format(pair_interaction_index_winner))
                        winning_pair_interaction = pair_interactions[pair_interaction_index_winner]
                        print('The pair interaction is {}'.format(winning_pair_interaction))
                        print("{}'s donor is interacting with {}'s acceptor.".format(motherstructure.topology.atom(winning_pair_interaction[0]), motherstructure.topology.atom(winning_pair_interaction[1])))

                        print('\nAtom {} will be deleted ONCE from known_don list.\n     It was added at the beginning of the analysis of the positive residues block.'.format(winning_pair_interaction[0]))
                        print('Atom {} will be deleted ONCE from known_don list.\n     It was added at the beginning of the analysis of the positive residues block.\n'.format(winning_pair_interaction[1]))
                        motherstructure.known_don = np.delete(motherstructure.known_don, np.where(motherstructure.known_don == winning_pair_interaction[0])[0][0])
                        motherstructure.known_don = np.delete(motherstructure.known_don, np.where(motherstructure.known_don == winning_pair_interaction[1])[0][0])

                        ################################## for making sure the donor H is appended to reduced_topology_not_available_donors ##################################
                        # now, we can go back the dict with the mediating H used for this winning pair interaction and make sure to append the index of the H used so it wouldn't be used again
                        # since pair_interactions_energy_dic has keys and values:
                        # Each key is the interaction index and the value of that key is a list of two items: the first is a list of the two energies and the second item is the index of the H that had the lowest E.
                        # this means that the result of the winning pair's h index will have to be an index ---> probably 0, 1, or 2 (in very rare cases when the neighbor is another LYS with three H)
                        the_used_up_h_index = pair_interactions_energy_dic[pair_interaction_index_winner][1]
                        # now use the_used_up_h_index to locate the topology index of the H used so we can append to reduced_topology_not_available_donors
                        the_used_up_h_topology_index = pair_interactions_mediating_h[pair_interaction_index_winner][the_used_up_h_index][0]
                        the_used_up_h_coords = pair_interactions_mediating_h[pair_interaction_index_winner][the_used_up_h_index][1]  # this is here because it will be needed for the next part
                        motherstructure.reduced_topology_not_available_donors.append(the_used_up_h_topology_index)
                        print('Atom {} just got added to reduced_topology_not_available_donors list. This ensures that this H can\'t be used later.\n'.format(the_used_up_h_topology_index))
                        ######################################################################################################################################################

                        ######## for making sure the acceptor lone pair (where the H cant be in its location) is appended to reduced_topology_not_available_donors #########
                        # here we want the find which location on the NZ atom is the lone pair of the LYS which will be turned into LYN.
                        # finding it will allow us to only have two locations distinguished for future H-bondind with this lysine
                        # easiest way will probably be just finding which of the 3 H's of the LYS (which will be turned into LYN) is closest to the_used_up_h_coords (the donor H coords of the donating residue)
                        the_accepting_residue_mediating_H = necessary_functions.coords_of_don_neighbor_atom(motherstructure, winning_pair_interaction[1], winning_pair_interaction[0])
                        if len(necessary_functions.flatten(the_accepting_residue_mediating_H)) == 4:
                            the_accepting_residue_mediating_H = [the_accepting_residue_mediating_H]
                        dist_dic = {}
                        for ndx, j in enumerate(the_accepting_residue_mediating_H):
                            dist = math.sqrt(((the_used_up_h_coords[0] - j[1][0]) ** 2) + ((the_used_up_h_coords[1] - j[1][1]) ** 2) + ((the_used_up_h_coords[2] - j[1][2]) ** 2))
                            dist_dic[ndx] = dist
                        best_location_for_the_lonepair_index = sorted(dist_dic.items(), key=operator.itemgetter(1))[0][0]
                        best_location_for_the_lonepair_topology_index = the_accepting_residue_mediating_H[best_location_for_the_lonepair_index][0]
                        motherstructure.reduced_topology_not_available_donors.append(best_location_for_the_lonepair_topology_index)
                        print('\nAtom {} just got added to reduced_topology_not_available_donors list.\n'
                              '       We know it is not a donor but we added it here such that we can make sure we only use the other H that can be used for H bond donors later for unknowns\n'
                              '       This ensures that a H in this location can\'t be used later.'.format(best_location_for_the_lonepair_topology_index))
                        ######################################################################################################################################################

                        positive_residues_to_be_changed_to_neutral_form.append(motherstructure.topology.atom(winning_pair_interaction[1]).residue.index)

                        for j in winning_pair_interaction:
                            if motherstructure.topology.atom(j).residue.name in ['LYS', 'LYN']:
                                lys_arg_sc_N_atms_analyzed.append(j)

                    else:
                        print('{} is going to be considered an isolated residue that doesn\'t have any neighbors (within the heavy-heavy atom cutoff) to its side chain nitrogen(s).'.format(motherstructure.topology.atom(pos_atm_ndx_3[i]).residue))
                        print('Atom {} will NOT get added to known_don list.\n     It was added at the beginning of the analysis of the positive residues block.'.format(pos_atm_ndx_3[i]))
                        lys_arg_sc_N_atms_analyzed.append(pos_atm_ndx_3[i])

            else:
                print('This residue has been previously identified.')
                if motherstructure.topology.atom(pos_atm_ndx_3[i]).residue.index in positive_residues_to_be_changed_to_neutral_form:
                    print('This residue was changed into its neutral form (LYN).')

            # print('known_acc has {} atoms now'.format(len(motherstructure.known_acc)))
            # print('last 20 items in known_acc are {}'.format(motherstructure.known_acc[-20:]))
            # print('known_don has {} atoms now'.format(len(motherstructure.known_don)))
            # print('last 20 items in known_don are {}'.format(motherstructure.known_don[-20:]))

            print('\n############################################################\n')

        else:
            print('It seems like no LYS residues found in this protein structure\n')

    print('\npositive_residues_to_be_changed_to_neutral_form is {}.\n'.format(positive_residues_to_be_changed_to_neutral_form))

    print('############################################################')

    print('\nConverting the positively charged residues into their neutral form .........\n')
    if len(positive_residues_to_be_changed_to_neutral_form) > 0:
        for i in positive_residues_to_be_changed_to_neutral_form:
            print('{} is {}'.format(i, motherstructure.topology.residue(i).name))
            if motherstructure.topology.residue(i).name == 'LYS':
                motherstructure.topology.residue(i).name = 'LYN'
                print('{} is {} after conversion to neutral form'.format(i, motherstructure.topology.residue(i).name))
            print('############')
    else:
        print('There are no positive residues to be changed to the neutral form')
    print('\n####################################################################################################################################################################################\n')

    ########################
    # END LYS and ARG
    ########################

    ##########################################################################################################################################################################################################################################################################################################################################################################################

    ########################
    # Check for SSBond
    ########################
    print('Dealing with Cysteine residues and Cystine formation:')
    print('##############################\n\n')

    cys_sg = motherstructure.topology.select('name SG and resname CYS CYX')
    cys_sg2 = np.copy(cys_sg)

    ssbnum_atmndx_pair = []
    ssbnum_resid = []
    sulfur_atoms_making_disulfide_bond = []

    if len(cys_sg2) > 0:
        for i in np.arange(len(cys_sg)):
            if cys_sg[i] not in necessary_functions.flatten(sulfur_atoms_making_disulfide_bond):
                print('Atom {} is the sulfur atom of {}'.format(cys_sg[i], motherstructure.topology.atom(cys_sg[i]).residue))
                ssndx = md.compute_neighbors(motherstructure.structure, necessary_functions.cys_disulfide_bond_cutoff, np.asarray([cys_sg[i]]), cys_sg2, periodic=False)
                print('List of neighbors (cysteine\'s sulfur atoms) are:\n{}'.format(ssndx[0]))

                if len(ssndx[0]) == 0:
                    # Isolated CYS sulfur atoms will NOT be added to known_don or known_acc lists because S-H is rarely considered as H-bond donors and never as acceptors due to electronegativity and size.
                    print('Atom {} has NO cysteine sulfur atoms neighbored to it within the cutoff.'.format(cys_sg[i]))
                    print('{} will be kept as CYS'.format(motherstructure.topology.atom(cys_sg[i]).residue))  # cys_sg2 = np.delete(cys_sg2, np.where(cys_sg2 == cys_sg[i])[0][0])

                elif len(ssndx[0]) == 1:
                    print('Atom {} has only one cysteine sulfur atom neighbored to it within the cutoff.'.format(cys_sg[i]))
                    ssbnum_atmndx_pair.append((cys_sg[i], ssndx[0][0]))
                    ssbnum_resid.append((motherstructure.topology.atom(cys_sg[i]).residue.index, motherstructure.topology.atom(ssndx[0][0]).residue.index))
                    sulfur_atoms_making_disulfide_bond.append(cys_sg[i])
                    sulfur_atoms_making_disulfide_bond.append(ssndx[0][0])
                    print('{} (residue {}) and {} (residue {}) are making a disulfide bond.\n     The name of the two residues will be changed to CYX (disulfide bonded cystine).'.format(motherstructure.topology.atom(cys_sg[i]).residue, motherstructure.topology.atom(cys_sg[i]).residue.index, motherstructure.topology.atom(ssndx[0][0]).residue, motherstructure.topology.atom(ssndx[0][0]).residue.index))
                    cys_sg2 = np.delete(cys_sg2, np.where(cys_sg2 == cys_sg[i])[0][0])
                    cys_sg2 = np.delete(cys_sg2, np.where(cys_sg2 == ssndx[0][0])[0][0])

                elif len(ssndx[0]) > 1:
                    dist_dic = {}
                    print('Atom {} has more than one cysteine sulfur atom neighbored to it within the cutoff.'.format(cys_sg[i]))
                    print('Only the sulfur atom neighbor with the shortest distance to {} will be chosen for the disulfide bond'.format(cys_sg[i]))
                    for j in ssndx[0]:
                        # distance between atoms in Hb
                        datoms = np.asarray([cys_sg[i], j], dtype="int").reshape(1, 2)
                        dist = md.compute_distances(motherstructure.structure, datoms)[0][0]
                        dist_dic[j] = dist

                    print(dist_dic)
                    bval = sorted(dist_dic.items(), key=operator.itemgetter(1))[0][0]
                    ssbnum_atmndx_pair.append((cys_sg[i], bval))
                    ssbnum_resid.append((motherstructure.topology.atom(cys_sg[i]).residue.index, motherstructure.topology.atom(bval).residue.index))
                    sulfur_atoms_making_disulfide_bond.append(cys_sg[i])
                    sulfur_atoms_making_disulfide_bond.append(bval)
                    print('{} (residue {}) and {} (residue {}) are making a disulfide bond.\n     The name of the two residues will be changed to CYX (disulfide bonded cystine).'.format(motherstructure.topology.atom(cys_sg[i]).residue, motherstructure.topology.atom(cys_sg[i]).residue.index, motherstructure.topology.atom(bval).residue, motherstructure.topology.atom(bval).residue.index))
                    cys_sg2 = np.delete(cys_sg2, np.where(cys_sg2 == cys_sg[i])[0][0])
                    cys_sg2 = np.delete(cys_sg2, np.where(cys_sg2 == bval)[0][0])

            else:
                print('Atom {} of {} was identified earlier in a CYX-CYX pair (meaning it\'s already disulfide bonded)'.format(cys_sg[i], motherstructure.topology.atom(cys_sg[i]).residue))

            print('\n############################################################\n')

    else:
        print('It seems like no CYS residues found in this protein structure\n')

    # Here we change the bridging cysteine residues from CYS to CYX, CYX is the cys resname of bridging cystines.
    print('Converting the CYS residues into their bridged form .........\n')
    if len(necessary_functions.flatten(ssbnum_atmndx_pair)) > 0:
        for ndx, i in enumerate(ssbnum_atmndx_pair):
            for j in i:
                print('Residue name before conversion is {}'.format(motherstructure.topology.atom(j).residue))
                motherstructure.topology.atom(j).residue.name = 'CYX'
                print('Residue name after conversion is {}'.format(motherstructure.topology.atom(j).residue))
                print('############')
    else:
        print('There are no cysteine residues to be changed to the bridged form (CYX)')
    print('\n####################################################################################################################################################################################\n')

    ########################
    # End SSBond
    ########################

    ##########################################################################################################################################################################################################################################################################################################################################################################################
    # Analyzing unknown residues of SER/THR/TYR/ASN/GLN/HIS
    necessary_functions.initialize_empty_lists(motherstructure)
    necessary_functions.protocol_of_branching(motherstructure, out_prefix)
    ##########################################################################################################################################################################################################################################################################################################################################################################################

    all_output_files = sorted(os.listdir(target_occu_branch_directory_path))
    for file in all_output_files:
        if file[-19:] == '_before_changes.pdb':
            print("We deleted the temporary file: {}".format(file))
            os.remove(file)  # here I just remove the files ending with *_before_changes.pdb because those are not the branches files but those were the files before we switched atom names or fixed indices of atoms and so on.

    with open("Tree_construction_commands.py", "w") as outfile:
        outfile.write('from anytree import Node, RenderTree\n')

        reference_file = glob.glob("*_file_reference.txt")[0]
        with open(reference_file, "r") as outfile_reference_txt:
            reference_file_lines = outfile_reference_txt.readlines()
            branches_detailed_path = [f.split(' ')[0] for f in reference_file_lines]
            output_files = [f.split('___')[1:] for f in branches_detailed_path]

            branches_paths = []
            for branch_output in output_files:
                branch_path = []
                for level_index in range(len(branch_output)):
                    branch_path.append(branch_output[level_index])
                branches_paths.append(branch_path)

            outfile.write('motherstucture = Node("Root")\n')

            for branch_path in branches_paths:
                for part_index in range(len(branch_path)):
                    if part_index == 0:
                        full_name = 'motherstructure_' + branch_path[part_index]
                        outfile.write('{} = Node("{}", parent=motherstucture)\n'.format(full_name, branch_path[part_index]))
                    else:
                        full_name_1 = 'motherstructure_' + "_".join(branch_path[:part_index + 1])
                        full_name_2 = 'motherstructure_' + "_".join(branch_path[:part_index])

                        outfile.write('{} = Node("{}", parent={})\n'.format(full_name_1, branch_path[part_index], full_name_2))

            outfile.write('\nfor pre, fill, node in RenderTree(motherstucture):\n')
            outfile.write('    print("%s%s" % (pre, node.name))')

    print("#############################################################################################################################################################################################")
    print("Here is the tree of branches of this target:")
    # Executing tree code with subprocess
    Tree_construction_path = os.path.join(target_occu_branch_directory_path, "Tree_construction_commands.py")
    Tree_construction_command = "python {}".format(Tree_construction_path)
    Tree_construction_command_line = shlex.split(Tree_construction_command)
    tree_proc = subp.Popen(Tree_construction_command_line)
    tree_proc_ex_code = tree_proc.wait()
    print("#############################################################################################################################################################################################")
    final_occupancy_branchings_time = timeit.default_timer()
    analysis_elapsed_time_in_seconds = final_occupancy_branchings_time - initial_occupancy_branchings_time
    print('This target was analyzed in {} minutes and the result was {} branches.'.format(round(analysis_elapsed_time_in_seconds / 60, 1), len(branches_paths)))
    print("#############################################################################################################################################################################################")

    ##########################################################################################################################################################################################################################################################################################################################################################################################
    os.chdir(target_directory_path)

os.chdir(parent_directory)
print("#############################################################################################################################################################################################")
print("Copying results to parent directory.")
# Copy results to parent directory
results_list = necessary_functions.findfiles(target_directory_path, out_prefix)
for results_file in results_list:
    if 'het_atms.txt' not in results_file:  # we will keep the het atm file in the directory. we don't want to move out for now unless Anthony wants it Copied outside but not moved.
        shutil.copy(results_file, parent_directory)
print("#############################################################################################################################################################################################")

final_time = timeit.default_timer()
analysis_elapsed_time_in_seconds = final_time - initial_time
if round(analysis_elapsed_time_in_seconds / 60, 1) < 60:
    print('This target was analyzed in {} minutes.'.format(round(analysis_elapsed_time_in_seconds / 60, 1)))
else:
    print('This target was analyzed in {} hours.'.format(round(analysis_elapsed_time_in_seconds / 3600, 2)))
print("#############################################################################################################################################################################################")

