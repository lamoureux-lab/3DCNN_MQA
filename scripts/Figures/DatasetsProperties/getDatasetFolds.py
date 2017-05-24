import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors as m_colors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as colormap
import seaborn as sea
import cPickle as pkl
import sys
import os
import argparse
from operator import itemgetter
from Bio import SeqIO
from Bio import SearchIO
from Bio.SeqRecord import SeqRecord
from plotLengthDistributions import read_dataset_targets, read_sequences_data
from plotSequenceSimilarities import write_sequences, protein_vs_database
from tqdm import tqdm

import requests
import json
import xml.etree.ElementTree as ET
# from ete3 import Tree, TreeStyle, TextFace, faces, AttrFace, TreeStyle, NodeStyle, add_face_to_node

def get_pdb_codes(targets, tmp_output):
	hits = {}

	for qresult in SearchIO.parse(tmp_output, 'blast-xml'):
		ht = []
		for hit in qresult:
			for hsp in hit:
				ht.append((hsp.hit_id, hsp.evalue))
		
		hits_sorted = sorted(ht, key=lambda x: np.log(x[1]+0.0001))
		if len(hits_sorted)>0:
			hits[qresult[0][0].query_id]=hits_sorted[0][0]
		else:
			print 'Something is wrong'
		
	return hits

def get_target_to_pdb_corr():
	"""Download pdb_seqres.txt from ftp://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt"""
	if not os.path.exists('tmp/pdb_seq_db.phr'):
		os.system('makeblastdb -in %s -dbtype prot -out %s'%('tmp/pdb_seqres.txt', 'tmp/pdb_seq_db'))
	else:
		print 'PDB Database found'
	
	training_dataset_targets = read_dataset_targets('/home/lupoglaz/ProteinsDataset/CASP_SCWRL/Description', 'datasetDescription.dat')
	test_dataset_targets = read_dataset_targets('/home/lupoglaz/ProteinsDataset/CASP11Stage2_SCWRL/Description', 'datasetDescription.dat')

	protein_vs_database('tmp/train_seq.fasta', 'tmp/pdb_seq_db', 'tmp/train_vs_rcsb.dat', num_threads=10)
	protein_vs_database('tmp/test_seq.fasta', 'tmp/pdb_seq_db', 'tmp/test_vs_rcsb.dat', num_threads=10)

	hits_train = get_pdb_codes(training_dataset_targets, 'tmp/train_vs_rcsb.dat')
	hits_test = get_pdb_codes(training_dataset_targets, 'tmp/test_vs_rcsb.dat')
	casp2pdb = hits_train
	for key in hits_test:
		casp2pdb[key] = hits_test[key]
	with open("tmp/CASP2PDB.pkl",'w') as fout:
		pkl.dump(casp2pdb, fout)

def get_catch_link(query):
	rcsb_query = query.replace('_','.')
	print rcsb_query
	r = requests.get('http://www.rcsb.org/pdb/rest/das/pdbchainfeatures/features', params = {'segment':rcsb_query})
	root = ET.fromstring(r.text)
	for child in root:
		for chchild in child:
			for feature in chchild:
				if feature.attrib['label']=='CATH':
					for at in feature:
						if at.tag == 'LINK':
							print at.attrib['href']
							return at.attrib['href']

def parse_scope_data(filename):
	pdb2scope = {}
	with open(filename,'r') as fin:
		[fin.readline() for i in range(0,4)]
		for n, line in enumerate(fin):
			sline = line.split()
			query = sline[1]+'_'+sline[2][:-1]
			sline = sline[5].split(',')
			cl = sline[0][3:]
			cf = sline[1][3:]
			sf = sline[2][3:]
			fa = sline[3][3:]
			dm = sline[4][3:]
			sp = sline[5][3:]
			px = sline[6][3:]
			pdb2scope[query]= ( cl, cf, sf, fa, dm, sp, px )
	return pdb2scope

def parse_ecod_data(filename):
	pdb2ecod = {}
	with open(filename,'r') as fin:
		[fin.readline() for i in range(0,5)]
		for n, line in tqdm(enumerate(fin)):
			sline = line.split('\t')
			uid = sline[0]
			dom = sline[1]
			f_id = sline[3]
			pdb_key = sline[4]
			chain_key = sline[5]

			arch = sline[8]
			x_name = sline[9]
			h_name = sline[10]
			t_name = sline[11]
			f_name = sline[12]
			asm = sline[13]
			# print arch, x_name, h_name, t_name, f_name
			if not pdb_key in pdb2ecod.keys():
				pdb2ecod[pdb_key] = []
			pdb2ecod[pdb_key].append( ( chain_key, (f_id, arch, x_name, h_name, t_name, f_name) ) )

	return pdb2ecod

def get_ecod_classes():
	"""Download ECOD data from  http://prodata.swmed.edu/ecod/distributions/ecod.latest.domains.txt"""
	with open("tmp/CASP2PDB.pkl",'r') as fin:
		casp2pdb = pkl.load(fin)
	
	pdb2ecod = parse_ecod_data('tmp/ecod.latest.domains.txt')

	casp2ecod = {}
	for key in casp2pdb.keys():
		pdb_key = casp2pdb[key][:4]
		if pdb_key in pdb2ecod.keys():
			casp2ecod[key] = pdb2ecod[pdb_key]
			print key, pdb_key, 'present'
		else:
			print key, pdb_key, 'absent'

	with open("tmp/CASP2ECOD.pkl",'w') as fout:
		pkl.dump(casp2ecod, fout)


def plot_graph(f_graph, f_annot, output):
	os.system('graphlan_annotate.py --annot %s %s tmp/graph_1.xml'%(f_annot, f_graph))
	os.system('graphlan.py tmp/graph_1.xml %s --dpi 300 --size 3.5'%(output))

def plot_graph_wrong(training_dataset_targets, casp2ecod, casp2pdb):
	with open("tmp/graph.txt", 'w') as fout:
		for target in training_dataset_targets:
			if (not target in casp2pdb.keys()) or (not target in casp2ecod.keys()):
				continue
			pdb_key = casp2pdb[target]
			chain = pdb_key[-1:]
			
			for var in casp2ecod[target]:
				if var[0] == chain:
					arch_type = var[1][1].replace(' ','_')
					arch_set.add(arch_type)
					if not arch_type in arch_sorted.keys():
						arch_sorted[arch_type]=[]
					arch_sorted[arch_type].append('%s.%s\n'%(arch_type, var[1][0]))
			
		for key in arch_sorted.keys():
			print key
			for entry in arch_sorted[key]:
				print entry
				fout.write(entry)

	colors = ['#057005', '#009025', '#000025', '#050300', '#011240', '#050330', '#010101', '#500750', '#900950', '#031205', '#101150',
				'#20DD20', '#FFAD20', '#FF0020', '#00ADFF', '#ADAD20', '#DD1020', '#ff0000', '#ff4000', '#ff8000', '#ffbf00', '#ffff00',
				'#0080ff', '#0000ff']

	with open('tmp/annotation.txt','w') as fout:
		fout.write('%s\t%d\n'%('clade_marker_size', 0))
		fout.write('%s\t%f\n'%('branch_thickness', 0.3))
		fout.write('%s\t%f\n'%('clade_separation', 1.0))
		fout.write('%s\t%f\n'%('branch_bracket_depth', 0.5))
		for n, arch in enumerate(arch_set):
			fout.write('%s\t%s\t%s\n'%(arch, 'annotation_background_color', colors[n]))
			# fout.write('%s\t%s\t%s\n'%(arch, 'annotation', arch))
			


	plot_graph('tmp/graph.txt', 'tmp/annotation.txt', 'folds_grap.png')

def get_children_by_name(tree, name):
	res = []
	for ch in tree.children:
		# print ch.name
		if ch.name == name:
			res.append(ch)
	return res

def add_branch(tree, branch, style=None, color_dict = None):
	sbranch = branch.split('.')
	# print sbranch
	root = tree.get_tree_root()
	
	if len(sbranch)==5:
		sbranch = sbranch[:-1]
	else:
		sbranch = sbranch

	for depth, ch in enumerate(sbranch):
		result = get_children_by_name(root, name = ch)
		if len(result)==0:
			root = root.add_child(name = ch)
		else:
			root = result[0]
		if not style is None:
			root.set_style(style)
		if not color_dict is None:
			root.set_style(color_dict[sbranch[0]])

def color_branch(tree, branch, style=None, color_dict = None):
	sbranch = branch.split('.')
	# print sbranch
	root = tree.get_tree_root()
	
	if len(sbranch)==5:
		sbranch = sbranch[:-1]
	else:
		sbranch = sbranch

	for depth, ch in enumerate(sbranch):
		result = get_children_by_name(root, name = ch)
		if len(result)==0:
			break
		else:
			root = result[0]
		if not style is None:
			root.set_style(style)
		if not color_dict is None:
			root.set_style(color_dict[sbranch[0]])


LOWERCASE, UPPERCASE = 'x', 'X'
def triplet(rgb, lettercase=LOWERCASE):
    return format(rgb[0]<<16 | rgb[1]<<8 | rgb[2], '06'+lettercase)

if __name__=='__main__':
	prepare = False
	if prepare:
		get_target_to_pdb_corr()
		get_ecod_classes()
		sys.exit()
		
	
	
	
	norm = m_colors.Normalize(vmin=0.0, vmax=20.0)
	cmap = colormap.get_cmap(name = "Pastel1")
	colors = [m_colors.rgb2hex(cmap(norm(i))) for i in range(0,20)]

	arch_types = ['a+b_four_layers', 'alpha_duplicates_or_obligate_multimers', 'beta_meanders', 'a+b_two_layers', 'alpha_superhelices', 
	'a+b_duplicates_or_obligate_multimers', 'alpha_bundles', 'a/b_three-layered_sandwiches', 'extended_segments', 'alpha_complex_topology', 
	'beta_barrels', 'a+b_complex_topology', 'a+b_three_layers', 'few_secondary_structure_elements', 'mixed_a+b_and_a/b', 'alpha_arrays', 
	'beta_duplicates_or_obligate_multimers', 'beta_complex_topology', 'beta_sandwiches', 'a/b_barrels']
	colors_dict_training = {}
	for n,arch in enumerate(arch_types):
		ns = NodeStyle()
		ns["bgcolor"] = colors[n]
		ns["size"] = 0
		ns["vt_line_color"] = "#050505"
		ns["hz_line_color"] = "#050505"
		ns["vt_line_width"] = 10
		ns["hz_line_width"] = 10
		ns["vt_line_type"] = 0 # 0 solid, 1 dashed, 2 dotted
		ns["hz_line_type"] = 0
		colors_dict_training[arch] = ns

	colors_dict_test = {}
	for n,arch in enumerate(arch_types):
		ns = NodeStyle()
		ns["bgcolor"] = colors[n]
		ns["size"] = 0
		ns["vt_line_color"] = "#AAAAAA"
		ns["hz_line_color"] = "#AAAAAA"
		ns["vt_line_width"] = 10.0
		ns["hz_line_width"] = 10.0
		ns["vt_line_type"] = 0 # 0 solid, 1 dashed, 2 dotted
		ns["hz_line_type"] = 0
		colors_dict_test[arch] = ns
	
	
	with open("tmp/CASP2ECOD.pkl",'r') as fin:
		casp2ecod = pkl.load(fin)
	with open("tmp/CASP2PDB.pkl",'r') as fin:
		casp2pdb = pkl.load(fin)
	# print casp2ecod

	training_dataset_targets = read_dataset_targets('/home/lupoglaz/ProteinsDataset/CASP_SCWRL/Description', 'datasetDescription.dat')
 	test_dataset_targets = read_dataset_targets('/home/lupoglaz/ProteinsDataset/CASP11Stage2_SCWRL/Description', 'datasetDescription.dat')
	
	t = Tree()
	for target in test_dataset_targets:
		if (not target in casp2pdb.keys()) or (not target in casp2ecod.keys()):
			continue
		pdb_key = casp2pdb[target]
		chain = pdb_key[-1:]
		for var in casp2ecod[target]:
			if var[0] == chain:
				arch_type = var[1][1].replace(' ','_')
				branch = '%s.%s'%(arch_type, var[1][0])
				add_branch(t, branch, color_dict=colors_dict_test)

	for target in training_dataset_targets:
		if (not target in casp2pdb.keys()) or (not target in casp2ecod.keys()):
			continue
		pdb_key = casp2pdb[target]
		chain = pdb_key[-1:]
		for var in casp2ecod[target]:
			if var[0] == chain:
				arch_type = var[1][1].replace(' ','_')
				branch = '%s.%s'%(arch_type, var[1][0])
				color_branch(t, branch, color_dict=colors_dict_training)

	
	for ch_n, child in enumerate(t.get_tree_root().children):
		node = ((child.children[-1]).children[-1]).children[-1]
		str_name = child.name.replace('_',' ')
		
		face = TextFace(str_name, fsize=32, fgcolor='black',tight_text=True, bold=False)
		node.add_face(face, 0, position='aligned')
		
	
	
	ts = TreeStyle()
	ts.allow_face_overlap = False
	ts.show_scale = False
	ts.scale=100.0
	ts.mode = "c"
	ts.show_leaf_name = False
	ts.root_opening_factor = 0.1
	ts.force_topology=True
	
	# ts.arc_start = -180 # 0 degrees = 3 o'clock
	# ts.arc_span = 180
	t.render("folds_graph.png", w=300, units="mm", tree_style=ts)

