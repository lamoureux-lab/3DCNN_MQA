import os
import sys
import numpy as np
import __main__
__main__.pymol_argv = [ 'pymol', '-qc'] # Quiet and no GUI

import sys, time, os
import pymol
pymol.finish_launching()

#parameters
DATASETS_PATH = '/media/lupoglaz/ProteinsDataset/' #slash in the end required
DATASET_NAME = 'CASP11Stage2_SCWRL'
DATASET_DESCRIPTION = os.path.join(DATASETS_PATH, DATASET_NAME, 'Description')

def read_dataset_description(dataset_description_dir, dataset_description_filename, decoy_ranging = 'gdt-ts'):
	description_path= os.path.join(dataset_description_dir,dataset_description_filename)
	fin = open(description_path, 'r')
	proteins = []
	for line in fin:
		proteins.append(line.split()[0])
	fin.close()

	decoys = {}
	for protein in proteins:
		decoys_description_path = os.path.join(dataset_description_dir,protein+'.dat')
		fin = open(decoys_description_path,'r')
		description_line = fin.readline()

		decoy_path_idx = None
		decoy_range_idx = None
		for n,name in enumerate(description_line.split()):
			if name=='decoy_path':
				decoy_path_idx = n
			elif name==decoy_ranging:
				decoy_range_idx = n

		# print 'Decoys ranging column number = ', decoy_range_idx

		decoys[protein]=[]
		for line in fin:
			sline = line.split()
			decoys[protein].append((sline[decoy_path_idx], float(sline[decoy_range_idx])))
		fin.close()
	return proteins, decoys

def average_B_factors(target = 'T0776', decoy = 'BAKER-ROSETTASERVER_TS3', num_samples = 30):
	bfactors = {}
	for i in range(1,num_samples+1):
		filename = 'GradCAM/%s/rs%d_%s.pdb'%(target,i,decoy)
		with open(filename,'r') as f:
			for n, line in enumerate(f):
				sline = line.split()
				if not n in bfactors.keys():
					bfactors[n] = []
				try:
					bfactors[n].append(float(sline[-1]))
				except:
					pass
	filename_in = 'GradCAM/%s/rs%d_%s.pdb'%(target,1,decoy)
	filename_out = 'GradCAM/%s/average_%s.pdb'%(target,decoy)
	with open(filename_in,'r') as f_in:
		with open(filename_out,'w') as f_out:
			for n,line in enumerate(f_in):
				if len(bfactors[n])>0:
					f_out.write(line[:61] + '%.2f\n'%(np.mean(bfactors[n])))#/(np.max(all_b) - np.min(all_b))))

def get_scores(target = 'T0776', decoy = 'BAKER-ROSETTASERVER_TS3', num_samples = 30):
	scores = []
	print decoy
	for i in range(1,num_samples+1):
		if decoy[0:2]=='T0':
			filename = 'GradCAM/%s/rs%d_%s.pdb.pdb_score'%(target,i,decoy)
		else:
			filename = 'GradCAM/%s/rs%d_%s.pdb_score'%(target,i,decoy)
			# if not os.path.exists(filename):
			# 	filename = 'GradCAM/%s/rs%d_%s.pdb.pdb_score'%(target,i,decoy)
		with open(filename,'r') as f:
			score = float(f.read())
		scores.append(score)
	return scores

def get_B_factors(target = 'T0776', decoy = 'BAKER-ROSETTASERVER_TS3', num_samples = 30):
	os.system('th TorchGradCAM.lua -test_datasets_path %s -test_dataset_name %s -target %s -decoy %s -num_samples %s'%(DATASETS_PATH, DATASET_NAME, target, decoy, num_samples))

def process_structure(target, decoys = []):
	
	pymol.cmd.reinitialize()
	time.sleep(1.0)
	
	native = target+'.pdb'
	native_path = os.path.abspath('GradCAM/%s/average_%s.pdb'%(target, native))
	pymol.cmd.load(native_path, native)
	pymol.cmd.hide('lines', native)
	#pymol.cmd.show('cartoon', native)
	pymol.cmd.spectrum('b', 'rainbow', native)
	#pymol.cmd.do('set cartoon_transparency, 0.5, %s'%native)

	for name in decoys:
		path = os.path.abspath('GradCAM/%s/average_%s.pdb'%(target, name))
		pymol.cmd.load(path, name)
		pymol.cmd.hide('lines', name)
		pymol.cmd.spectrum('b', 'rainbow', name)
	
	for name in decoys:
		pymol.cmd.align(name, native)

	pymol.cmd.bg_color("white")
	pymol.cmd.center(native)
	pymol.cmd.zoom(native)

	proteins = decoys+[native]
	for n,name in enumerate(proteins):
		pymol.cmd.show('cartoon', name)
		if n>0: pymol.cmd.hide('cartoon', proteins[n-1])
		if name.find('.')!=-1:
			name = name[:name.find('.')]
		output_name = 'GradCAMOutput/%s/%s_%s.png'%(target, target, name)
		pymol.cmd.png(output_name, 800, 600, dpi=30, ray=1)
		time.sleep(1)

	

def sort_into_bins(decoys, n_bins = 4):
	paths, gdts = zip(*decoys)
	min_gdt, max_gdt = np.min(gdts), np.max(gdts)
	print 'Decoys gdt min and max:',min_gdt, max_gdt
	bins = [[] for i in range(n_bins)]
	for decoy in decoys:
		n_bin = int(n_bins * (decoy[1] - min_gdt)/(max_gdt-min_gdt+0.001))
		bins[n_bin].append(decoy)
	return bins

if __name__=='__main__':
	try:
		os.mkdir("GradCAMOutput")
	except:
		pass
	try:
		os.mkdir("GradCAM")
	except:
		pass

	generate = False
	process = False
	make_table = True

	proteins, decoys = read_dataset_description(DATASET_DESCRIPTION, 'datasetDescription.dat')
	if generate:
		for target in proteins:
			
			get_B_factors(target = target, decoy = '%s.pdb'%target, num_samples = 30)
			average_B_factors(target = target, decoy = '%s.pdb'%target, num_samples = 30)

			bins = sort_into_bins(decoys[target])
			decoys_names = []
			for sel_bin in bins:
				decoy_path = sel_bin[0][0]
				decoy_gdt = sel_bin[0][1]
				decoy_name = decoy_path[decoy_path.rfind('/')+1:]
				decoys_names.append(decoy_name)
				print 'Selected decoy', decoy_name, decoy_gdt

				get_B_factors(target = target, decoy = decoy_name, num_samples = 30)
				average_B_factors(target = target, decoy = decoy_name, num_samples = 30)
			
			if not os.path.exists('GradCAMOutput/%s'%target):
				os.mkdir('GradCAMOutput/%s'%target)
	if process:
		for target in proteins:
			bins = sort_into_bins(decoys[target])
			decoys_names = []
			for sel_bin in bins:
				decoy_path = sel_bin[0][0]
				decoy_gdt = sel_bin[0][1]
				decoy_name = decoy_path[decoy_path.rfind('/')+1:]
				decoys_names.append(decoy_name)
				print 'Selected decoy', decoy_name, decoy_gdt
			
			process_structure(target = target, decoys = decoys_names)
	
	pymol.cmd.quit()

	if make_table:
		num_proteins = [ (s,int(s[1:])) for s in proteins]
		num_proteins = sorted(num_proteins, key=lambda x: x[1])
		proteins, num_proteins = zip(*num_proteins)
		with open('GradCAMTable.tex', 'w') as fout:

			fout.write("""%\\documentclass[letter,10pt]{article}
%\\usepackage{graphicx}
%\\usepackage{capt-of}
%\\usepackage{tabularx}
%\\usepackage{ltxtable}
%\\begin{document}""")
			num_pages = int(len(proteins)/4)
			for page_num in range(0,num_pages):
				# if page_num==0:
# 					fout.write("""
# \\captionof{table}{Output of the Grad-CAM algorithm on the selected representative decoys in CASP11 Stage2.
# The interval of all the decoys GDT\\_TS was divided into four bins of equal size and random decoys from each bin were 
# selected. Then 30 samples with random rotations and translations for each selected decoys were used to generate Grad-CAM
# output dencity maps. Afterwards, each map was projected on the atoms of each sampled decoy. Finally, average values of the 
# projected values were calculated for each atom. These values were plotted using Pymol with the rainbow color scheme.}
# 	""")
				fout.write("""	
	\\begin{center}
	\\makebox[0pt][c]{
	\\hskip\\footskip
	\\begin{tabularx}{\paperwidth}{X*{5}{p{3.9cm}}}
					""")
				from_num = 4*page_num
				till_num = int(np.min([from_num+4, len(proteins)]))
				for target in proteins[from_num:till_num]:
					bins = sort_into_bins(decoys[target])
					fout.write("\\hline\n")

					target_bin = [ [target, 1.0]]
					all_bins = bins+[target_bin]
					#target name row
					for n,sel_bin in enumerate(all_bins):
						decoy_gdt = sel_bin[0][1]
						print decoy_gdt
						if n<(len(all_bins)-1):
							fout.write("\\tiny{%s} &"%target)
						else:
							fout.write("\\tiny{%s} \\\\\n"%target)
					#decoy name row
					for n,sel_bin in enumerate(bins+[target_bin]):
						decoy_path = sel_bin[0][0]
						decoy_name = decoy_path[decoy_path.rfind('/')+1:]
						decoy_name = decoy_name.replace('_','\\_')
						if n<(len(all_bins)-1):
							fout.write("\\tiny{%s} &"%(decoy_name))
						else:
							fout.write("\\tiny{%s} \\\\\n"%(decoy_name))
					#decoy GDT_TS row
					for n,sel_bin in enumerate(bins+[target_bin]):
						decoy_gdt = sel_bin[0][1]
						if n<(len(all_bins)-1):
							fout.write("\\tiny{GDT\\_TS = %.2f} &"%(decoy_gdt))
						else:
							fout.write("\\tiny{GDT\\_TS = %.2f} \\\\\n"%(decoy_gdt))
					#decoy score row
					for n,sel_bin in enumerate(bins+[target_bin]):
						decoy_path = sel_bin[0][0]
						decoy_name = decoy_path[decoy_path.rfind('/')+1:]
						idx = decoy_name.find('.')
						if idx!=-1:
							decoy_name = decoy_name[:idx]
						# decoy_name = decoy_name.replace('_','\\_')
						scores = get_scores(target = target, decoy = decoy_name, num_samples = 30)
						if n<(len(all_bins)-1):
							fout.write("\\tiny{Score = $%.2f \pm %.2f$} &"%(np.average(scores), np.std(scores)))
						else:
							fout.write("\\tiny{Score = $%.2f \pm %.2f$} \\\\\n"%(np.average(scores), np.std(scores)))
					#Figures row
					for n,sel_bin in enumerate(bins+[target_bin]):
						decoy_path = sel_bin[0][0]
						decoy_name = decoy_path[decoy_path.rfind('/')+1:]
						fout.write("\\begin{minipage}{\linewidth}\\includegraphics[width=\linewidth]{GradCAMOutput/%s/%s_%s}\\end{minipage}"%(target, target, decoy_name))
						if n<(len(all_bins)-1):
							fout.write("&")
						else:
							fout.write("\\\\\n")

				fout.write("""
\\end{tabularx}
%}
\\hskip\\headheight}
\\end{center}""")
			fout.write("""
%\\end{document}""")
