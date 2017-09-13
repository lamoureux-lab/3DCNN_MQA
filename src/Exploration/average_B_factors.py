import os
import sys
import numpy as np
import __main__
__main__.pymol_argv = [ 'pymol', '-qc'] # Quiet and no GUI

import sys, time, os
import pymol
pymol.finish_launching()

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

def get_B_factors(target = 'T0776', decoy = 'BAKER-ROSETTASERVER_TS3', num_samples = 30):
	os.system('th TorchGradCAM.lua -target %s -decoy %s -num_samples %s'%(target, decoy, num_samples))

def process_structure(target, decoys = []):
	
	pymol.cmd.reinitialize()
	time.sleep(1.0)
	
	native = target+'.pdb'
	native_path = os.path.abspath('GradCAM/%s/average_%s.pdb'%(target, native))
	pymol.cmd.load(native_path, native)
	pymol.cmd.hide('lines', native)
	pymol.cmd.show('cartoon', native)
	pymol.cmd.spectrum('b', 'rainbow', native)
	pymol.cmd.do('set cartoon_transparency, 0.5, %s'%native)

	for name in decoys:
		path = os.path.abspath('GradCAM/%s/average_%s.pdb'%(target, name))
		pymol.cmd.load(path, name)
		pymol.cmd.hide('lines', name)
		pymol.cmd.spectrum('b', 'rainbow', name)
	
	for name in decoys:
		pymol.cmd.align(name, native)

	pymol.cmd.bg_color("white")
	pymol.cmd.center(native)
	pymol.cmd.zoom('all')

	for n,name in enumerate(decoys):
		pymol.cmd.show('cartoon', name)
		if n>0: pymol.cmd.hide('cartoon', decoys[n-1])

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
	generate = False
	make_table = True

	proteins, decoys = read_dataset_description('/media/lupoglaz/ProteinsDataset/CASP11Stage2_SCWRL/Description', 'datasetDescription.dat')
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
			
			process_structure(target = target, decoys = decoys_names)
	pymol.cmd.quit()

	if make_table:
		num_proteins = [ (s,int(s[1:])) for s in proteins]
		num_proteins = sorted(num_proteins, key=lambda x: x[1])
		proteins, num_proteins = zip(*num_proteins)
		with open('GradCAMTable.tex', 'w') as fout:

			fout.write("""
%\\documentclass[letter,10pt]{article}
%\\usepackage{graphicx}
%\\usepackage{tabularx}
%\\usepackage{ltxtable}
%\\begin{document}""")
			num_pages = int(len(proteins)/4)
			for page_num in range(0,num_pages):
				fout.write("""	
\\begin{center}
\\makebox[0pt][c]{
\\hskip-\\footskip
\\begin{tabularx}{0.9\paperwidth}{X*{4}{p{4.5cm}}}

				""")
				from_num = 4*page_num
				till_num = int(np.min([from_num+4, len(proteins)]))
				for target in proteins[from_num:till_num]:
					bins = sort_into_bins(decoys[target])
					fout.write("\\hline\n")
					# fout.write("%s &"%target)
					for n,sel_bin in enumerate(bins):
						decoy_gdt = sel_bin[0][1]
						if n<(len(bins)-1):
							fout.write("%s &"%target)
						else:
							fout.write("%s \\\\\n"%target)
					for n,sel_bin in enumerate(bins):
						decoy_path = sel_bin[0][0]
						decoy_name = decoy_path[decoy_path.rfind('/')+1:]
						decoy_name = decoy_name.replace('_','\\_')
						if n<(len(bins)-1):
							fout.write("\\tiny{%s} &"%(decoy_name))
						else:
							fout.write("\\tiny{%s} \\\\\n"%(decoy_name))
					# fout.write("GDT\\_TS&")
					for n,sel_bin in enumerate(bins):
						decoy_gdt = sel_bin[0][1]
						if n<(len(bins)-1):
							fout.write("GDT\\_TS = %.2f &"%decoy_gdt)
						else:
							fout.write("GDT\\_TS = %.2f \\\\\n"%decoy_gdt)
					# fout.write("&")
					for n,sel_bin in enumerate(bins):
						decoy_path = sel_bin[0][0]
						decoy_name = decoy_path[decoy_path.rfind('/')+1:]
						fout.write("\\begin{minipage}{\linewidth}\\includegraphics[width=\linewidth]{GradCAMOutput/%s/%s_%s.png}\\end{minipage}"%(target, target, decoy_name))
						if n<(len(bins)-1):
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
