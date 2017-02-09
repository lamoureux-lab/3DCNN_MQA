import os
import sys
import cPickle as pkl
import numpy as np
from tqdm import tqdm
import requests
from xml.etree.ElementTree import fromstring
import argparse
from Bio import ExPASy
from Bio import SwissProt
import urllib2

#http addresses for getting the keys
pdb_mapping_url = 'http://www.rcsb.org/pdb/rest/das/pdb_uniprot_mapping/alignment'
uniprot_url = 'http://www.uniprot.org/uniprot/{}.xml'
unprot_pfam_url = 'http://pfam.xfam.org/protein/{}.xml'

PDB_PFAM_MAPPING_URL = 'ftp://ftp.ebi.ac.uk/pub/databases/Pfam/mappings/pdb_pfam_mapping.txt'
PDB_PFAM_MAPPING_PATH = 'pdb_pfam_mapping.txt'
PDB_SCOP_MAPPING_PATH = 'pdb_scop_mapping.txt'
PDB_PFAM_MAPPING_PROCESSED_PATH = 'pdb_pfam_mapping.pkl'
PDB_SCOP_MAPPING_PROCESSED_PATH = 'pdb_scop_mapping.pkl'

def get_uniprot_accession_id(response_xml):
	"""returns uniprot id in the xml response"""
	root = fromstring(response_xml)
	return next(
		el for el in root.getchildren()[0].getchildren()
		if el.attrib['dbSource'] == 'UniProt'
	).attrib['dbAccessionId']

def get_uniprot_seq_pos(response_xml, pdb_query, uniprot_query):
	"""Finds position of uniprot entry on a pdb chain"""
	root = fromstring(response_xml)
	
	for el_up in  root.getchildren()[0].getchildren():
		for el in  el_up.getchildren():
			if not 'intObjectId' in el.attrib:
				continue
			if el.attrib['intObjectId'] == pdb_query:
				#print el.attrib
				pdb_span_start = el.attrib['start']
				pdb_span_end = el.attrib['end']
			if el.attrib['intObjectId'] == uniprot_query:
				#print el.attrib
				uniprot_span_start = el.attrib['start']
				uniprot_span_end = el.attrib['end']

	return (pdb_span_start, pdb_span_end), (uniprot_span_start,uniprot_span_end)


def preprocess_pdb_to_pfam_database():
	"""Downloads pdb2pfam database and converts it to dictionary"""
	if os.path.exists(PDB_PFAM_MAPPING_PROCESSED_PATH):
		return 
	
	if not os.path.exists(PDB_PFAM_MAPPING_PATH):
		os.system('wget %s'%PDB_PFAM_MAPPING_URL)
		os.system('mv %s %s'%(PDB_PFAM_MAPPING_URL[PDB_PFAM_MAPPING_URL.rfind('/')+1:], PDB_PFAM_MAPPING_PATH))
	
	data = {}
	pdb2pfamF = open(PDB_PFAM_MAPPING_PATH, 'r')
	pdb2pfamF.readline()
	for line in tqdm(pdb2pfamF):
		sline = line.split()
		# print sline
		query_id = sline[0]+'_'+sline[1]
		if not query_id in data.keys():
			data[query_id] = []
		data[query_id].append( (sline[2], sline[3], sline[4], float(sline[-1])) )
	pdb2pfamF.close()

	fout = open(PDB_PFAM_MAPPING_PROCESSED_PATH,'w')
	pkl.dump(data, fout)
	fout.close()
	return


def preprocess_pdb_to_scop_database():
	"""Downloads pdb2pfam database and converts it to dictionary"""
	if os.path.exists(PDB_SCOP_MAPPING_PROCESSED_PATH):
		return 
	
	# if not os.path.exists(PDB_PFAM_MAPPING_PATH):
	# 	os.system('wget %s'%PDB_PFAM_MAPPING_URL)
	# 	os.system('mv %s %s'%(PDB_PFAM_MAPPING_URL[PDB_PFAM_MAPPING_URL.rfind('/')+1:], PDB_PFAM_MAPPING_PATH))
	
	data = {}
	pdb2pfamF = open(PDB_SCOP_MAPPING_PATH, 'r')
	[pdb2pfamF.readline() for i in range(0,4)]
	for line in tqdm(pdb2pfamF):
		sline = line.split()
		# print sline
		query_id = sline[1]+'_'+sline[2][:sline[2].find(':')]
		if not query_id in data.keys():
			data[query_id] = []
		data[query_id].append( (sline[3], sline[4], sline[5]) )
	pdb2pfamF.close()

	fout = open(PDB_SCOP_MAPPING_PROCESSED_PATH,'w')
	pkl.dump(data, fout)
	fout.close()
	return


def map_pdb_to_pfam_family(query_id, res_start=None, res_end=None, data=None):
	"""Tries to find pfam family that corresponds to given pdb+chain"""
	if data is None:
		fin = open(PDB_PFAM_MAPPING_PROCESSED_PATH,'r')
		data = pkl.load(fin)
		fin.close()

	if not query_id in data.keys():
		return 'None'
	
	max_overlap = 0
	max_candidate = ''
		
	if (res_start is None) or (res_end is None):
		for candidate in data[query_id]:
			overlap = int(candidate[1]) - int(candidate[0])
			if overlap>max_overlap:
				max_candidate = candidate[2]
				max_overlap = overlap
	else:
		for candidate in data[query_id]:
			overlap = min(int(candidate[1]),res_end) - max(int(candidate[0]),res_start)
			if overlap>max_overlap:
				max_candidate = candidate[2]
				max_overlap = overlap

	return max_candidate


def map_pdb_to_scop_family(query_id, data=None):
	"""Tries to find pfam family that corresponds to given pdb+chain"""
	if data is None:
		fin = open(PDB_SCOP_MAPPING_PROCESSED_PATH,'r')
		data = pkl.load(fin)
		fin.close()
	
	if not query_id in data.keys():
		return None
	
	return data[query_id][0][0]
	
		

def map_pdb_to_uniprot(query_id):
	"""Tries to find uniprot entry that corresponds to given pdb+chain"""
	pdb_query = query_id.replace('_','.')
	pdb_mapping_response = requests.get(
		pdb_mapping_url, params={'query': pdb_query}
	).text
	uniprot_id = get_uniprot_accession_id(pdb_mapping_response)

	pdb_span, uniprot_span = get_uniprot_seq_pos(pdb_mapping_response, pdb_query, uniprot_id)

	return {
		'pdb_id': query_id,
		'uniprot_id': uniprot_id,
		'pdb_span' : pdb_span,
		'uniprot_span' : uniprot_span
	}


def get_keys_information(dataset_sequences, output_path):
	"""Obtain keys in the format of previos dataset:
	family$uniprot_id$sequence$uniprot_start$uniprot_end"""
	data = get_fasta_seq(dataset_sequences)

	fin = open(PDB_PFAM_MAPPING_PROCESSED_PATH,'r')
	data_pdb2pfam = pkl.load(fin)
	fin.close()

	formatted_keys = []
	for query_id,seq in tqdm(data):
		try:
			mapping = map_pdb_to_uniprot(query_id)
			family = map_pdb_to_pfam_family(query_id,int(mapping['pdb_span'][0]),int(mapping['pdb_span'][1]))
			formatted_key = '%s$%s$%s$%s_%s'%(family,mapping['uniprot_id'],seq,mapping['uniprot_span'][0],mapping['uniprot_span'][1])
			formatted_keys.append(formatted_key)
		except:
			logging.info('Problem in pdb: '+query_id)
			formatted_keys.append('error')

	fout = open(output_path,'w')
	pkl.dump(formatted_keys, fout)
	fout.close()

def getFold(scop_entry):
	import Bio.SCOP as scop
	try:
		handle = scop.search(key=scop_entry)
	except:
		return None
	text = handle.read()
	# print text
	strBeg = text.find("Class:")
	classHTMLText =  text[ strBeg: strBeg + text[strBeg:].find("</a>")]
	classText = classHTMLText[classHTMLText.find(">")+1:]
	#print classText

	strBeg = text.find("Fold:")
	foldHTMLText =  text[ strBeg: strBeg + text[strBeg:].find("</a>")]
	foldText = foldHTMLText[foldHTMLText.find(">")+1:]
	#print foldText

	strBeg = text.find("Superfamily:")
	famHTMLText =  text[ strBeg: strBeg + text[strBeg:].find("</a>")]
	famText = famHTMLText[famHTMLText.find(">")+1:]
	#print famText
	if len(foldText)==0 and len(classText)==0 and len(famText)==0:
		return None
	return foldText, classText, famText

if __name__=='__main__':
	parser = argparse.ArgumentParser(prog='ProteinMapping', 
									formatter_class=argparse.RawDescriptionHelpFormatter,
									description="""\
									Maps protein id_chain to families and uniprot entries.
									""")
	parser.add_argument('--query_list_filename', metavar='query_list_filename', type=str, 
				   help='List of protein queries', default='native_queries.pkl')
	parser.add_argument('-keywords', metavar='keywords', type=bool, 
				   help='Generate keywords', default=False)
	parser.add_argument('-folds', metavar='folds', type=bool, 
				   help='Generate folds', default=False)
	args = parser.parse_args()

	with open(args.query_list_filename,"r") as inp:
		queries = pkl.load(inp)
	
	if args.keywords:
		mapping  = {}
		for query in tqdm(queries):
			try:
				mapp = map_pdb_to_uniprot(query)
				handle = ExPASy.get_sprot_raw(mapp['uniprot_id'])
				record = SwissProt.read(handle)
				mapping[query] = (record.organism, record.keywords)
			except:
				print 'Skipping ', query
			
		with open('native_queries_mapping.pkl','w') as f:
			pkl.dump(mapping,f)
	
	if args.folds:
		preprocess_pdb_to_scop_database()
		with open(PDB_PFAM_MAPPING_PROCESSED_PATH,'r') as fin:
			data = pkl.load(fin)

		with open(PDB_SCOP_MAPPING_PROCESSED_PATH,'r') as fin:
			data = pkl.load(fin)
		
		folds = {}
		for query in tqdm(queries):
			mapp = map_pdb_to_scop_family(query[:-2].lower()+query[-2:], data=data)
			if mapp is None:
				continue
			fold = getFold(mapp)
			if fold is None:
				continue
			folds[query]=fold
			# print query,':',fold
		
		with open('native_queries_folds.pkl','w') as f:
			pkl.dump(folds,f)
		
