#!/usr/bin/python
from matplotlib import pyplot as p
import scipy as sp
import numpy as np
import sys
import csv
import glob

def get_graphs(ms):
	data = [] 
	head = []
	for f in ms:
		 head = f[0]
		 data.append(f[-1])
	tci_index = np.where(head == 'TCITranslationTime')
	llvm_index = np.where(head == 'LLVMTranslationTime')
	idx = []
	idx.append(tci_index[0])
	idx.append(llvm_index[0])
	idx.append(np.where(head == 'SolverTime')[0])
	idx.append(np.where(head == 'SymbolicModeTime')[0])
	symb_time = data[0][np.where(head == 'SymbolicModeTime')[0]]
	num = data[0][np.where(head == 'NumStates')[0]]
	print data[0][np.where(head == 'WallTime')[0]]
	print data[0][np.where(head == 'ConcreteModeTime')[0]]
	print data[0][np.where(head == 'ForkTime')[0]]
	print data[0][np.where(head == 'SolverTime')[0]]
	colors = ['r','k','m','b']
	vals = []
	for j in range(len(ms)):
		vals.append([np.asscalar(np.float64(data[j][i])) for i in idx])
	print symb_time
	N = len(ms)
	vals = zip(*vals)
	ind = np.arange(N)
	for n in np.arange(len(vals)):
		print vals[n] 
	width = 0.3
	p1 = p.bar(ind,vals[0],width,color=colors[0])
	p2 = p.bar(ind,vals[1],width,color=colors[1],bottom=vals[0])
	bottom2 = [vals[0][l]+vals[1][l] for l in range(len(vals[0]))]
	p3 = p.bar(ind,vals[2],width,color=colors[2],bottom=bottom2)
	bottom3 = [vals[0][l]+vals[1][l]+vals[2][l] for l in range(len(vals[0]))]
	p4 = p.bar(ind,vals[3],width,color=colors[3],bottom=bottom3)
	#p.xticks(ind+width/2., ('S1', 'S2'))
	#p.yticks(np.arange(0,2,10))
	p.legend((p1[0], p2[0], p3[0],p4[0]), ( str(head[idx[0]]) , str(head[idx[1]]) , str(head[idx[2]]), str(head[idx[3]])), loc='best')
	p.show()

def process_data(args):
	ms = []	
	for i in range(1,len(args)):
		try:
			ms.append(np.genfromtxt(args[i],delimiter=",",dtype=None))
		except IOError:
			print "Cannot open the file"
			exit(0)
	#Get graphs
	get_graphs(ms)

if __name__ == '__main__':
	if len(sys.argv) == 1:
		files = glob.glob("stats/*_stat")
		process_data(files)	
	else:
		print "read_data.py <directory_stats>"
		sys.exit(0)
	

	#Parse data
		
