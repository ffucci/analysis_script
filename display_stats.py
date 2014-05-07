#!/usr/bin/python
from matplotlib import pyplot as p
import scipy as sp
import numpy as np
import sys
import csv
import glob
from sets import Set
from mpltools import color

global colors,pvalues
pvalues = np.logspace(-1,0,4)
parameter_range = (pvalues[0],pvalues[-1])
colors = color.color_mapper(parameter_range,cmap='BuPu',start=0.2)


def autolabel_float(ax,rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.07*height, '%1.1f'%float(height),
                ha='center', va='bottom')


def autolabel(ax,rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                ha='center', va='bottom')

def first_graph(ind,vals1,vals2,width,drivers):
        global head,idx
        fig,ax = p.subplots()
        ax.set_xticks(ind+width)
        ax.set_xticklabels(drivers,fontsize=10,rotation=0)

        ###########################
        # First bar TCI Translation Time
        ###########################

        p1 = p.bar(ind,vals1[0],width,color=colors(pvalues[0]))
        p11 = p.bar(ind+width,vals2[0],width,color=colors(pvalues[0]))

        ###########################
        # Second bar LLVM Translation Time
        ###########################

        p2 = p.bar(ind,vals1[1],width,color=colors(pvalues[1]),bottom=vals1[0])
        p21 = p.bar(ind+width,vals1[1],width,color=colors(pvalues[1]),bottom=vals2[0])
        bottom2_1 = [vals1[0][l]+vals1[1][l] for l in range(len(vals1[0]))]
        bottom2_2 = [vals2[0][l]+vals2[1][l] for l in range(len(vals2[0]))]

        ###########################
        # Third bar Solver Time
        ###########################
        p3 = p.bar(ind,vals1[2],width,color=colors(pvalues[2]),bottom=bottom2_1)
        p31 = p.bar(ind+width,vals2[2],width,color=colors(pvalues[2]),bottom=bottom2_2)
        bottom3_1 = [vals1[0][l]+vals1[1][l]+vals1[2][l] for l in range(len(vals1[0]))]
        bottom3_2 = [vals2[0][l]+vals2[1][l]+vals2[2][l] for l in range(len(vals2[0]))]

        p4 = p.bar(ind,vals1[3],width,color=colors(pvalues[3]),bottom=bottom3_1)
        p41 = p.bar(ind+width,vals2[3],width,color=colors(pvalues[3]),bottom=bottom3_2)
        #print files,bb_number

        #bottom4 = [vals[0][l]+vals[1][l]+vals[2][l]+vals[3][l] for l in range(len(vals[0]))]
            #p5 = p.bar(ind,vals[4],width,color=colors[4],bottom=bottom4)
        #p.xticks(ind+width/2., ('S1', 'S2'))
        #p.yticks(np.arange(0,2,10))
        autolabel_float(ax,p3)
        autolabel_float(ax,p31)
        p.legend((p1[0], p2[0], p3[0],p4[0]), ( str(head[idx[0]]) , str(head[idx[1]]) , str(head[idx[2]]), str(head[idx[3]])), loc='best')
        fig.suptitle('Performance evaluation', fontsize=16)
        p.xlabel('Driver', fontsize=14)
        p.ylabel('Time spent (s)', fontsize=14)
        ax.yaxis.grid(True)
        p.savefig('perfom_eval.png')


def second_graph(files,drivers,ind,width):
        global ind_1,ind_2, c_paths
        f_clust1 = [s for s in files if "tci" in s]
        f_clust2 = [s for s in files if "klee" in s]
        fig2,a2 = p.subplots()
        p.xlabel('Driver' ,fontsize=14)
        a2.set_xticks(ind+width)
        a2.set_xticklabels(drivers,fontsize=10,rotation=0)
        p.ylabel('# paths completed')
        c_paths1 = [c_paths[ind_1[j]] for j in range(len(ind_1))]
        c_paths2 = [c_paths[ind_2[j]] for j in range(len(ind_2))]
        g2 = p.bar(ind,c_paths1,width,facecolor=colors(pvalues[0]),edgecolor='white')
        g3 = p.bar(ind+width,c_paths2,width,facecolor=colors(pvalues[-1]),edgecolor='white')
        autolabel(a2,g2)
        autolabel(a2,g3)
        p.legend((g2[0], g3[0]), ( "KLEE" , "TCI"), loc='best')
        a2.yaxis.grid(True)

def third_graph(vals,drivers,ind,width):
        global ind_1,ind_2
        fig3,a3 = p.subplots()
        act_width = 3*width
        alpha = 1.1
        tran_times = vals[0] + vals[1]
        #The first is LLVM the second TCI
        tran_times1 = [tran_times[ind_1[i]] for i in range(len(ind_1))]
        tran_times2 = [tran_times[ind_2[i]] for i in range(len(ind_2))]
        ratio_time = np.array(tran_times1)/np.array(tran_times2)
        q1 = p.bar(ind_1+act_width/2,ratio_time,act_width,facecolor=colors(pvalues[-2]),edgecolor='white')
        #q2 = p.bar(ind+width,tran_times2,width,color=colors(pvalues[-2]))
        a3.set_xticks(ind_1+act_width)
        a3.set_xticklabels(drivers,fontsize=10,rotation=0)
        p.xlabel('Driver',fontsize=14)
        p.ylabel('Ratio of time spent translating LLVM code vs TCI code')
        p.ylim((0,alpha*max(ratio_time)))
        autolabel_float(a3,q1)
        a3.yaxis.grid(True)
        p.savefig('')



def get_graphs_tci(files,ms):
        data = []
        global head,idx
        head = []
        mapping = {}
        files = [ s.replace('stats/','') for s in files]
        print "Map: ", mapping
        i = 0
        for f in ms:
            head = f[0]
            data.append(f[-1])
            mapping[files[i]] = data[i]
            i = i+1

        files = sorted(mapping)
        tci_index = np.where(head == 'TCITranslationTime')
        llvm_index = np.where(head == 'LLVMTranslationTime')
        idx = []
        idx.append(tci_index[0])
        idx.append(llvm_index[0])
        idx.append(np.where(head == 'SolverTime')[0])
        idx.append(np.where(head == 'SymbolicModeTime')[0])
        #idx.append(np.where(head == 'ResolveTime')[0])
        wall_time = []
        for i in range(len(files)):
            wall_time.append(np.asscalar(np.float64(data[i][np.where(head == 'WallTime')])))
        print "WallTime = ",wall_time
        names = ["TCITranslationTime","LLVMTranslationTime","SolverTime","SymbolicModeTime"]
        #colors = ['r','k','m','b']
        #vals = np.matrix([])
        vals = np.zeros([len(ms), len(idx)])
        print "Size: ", vals.shape
        print vals
        bb_number = []
        global c_paths
        c_paths = []
        num_paths = np.where(head == 'CompletedPaths')[0]
        index_num = np.where(head == 'TranslationBlocks')[0]
        for j in range(len(files)):
            temp = mapping[files[j]]
            c_paths.append(temp[num_paths])
            bb_number.append(temp[index_num])
            #vals.append([np.asscalar(np.float64(data[j][i])) for i in idx])
            line = [np.asscalar(np.float64(temp[i])) for i in idx]
            vals[j] = line
        c_paths = [np.asscalar(np.float64(c_paths[i])) for i in range(len(c_paths))]
        print "Completed paths: ", c_paths

        percentage = []
        name_drivers = Set()
        print files
        for i in range(len(files)):
            name_drivers.add(files[i].split('_')[0])

        print "Name drivers: ",name_drivers
        print "Mapping: ", mapping
        N = len(name_drivers)
        #print vals
        #vals = zip(*vals)

        vals = vals.transpose()
        global ind_1,ind_2
        ind_1 = np.array([int(2*i) for i in range(len(ms)/2)])
        ind_2 = np.array([int(2*i + 1) for i in range(len(ms)/2)])

        print "Ind 1: ",ind_1
        print "Ind 2: ",ind_2
        print "Vals: ",vals
        vals1 = []
        vals2 = []

        for i in range(0,len(vals)):
            print i
            temp = mapping[files[i]]
            vals1.append([vals[i].tolist()[ind_1[j]] for j in range(len(ind_1))])
            vals2.append([vals[i].tolist()[ind_2[j]] for j in range(len(ind_2))])

        ind = np.arange(N)
        print "Indices: ", ind
        for n in np.arange(len(vals)):
            percentage.append([100*np.asscalar(np.float64(vals[n][l]))/wall_time[n] for l in range(len(vals[n]))])
            print vals[n]
        width = 0.3
        #Generate drivers list
        drivers = list(name_drivers)
        drivers.sort()

        #first_graph(ind,vals1,vals2,width,drivers)
        #Number of paths graph
        second_graph(files,drivers,ind,width)
        third_graph(vals,drivers,ind,width)


def process_data(args,b):
        ms = []
        for i in range(len(args)):
            try:
                ms.append(np.genfromtxt(args[i],delimiter=",",dtype=None))
            except IOError:
                print "Cannot open the file"
                exit(0)
        #Get graphs
        get_graphs_tci(args,ms)

if __name__ == '__main__':
        if len(sys.argv) == 1:
            files = glob.glob("stats/*_stat")
            process_data(files,True)
            #files = glob.glob("stats/*_klee_stat")
            #process_data(files,False)
        else:
            print "read_data.py <directory_stats>"
            sys.exit(0)
        #Parse data
		
