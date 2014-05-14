#!/usr/bin/python
from matplotlib import pyplot as p
import scipy as sp
import numpy as np
import sys
import csv
import glob
from sets import Set
from mpltools import color
from mpltools import layout
import os

global colors,pvalues
pvalues = np.logspace(-1,0,6)
parameter_range = (pvalues[0],pvalues[-1])
colors = color.color_mapper(parameter_range,cmap='BuPu',start=0.2)
colors2 = color.color_mapper(parameter_range,cmap='PuOr',start=0.2)


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

def autolabel_pos(ax,rects,pos,perc,offset=None):
    if offset == None:
        offset = np.zeros(len(rects))
    # attach some text labels
    for i,rect in enumerate(rects):
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., pos[i] - height/2 + offset[i], '%1.1f'%float(perc[i]) + '%',
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


def second_graph(files,drivers,ind,width,wall_time):
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
        file_name = "num_paths_" + str(int(wall_time/60));
        p.savefig("results/"+file_name)

def third_graph(vals,drivers,ind,width,wall_time):
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
        file_name = "ratio_translation_time_llvm_tci_" + str(int(wall_time/60));
        p.savefig("results/"+file_name)

def fourth_graph(symb_tb,drivers,ind_1,ind_2,width,wall_time):
        figsize = layout.figaspect(1.4)
        fig3,a3 = p.subplots(figsize=figsize)
        act_width = 2*width
        alpha = 1.3
        #The first is KLEE the second TCI
        klee_num_tb = [np.float64(symb_tb[ind_1[i]][0]) for i in range(len(ind_1))]
        tci_num_tb = [np.float64(symb_tb[ind_2[i]][0]) for i in range(len(ind_2))]
        print "KLEE tb:", klee_num_tb
        ratio_tbs = np.array(tci_num_tb)/np.array(klee_num_tb)
        print ratio_tbs
        q1 = p.bar(ind_1+act_width/2,ratio_tbs,act_width,facecolor=colors(pvalues[-2]),edgecolor='white')
        #q2 = p.bar(ind+width,tran_times2,width,color=colors(pvalues[-2]))
        a3.set_xticks(ind_1+act_width)
        a3.set_xticklabels(drivers,fontsize=10,rotation=0)
        p.xlabel('Driver',fontsize=14)
        p.ylabel('Ratio of number of translation blocks executed TCI vs KLEE')
        p.ylim((0,alpha*max(ratio_tbs)))
        autolabel_float(a3,q1)
        a3.yaxis.grid(True)
        file_name = "ratio_translation_blocks_klee_tci_" + str(int(wall_time/60));
        p.savefig("results/"+file_name)

def breakdown_graph(ind,vals,mapping,files,tci_time,klee_time,width,drivers,wall_time):
        tci_time = np.array([np.float64(i[0]) for i in tci_time])
        klee_time = np.array([np.float64(i[0]) for i in klee_time])
        vals1 = []
        vals2 = []
        for i in range(0,len(vals)):
            temp = mapping[files[i]]
            vals1.append([vals[i].tolist()[ind_1[j]] for j in range(len(ind_1))])
            vals2.append([vals[i].tolist()[ind_2[j]] for j in range(len(ind_2))])
        figsize  = layout.figaspect(1.2)
        fig,ax = p.subplots(figsize=figsize)
        ax.set_xticks(ind+width)
        ax.set_xticklabels(drivers,fontsize=10,rotation=0)
        psolv = p.bar(ind,vals1[2],width,color=colors(pvalues[0]),edgecolor='white')
        psolv2 = p.bar(ind+width,vals2[2],width,color=colors2(pvalues[0]),edgecolor='white')
        perc1 = 100*(vals1[2]/np.array(wall_time)[0:2:len(wall_time)])
        perc2 = 100*(vals2[2]/np.array(wall_time)[1:2:len(wall_time)+1])
        autolabel_pos(ax,psolv,vals1[2],perc1)
        autolabel_pos(ax,psolv2,vals2[2],perc2)
        print "Vals2: ",vals[2]
        print "TCI time:",tci_time
        print "KLEE time:",klee_time
        #Separate timing
        tci1 = [tci_time[ind_1[j]] for j in range(len(ind_1))]
        tci2 = [tci_time[ind_2[j]] for j in range(len(ind_2))]
        klee1 = [klee_time[ind_1[j]] for j in range(len(ind_1))]
        klee2 = [klee_time[ind_2[j]] for j in range(len(ind_2))]
        #Create bar
        #ptci = p.bar(ind,tci1,width,color=colors(pvalues[1]),bottom=vals1[2],edgecolor='white')
        #ptci2 = p.bar(ind+width,tci2,width,color=colors(pvalues[1]),bottom=vals2[2],edgecolor='white')

        #perc1 = 100*(tci1/np.array(wall_time)[0:2:len(wall_time)])
        #perc2 = 100*(tci2/np.array(wall_time)[1:2:len(wall_time)+1])
        #perc2 = 100*(klee_time[1:2:len(wall_time)+1]/np.array(wall_time)[1:2:len(wall_time)+1])
        #autolabel_pos(ax,ptci,tci1,perc1,vals1[2])
        #autolabel_pos(ax,ptci2,tci2,perc2,vals2[2])
        int_time = np.array(klee1)+np.array(tci1)
        int_time2 = np.array(klee2)+np.array(tci2)
        pint = p.bar(ind,int_time,width,color=colors(pvalues[2]),bottom=np.array(vals1[2]),edgecolor='white')
        pint2 = p.bar(ind+width,int_time2,width,color=colors2(pvalues[2]),bottom=np.array(vals2[2]),edgecolor='white')
        perc1 = 100*(int_time/np.array(wall_time)[0:2:len(wall_time)])
        perc2 = 100*(int_time2/np.array(wall_time)[1:2:len(wall_time)+1])
        print np.array(vals2[2])+ np.array(tci2)
        autolabel_pos(ax,pint,int_time,perc1,np.array(vals1[2]))
        autolabel_pos(ax,pint2,int_time2,perc2,np.array(vals2[2]))

        remaining1 = (np.array(wall_time)[0:2:len(wall_time)] - np.array(vals1[2]) - np.array(klee1))
        remaining2 = (np.array(wall_time[1:2:len(wall_time)+1]) - np.array(vals2[2]) - np.array(tci2) - klee2)
        perc1 = 100*remaining1/np.array(wall_time)[0:2:len(wall_time)]
        perc2 = 100*remaining2/np.array(wall_time[1:2:len(wall_time)+1])
        off1 = np.array(vals1[2])+ np.array(tci1)+np.array(klee1)
        off2 = np.array(vals2[2])+ np.array(tci2)+np.array(klee2)
        p2 = p.bar(ind,remaining1,width,color=colors(pvalues[3]),bottom=off1,edgecolor='white')
        p21 = p.bar(ind+width,remaining2,width,color=colors2(pvalues[3]),bottom=off2,edgecolor='white')
        autolabel_pos(ax,p2,remaining1,perc1,off1)
        autolabel_pos(ax,p21,remaining2,perc2,off2)
        ax.yaxis.grid(True)
        p.xlabel("Drivers")
        p.ylabel("% of time")
        alpha = 1.2
        p.legend((psolv[0], pint[0], p2[0]), ( "Solving Time" , "Interpretation Time" , "Remaining Time"), loc='best')
        file_name = "breakdown_klee_tci_" + str(int(wall_time[0]/60));
        p.ylim((0,alpha*max(wall_time)))
        p.savefig("results/"+file_name)
        #p.show()

def instantiate_directory(dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

def get_graphs_tci(files,ms):
        data = []
        global head,idx
        head = []
        head2 = []
        mapping = {}
        files = [ s.replace('stats/','') for s in files]
        print "Map: ", mapping
        i = 0
        for i,f in enumerate(ms):
            print files[i]
            if i == 0:
                head = f[0]
            if i == 1:
                head2 = f[0]
            data.append(f[-1])
            mapping[files[i]] = data[i]
        print head
        print head2
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
        symb_tb = []
        tci_time = []
        klee_time = []
        num_paths = np.where(head == 'CompletedPaths')[0]
        index_num = np.where(head == 'TranslationBlocks')[0]
        idx_symb_blocks = np.where(head == 'TranslationBlocksKlee')
        idx_tci_time = np.where(head == 'CumulativeTCIInterpretationTime')
        idx_klee_time = np.where(head == 'CumulativeKLEEInterpretationTime')
        print idx_tci_time
        print idx_klee_time
        for j in range(len(files)):
            temp = mapping[files[j]]
            c_paths.append(temp[num_paths])
            bb_number.append(temp[index_num])
            symb_tb.append(temp[idx_symb_blocks])
            tci_time.append(temp[idx_tci_time])
            klee_time.append(temp[idx_klee_time])
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

        instantiate_directory("results")

        #first_graph(ind,vals1,vals2,width,drivers)
        breakdown_graph(ind,vals,mapping,files,tci_time,klee_time,width,drivers,wall_time)
        #Graph that shows the number of paths
        second_graph(files,drivers,ind,width,wall_time[0])
        #Ratio translation time LLVM vs TCI
        third_graph(vals,drivers,ind,width,wall_time[0])
        #Ratio number of blocks executed KLEE vs TCI
        fourth_graph(symb_tb,drivers,ind_1,ind_2,width,wall_time[0])

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
		
