#!/usr/bin/python

# import modules
import argparse
import sys
import numpy as np
####################################################################
### notes: in this library, bed files for operations like merge  ###
### and substracting contains require at least four columns:     ###
### chr,start,end,strand.                                        ###
####################################################################

################################
######## function list  ########
################################
#1.	neighbor_merge: merge two ranges, say two exons
#2.	bedmerge: merge a list of ranges, say a few exons 
#3.	exon2intron: calculate intron bed from a list of exons.
#4.	masked_intron: identify introns which are covered by exons of other isoforms
#5.	get_UTR: get 5' and 3' UTR of isoforms.
#6.	get_gene_bed: generate bed files for a list of input genes given in a txt file
#7.	merge_exon: merge all exons of multiple isoforms from the same gene.
#8. 	get_intron: calculate intron bed for each isoform
#
#
#
#################################
############ functions:  ########
#################################


def neighbor_merge(range1,range2):
	if range2[1]<=range1[2]:
		merged =[(range1[0],range1[1],max(range1[2],range2[2]),range1[3])]
	else:
		merged=[range1,range2]
	return merged



def bedmerge(featureRange):
	# featureRange: a list of ranges in bed format, such as [(1,1000,2000,+),(1,2200,3000,-)]
	featureRange.sort(key = lambda x: (x[0],x[1]))	
	merged=[]
	nRange=len(featureRange)
	if nRange == 1:
		merged=featureRange
	elif nRange == 2:
		imerge=neighbor_merge(featureRange[0],featureRange[1])
		for each in imerge:
			merged.append(each)
	else:
		i = 2
		imerge=neighbor_merge(featureRange[0],featureRange[1])
		n_imerge=len(imerge)
		while n_imerge > 0:
			if n_imerge == 2:
				merged.append(imerge[0])
			
			imerge=neighbor_merge(imerge[n_imerge-1],featureRange[i])
			n_imerge=len(imerge)
			if i == nRange-1:
				for each in imerge:
					merged.append(each)
				n_imerge = -1
			i+=1

	return merged	

def exon2intron(featureRange):
	featureRange.sort(key = lambda x: (x[0],x[1]))	
	nRange=len(featureRange)
	intron=[]
	if nRange > 1:
		for i in range(1,nRange):
			exon0=featureRange[i-1]
			exon1=featureRange[i]
			thisintron=(exon0[0],exon0[2],exon1[1],exon1[3])
			intron.append(thisintron)
	return intron



def masked_intron(GTFfile,maskedfile="masked_intron.bed",chroms=list(map(str,range(1,23)))+['X','Y']):
	merged = merge_exon(GTFfile,chroms=chroms)
	introns= get_intron(GTFfile,chroms=chroms)

	exon = merged['merged_exon']
	intron   = introns['intron']
	iso2gene = introns['iso2gene']
	fnew = open(maskedfile,'w')

	for isoform in intron:
		the_introns = intron[isoform]
		the_gene = iso2gene[isoform]
		the_exons = exon[the_gene]
		for i in range(len(the_introns)):
			for j in range(len(the_exons)):
				if the_introns[i][1] >= the_exons[j][1]  and the_introns[i][2] <= the_exons[j][2]:
					output = "\t".join(map(str,the_introns[i]))+'\t'+isoform+'\t'+the_gene+'\n'
					fnew.write(output)
	fnew.close()

def get_UTR(GTFfile, utr_file="",chroms=list(map(str,range(1,23)))+['X','Y']):
	f=open(GTFfile)
	# record UTR, CDS and strand information for determining iUTR5 or iUTR3
	utr = {} 
	cds = {}
	strand = {} 
	for line in f:
		table = line.split('\t')
		if line[0] != '#' and table[0] in chroms:
			if table[2] == 'UTR' and 'transcript_id' in line:
				tcx  = line.split('transcript_id')[1].split('"')[1]
				gene = line.split('gene_id')[1].split('"')[1] 
				infor=[table[0],int(table[3]),int(table[4]),table[6],gene]
				if tcx in utr:
					utr[tcx].append(infor)
				else:
					utr[tcx]=[infor]
				strand[tcx] = table[6]
			if table[2] == 'CDS' and 'transcript_id' in line:
				tcx  = line.split('transcript_id')[1].split('"')[1]
				gene = line.split('gene_id')[1].split('"')[1] 
				cds[tcx] = int((int(table[3])+int(table[4]))/2)
				strand[tcx] = table[6]
		
	f.close()


	# determine 5UTR and 3UTR
	allUTR=[]
	for tcx in cds:
		if tcx in utr:
			istrand = strand[tcx]
			icds = cds[tcx]
			iutr = utr[tcx]
			iUTR5 = []
			iUTR3 = []
			for thisUTR in iutr:
				if istrand == '+':
					if thisUTR[1] > icds:
						iUTR3.append(thisUTR[1])
						iUTR3.append(thisUTR[2])
					else:
						iUTR5.append(thisUTR[1])
						iUTR5.append(thisUTR[2])
				elif istrand == '-':
					if thisUTR[1] > icds:
						iUTR5.append(thisUTR[1])
						iUTR5.append(thisUTR[2])
					else:
						iUTR3.append(thisUTR[1])
						iUTR3.append(thisUTR[2])
			iUTR5.sort()
			iUTR3.sort()
			iiutr=iutr[0]
			if len(iUTR5) > 1:
				allUTR.append([iiutr[0],iUTR5[0]-1,iUTR5[-1],iiutr[3],iiutr[4],'5UTR',tcx])
			if len(iUTR3) > 1:
				allUTR.append([iiutr[0],iUTR3[0]-1,iUTR3[-1],iiutr[3],iiutr[4],'3UTR',tcx])

	allUTR.sort(key = lambda x: (x[0],x[1]))				
	

	# print UTR to file if required
	if len(utr_file) > 1:
		f=open(utr_file,'w')
		for iUTR in allUTR:
			table = iUTR
			iUTR[1] = str(iUTR[1])
			iUTR[2] = str(iUTR[2])
			out = "\t".join(iUTR)+'\n'
			f.write(out)
		f.close()



	return(allUTR)


def get_tss_region(GTFfile,w='1000-300',tss_bed_file='',chroms=list(map(str,range(1,23)))+['X','Y']):
    # assume genelistfile contains one gene per line.
    # get bed format
    TSSbed=[]
    wup   = int(w.split('-')[0])
    wdown = int(w.split('-')[1])

    #wup   = w[0]
    #wdown = w[1]  # dbSNP  based on ref: Genome-wide promoter extraction and analysis in human, mo    use, and rat, Genome Biology
    #print(wup)
    #print(wdown)
    f = open(GTFfile)
    for line in f:
        table = line.split('\t')
        if table[0] in chroms and table[2] == 'transcript':
            chrom  = table[0]
            strand = table[6] 
            tcx = line.split('transcript_id')[1].split('"')[1]
            geneid = line.split('gene_id')[1].split('"')[1]
            genesymbol = line.split('gene_name')[1].split('"')[1]
            if  strand == "+":
                iregion = [chrom,int(table[3])-1-wup,int(table[3])-1+wdown,strand,tcx,geneid,genesymbol]
            elif strand == '-':
                iregion = [chrom,int(table[4])-wdown,int(table[4])+wup,strand,tcx,geneid,genesymbol]
            TSSbed.append(iregion)
    f.close()
	# write to file
    if len(tss_bed_file) > 1:
        f=open(tss_bed_file,'w')
        for item in TSSbed:
            out = '\t'.join([item[0],str(item[1]),str(item[2]),item[3],item[4],item[5],item[6]])+'\n'
            f.write(out)
        f.close()

def get_intergenic_region(GTFfile,intergenic_region_file='',chroms=list(map(str,range(1,23)))+['X','Y']):
	# get bed format
    genebed={}
    chrommax={}
    for ichrom in chroms:
        chrommax[ichrom] = 0
    f = open(GTFfile)
    for line in f:
        table = line.split('\t')
        chrom = table[0]
        if chrom in chroms:
            if table[2] == "gene":
                left  = int(table[3])-1
                right = int(table[4]) 
                ensid  = line.split('gene_id')[1].split('"')[1]
                symbol = line.split('gene_name')[1].split('"')[1].upper()
                record = (chrom,left,right,table[6])
                if chrom in genebed:
                    genebed[chrom].append(record)
                else:
                    genebed[chrom] = [record]
                if right > chrommax[chrom]:
                    chrommax[chrom] = right
    f.close()


	# print to file if required 
    if len(intergenic_region_file) > 1:
        f=open(intergenic_region_file,'w')
        for ichrom in genebed.keys():
            #print(ichrom)
            chrombed= [(ichrom,1,chrommax[ichrom],'+')]
            this_genebed = genebed[ichrom]
            regions = bed_subtract(chrombed,this_genebed)
            if len(regions) > 0:
                for iregion in regions:
                    out = ichrom+'\t'+str(iregion[1])+'\t'+str(iregion[2])+'\n'
                    f.write(out)
        f.close()

def get_gene_bed(GTFfile,gene_bed_file='',chroms=list(map(str,range(1,23)))+['X','Y'],cis='0-0'):
	# get bed format
	genebed={}
	gene2type={}
	f = open(GTFfile)
	for line in f:
		table = line.split('\t')
		if table[0] in chroms:
			if table[2] == "gene":
				#print(line)
				ensid  = line.split('gene_id')[1].split('"')[1]			
				symbol = line.split('gene_name')[1].split('"')[1].upper()			
				record = (table[0],int(table[3])-1,int(table[4]),table[6],ensid,symbol)

				if 'gene_biotype' in line:
					thistype = line.split('gene_biotype')[1].split('"')[1]
				if 'gene_type' in line:
					thistype = line.split('gene_type')[1].split('"')[1]



				gene2type[ensid] = thistype

				if table[0] in genebed:
					genebed[table[0]].append(record)
				else:
					genebed[table[0]] = [record]
	f.close()

	# print to file if required 
	udist=0
	if cis != '0-0':
		udist = int(cis.split('-')[0])
		ddist = int(cis.split('-')[1])


	if len(gene_bed_file) > 1:
		f=open(gene_bed_file,'w')
		for ichrom in genebed.keys():
			ichrom_gene = genebed[ichrom]
			for igene in ichrom_gene:
				if udist == 0:
					f.write('\t'.join([igene[0],str(igene[1]),str(igene[2]),igene[3],igene[4],igene[5],gene2type[igene[4]]])+'\n')
				else:
					if igene[3] == '+':
						f.write('\t'.join([igene[0],str(igene[1]),str(igene[2]),igene[3],igene[4],igene[5],gene2type[igene[4]],str(igene[1]-udist),str(igene[2]+ddist)])+'\n')
					else:
						f.write('\t'.join([igene[0],str(igene[1]),str(igene[2]),igene[3],igene[4],igene[5],gene2type[igene[4]],str(igene[1]-ddist),str(igene[2]+udist)])+'\n')
		f.close()
	return(genebed)



def get_cissnp_bed(GTFfile,cissnp_file='',chroms=list(map(str,range(1,23)))+['X','Y'],cis='0-0'):
	# get bed format
	
	f = open(GTFfile)
	# specify cis-ranges of each genes
	gchr={}
	gcisstart={}
	gcisend={}
	gstrand={}
	gene2symbol={}
	updistance  = int(cis.split('-')[0])
	dndistance  = int(cis.split('-')[1])
	ref={}
	for line in f:
		table = line.split('\t')
		chrom=table[0]
		if chrom in chroms:
			if table[2] == "gene":
				ensid  = line.split('gene_id')[1].split('"')[1]			
				symbol = line.split('gene_name')[1].split('"')[1]		
				
				
				start  = int(table[3])
				end    = int(table[4])
				strand =     table[6]

				if strand == '+':
					tss = start
					left=tss-updistance
					right= max(end,tss+dndistance)
					#gtss[ensid] = tss
				elif strand == '-':
					tss = end
					left = min(start,tss-dndistance)
					right = tss+updistance
					#gtss[ensid] = tss

				gcisstart[ensid] = left
				gcisend[ensid] = right
				
				gene2symbol[ensid]=symbol
				#gstrand[ensid] = strand   

				### 
				if chrom in ref:
					new=ref[chrom]+','+ensid
					ref[chrom]=new
				else:
					ref[chrom]=ensid
				## store genes in each chromosome
	f.close()

	# variant_id	chr	variant_pos	ref	alt	num_alt_per_site	rs_id_dbSNP151_GRCh38p7	variant_id_b37
	#1	13526	rs1209314672
	f = open(cissnp_file)
	#print('...scaning SNPs')
	#i=0
	for line in f:
		t = line.strip().split('\t')
		chrom = t[0]
		co    = int(t[1])
		snp   = t[2]

		if chrom in ref:
			refgenes=list(set(ref[chrom].split(',')))
			for igene in refgenes:
				if co >= gcisstart[igene] and co<= gcisend[igene]:
					out = line.strip()+'\t'+igene+'\t'+gene2symbol[igene]
					#out = line.strip()+'\t'+igene+'\t'+gene2symbol[igene]+'\t'+str(gcisstart[igene])+'\t'+str(gcisend[igene])
					print(out)	
	f.close()




def get_isoform_bed(GTFfile,isoform_bed_file='',chroms=list(map(str,range(1,23)))+['X','Y']):
	# assume genelistfile contains one gene per line.
	# get bed format
	isoformbed={}
	gene2type = {}
	f = open(GTFfile)
	for line in f:
		table = line.split('\t')
		if table[0] in chroms:
			if table[2] == "transcript":
				isoformid  = line.split('transcript_id')[1].split('"')[1]			
				ensid  = line.split('gene_id')[1].split('"')[1]			
				symbol = line.split('gene_name')[1].split('"')[1].upper()			
				record = (table[0],int(table[3])-1,int(table[4]),table[6],isoformid,ensid,symbol)


				if 'gene_biotype' in line:
					thistype = line.split('gene_biotype')[1].split('"')[1]
				if 'gene_type' in line:
					thistype = line.split('gene_type')[1].split('"')[1]

				


				gene2type[ensid] = thistype
				if table[0] in isoformbed:
					isoformbed[table[0]].append(record)
				else:
					isoformbed[table[0]] = [record]
	f.close()

	# print to file if required 
	if len(isoform_bed_file) > 1:
		f=open(isoform_bed_file,'w')
		for ichrom in isoformbed.keys():
			ichrom_isoform = isoformbed[ichrom]
			for iisoform in ichrom_isoform:
				f.write('\t'.join([iisoform[0],str(iisoform[1]),str(iisoform[2]),iisoform[3],iisoform[4],iisoform[5],iisoform[6],gene2type[iisoform[5]]])+'\n')
		f.close()
	return(isoformbed)

def merge_exon(GTFfile,merged_exon_file='',chroms=list(map(str,range(1,23)))+['X','Y']):
    # record exon coordination
    exon={}
    f=open(GTFfile)
    #print(GTFfile)
        ## V       WormBase        gene    180     329     .       +       .       gene_id "WBGene00197333"; gene      _version "1"; gene_name "cTel3X.2"; gene_source "WormBase"; gene_biotype "ncRNA";
    for line in f:
        table=line.strip().split('\t')   
        if line[0] != '#' and table[0] in chroms:  # skip comment line
            if table[2] == 'exon':
                gene=line.strip().split('gene_id')[1].split('"')[1]
                # print('ok:',gene)
                iexon=(table[0],int(table[3])-1,int(table[4]),table[6])  # gtf to bed coordination
                if gene in exon:
                    exon[gene].append(iexon)
                else:
                    exon[gene]=[iexon]
    f.close()
    # merge all exons of each gene
    merged_exon={}
    gene_length={}
    for gene in exon:
        merged=bedmerge(exon[gene])
        merged_exon[gene]=merged
	
		# calculate merged gene length(sum of non-overlapping exons)
        length=0
        for each in merged:
            length+=each[2]-each[1]
        gene_length[gene]=length

	# assign merged exons and gene lengths
       # print(len(gene_length),'in f: merge_exon')
    merged = {}
    merged['merged_exon']=merged_exon
    merged['merged_gene_length']=gene_length


	# print merged exons if required
    if len(merged_exon_file) > 1:
        m=open(merged_exon_file,'w')
        for gene in merged_exon:
            merged=merged_exon[gene]
            for each in merged:
                joined="\t".join([each[0],str(each[1]),str(each[2]),gene,'0',each[3]])+"\n"
                m.write(joined)
        m.close()
    return merged


def get_independent_intron(GTFfile,independent_intron_file='iintron.tmp',chroms=list(map(str,range(1,23)))+['X','Y']):
	# record exon coordination
	exon={}
	f=open(GTFfile)
	for line in f:
		table=line.split('\t')
		if line[0] != '#' and table[0] in chroms:  # skip comment line
			ichr = table[0]
			if table[2] == 'exon':
				iexon=(ichr,int(table[3])-1,int(table[4]),table[6])  # gtf to bed coordination
				if ichr in exon:
					exon[ichr].append(iexon)
				else:
					exon[ichr]=[iexon]
	f.close()
	
	# get gene coordinates in bed formats
	genebed=get_gene_bed(GTFfile,chroms=chroms)

	# calculate and write independent introns
	f = open(independent_intron_file,'w')
	record=[]
	for ichrom in chroms:
		if ichrom in genebed.keys():
			ichrom_gene = genebed[ichrom]
			ichrom_exon = exon[ichrom]
			sub = bed_subtract(ichrom_gene,ichrom_exon)
			for item in sub:
				if item[2] - item[1] >= 10:
					uniqued = unique_judge(item,ichrom_gene)	
					if len(uniqued)>0:
						record.append(uniqued)
	# write to file	
	f = open(independent_intron_file,'w')
	record = list(set(record))
	record.sort(key = lambda x: (x[0],x[1]))
	for item in record:
		ilength = item[2]-item[1]
		f.write('\t'.join([item[0],str(item[1]),str(item[2]),item[4],str(ilength)])+'\n')
	f.close()


def unique_judge(intron,genes):
	# intron: a single intron
	# genes: a list of genes
	hostgene=[]
	ngene=len(genes)
	i = 0
	count = 0
	flag = 0
	while i < ngene and flag == 0:
		igene = genes[i]
		if intron[1] > igene[1] and intron[1] < igene[2]:
			count = count + 1
			hostgene = igene
			if count > 1:
				flag = 1
		i = i+1

	if  count == 1:
		#return((intron[0],intron[1],intron[2],intron[3]))
		return((intron[0],intron[1],intron[2],intron[3],hostgene[4]))
	else:
		return([])

def bed_subtract(bedA,bedB):
	# assume that A and B are on the same chromosome.
	# calculate bedA-bedB
	bedA=bedmerge(bedA)
	bedB=bedmerge(bedB)
	AminusB=[]
	for ibed in bedA:
		ichr    = ibed[0]
		istart  = ibed[1]
		iend    = ibed[2]
		istrand = ibed[3]

		# find overlapping regions in bedB
		overlapped=[]
		for ibedB in bedB:
			ibedB = list(ibedB)
			judgeleft  = (ibedB[1] > istart and ibedB[1] < iend)
			judgeright = (ibedB[2] > istart and ibedB[2] < iend)
			if judgeleft == 1 or judgeright == 1:
				if ibedB[1] <= istart:
					ibedB[1] = istart
				if ibedB[2] >= iend:
					ibedB[2] = iend
				overlapped.append(ibedB)

		# calculate independent introns
		nexons = len(overlapped)
		if nexons == 0:
			AminusB.append((ichr,istart,iend,istrand))
		else:
			for i in range(nexons):
				iexon = overlapped[i]
				if i == 0:
					if iexon[1] > istart:
						AminusB.append((ichr,istart,iexon[1],istrand))
				if i >= 1:
					pexon = overlapped[i-1]
					AminusB.append((ichr,pexon[2],iexon[1],istrand))
				if i == nexons-1:
					if iexon[2] < iend:
						AminusB.append((ichr,iexon[2],iend,istrand))
	
	# return
	return(AminusB)


def subtract(fileA, fileB):
	A=[]
	f = open(fileA)
	for line in f:
		t = line.strip().split('\t')
		A.append((t[0],int(t[1]),int(t[2]),t[3]))
	f.close()

	B=[]
	f = open(fileB)
	for line in f:
		t = line.strip().split('\t')
		B.append((t[0],int(t[1]),int(t[2]),t[3]))
	f.close()

	A=bedmerge(A)
	for item in A:
		print('\t'.join([item[0],str(item[1]),str(item[2]),item[3]]))


	B=bedmerge(B)

	R=bed_subtract(A,B)
	length = len(R)
	print('length:'+str(length))
	for item in R:
		print('\t'.join([item[0],str(item[1]),str(item[2]),item[3]]))

def get_isoform_length(GTFfile,isoformlength_file='',chroms=list(map(str,range(1,23)))+['X','Y']):
	# calculate length of isoforms
    f=open(GTFfile)
    isolength = {}
    gene2iso= {}
    iso2gene= {}
    for line in f:
        table=line.split('\t')
        if line[0] != '#' and table[0] in chroms:  # skip comment line
            if table[2] == 'exon':
                gene= line.split('gene_id')[1].split('"')[1]  # gene ID
                tcx = line.split('transcript_id')[1].split('"')[1]  # tcx ID
                exon_length = int(table[4])-int(table[3])+1
				# isoform length calculate
                if tcx in isolength:
                    isolength[tcx] = isolength[tcx] + exon_length
                else:
                    isolength[tcx] =  exon_length

                # record isoforms for each gene 
                if gene in gene2iso:
                    if tcx not in gene2iso[gene]:
                        gene2iso[gene].append(tcx)
                else:
                    gene2iso[gene] = [tcx]
                iso2gene[tcx] = gene

    if len(isoformlength_file) > 1:
        f=open(isoformlength_file,'w')
        f.write('isoform\tgene\tlength\n')
        for thisiso in iso2gene:
            out=thisiso+'\t'+iso2gene[thisiso]+'\t'+str(isolength[thisiso])+'\n'
            f.write(out)
        f.close()	

	# return
    ret={}
    ret['gene2iso']  = gene2iso
    ret['isoform_length'] = isolength
    return(ret)

def get_gene_length(GTFfile,genelength_file='',chroms=list(map(str,range(1,23)))+['X','Y']):
	# record exon coordination
	# calculate length of merged exons
	merged_gene_length = merge_exon(GTFfile,chroms=chroms)['merged_gene_length']


	# calculate length of each transcrpt isoform
	ret_isoform_length = get_isoform_length(GTFfile,chroms=chroms)
	gene2iso =ret_isoform_length['gene2iso']
	isolength=ret_isoform_length['isoform_length']

	# calculate gene length as mean, median or max of isoforms
	gene_length = {}
	for igene in gene2iso:
		isoforms = gene2iso[igene]
		length_set = []
		for iisoform in isoforms:
			length_set.append(isolength[iisoform])
		gene_length[igene] = [int(list_mean(length_set)),int(list_median(length_set)),max(length_set),merged_gene_length[igene]]

	# print
	if len(genelength_file) > 1:
		f = open(genelength_file,'w')
		f.write('gene\tmean\tmedian\tlongest_isoform\tmerged\n')
		for igene in gene_length:
			tmp =  gene_length[igene]
			tmplst = [igene,str(tmp[0]),str(tmp[1]),str(tmp[2]),str(tmp[3])]
			f.write('\t'.join(tmplst)+'\n')
		f.close()



def get_exon(GTFfile,exon_file='',chroms=list(map(str,range(1,23)))+['X','Y']):
	# record exon coordination
	f=open(GTFfile)
	exon = {}
	iso2gene= {}
	for line in f:
		table=line.split('\t')
		if line[0] != '#' and table[0] in chroms:  # skip comment line
			if table[2] == 'exon':

				gene= line.split('gene_id')[1].split('"')[1]  # gene ID
				tcx = line.split('transcript_id')[1].split('"')[1]  # tcx ID
				iso2gene[tcx] = gene   # map isoforms to genes
				iexon=(table[0],int(table[3])-1,int(table[4]),table[6])  # gtf to bed coordination
				if tcx in exon:
					exon[tcx].append(iexon)
				else:
					exon[tcx]=[iexon]
	f.close()
	ret={}
	ret['exon']=exon
	ret['iso2gene']=iso2gene


	if len(exon_file) > 1:
		f=open(exon_file,'w')
		alltcx = exon.keys()
		for itcx in alltcx:
			exonlist = exon[itcx]
			for item in exonlist:
				exonlength = str(item[2] - item[1])
				out = list(map(str,item))+[itcx, iso2gene[itcx],exonlength]
				out = '\t'.join(out)+'\n'
				f.write(out)
	#return
	return(ret)




def cal_exon_skip_tags(exonlist):
	nexon = len(exonlist)
	nexon_middle = nexon-2
	#if nexon == 3:
	#	read_left  = 
	#	read_right =
	#	read_skip  = 

	#if nexon >=4:
	#for i in range(1,nexon_middle):
	#	for j in range(i+1,n



#def get_exon_skip_db(GTFfile,skipdb_file='',chroms=map(str,range(1,23))+['X','Y']):
#	# record exon coordination
#	exon_raw = get_exon(GTFfile,exon_file='',chroms=chroms)
#	exon = exon_raw['exon']
#	iso2gene= exon_raw['iso2gene']
#	alltcx = exon.keys()
#	for itcx in alltcx:
#		exonlist = exon[itcx]
#		nexon = len(exonlist)
#		if nexon >= 3:
#			xxx=cal_exon_skip_tags(exonlist)
		



def get_splice_site(GTFfile,splice_site_file='',chroms=list(map(str,range(1,23)))+['X','Y']):

	introninfo=get_intron(GTFfile,intron_file=splice_site_file,chroms=chroms)
	tcx2intron=introninfo['intron']
	iso2gene=introninfo['iso2gene']

	# print splice sites to file if required
	if len(splice_site_file) > 0:
		f=open(splice_site_file,'w')   
		for tcx in tcx2intron:
			allintrons=tcx2intron[tcx]
			for each in allintrons:
				intron_name=each[0]+':'+str(each[1])+'-'+str(each[2])
				strand=each[3]
				if strand=='+':
					donor_left=each[1]-3
					donor_right=each[1]+6

					acceptor_left=each[2]-20
					acceptor_right=each[2]+3
				elif strand=='-':
					donor_left=each[2]-6
					donor_right=each[2]+3

					acceptor_left=each[1]-3
					acceptor_right=each[1]+20
					

				joined_donor   ="\t".join([each[0],str(donor_left),str(donor_right),intron_name,'0',each[3],'donor',tcx,iso2gene[tcx]])+"\n"
				joined_acceptor="\t".join([each[0],str(acceptor_left),str(acceptor_right),intron_name,'0',each[3],'acceptor',tcx,iso2gene[tcx]])+"\n"
				
				f.write(joined_donor)
				f.write(joined_acceptor)
		f.close()

	# return
	return('0')




def get_intron(GTFfile,intron_file='',chroms=list(map(str,range(1,23)))+['X','Y']):
	# record exon coordination
	f=open(GTFfile)
	exon = {}
	iso2gene= {}
	for line in f:
		table=line.split('\t')
		if line[0] != '#' and table[0] in chroms:  # skip comment line
			if table[2] == 'exon':
				gene= line.split('gene_id')[1].split('"')[1]  # gene ID
				tcx = line.split('transcript_id')[1].split('"')[1]  # tcx ID
				iso2gene[tcx] = gene   # map isoforms to genes
				iexon=(table[0],int(table[3])-1,int(table[4]),table[6])  # gtf to bed coordination
				if tcx in exon:
					exon[tcx].append(iexon)
				else:
					exon[tcx]=[iexon]
	f.close()

	# calculate intron location based on exons for each tcx
	intron={}
	for tcx in exon:
		iexon = exon[tcx]
		iintron=exon2intron(iexon)
		if len(iintron) > 0:
			intron[tcx]=iintron
	to_return={}
	to_return['intron']=intron
	to_return['iso2gene']=iso2gene


	# print intron to file if required
	if len(intron_file) > 0:
		f=open(intron_file,'w')
		for tcx in intron:
			iintron=intron[tcx]
			for each in iintron:
				#intron_name=each[0]+':'+str(each[1])+'-'+str(each[2])
				intronlength=str(each[2]-each[1])
				joined="\t".join([each[0],str(each[1]),str(each[2]),each[3],tcx,iso2gene[tcx],intronlength])+"\n"
				f.write(joined)
		f.close()


	# return
	return to_return

def chroms_interpreter(chroms_input):
	t = chroms_input.strip().split(',')
	chroms = []
	for it in t:
		if '-' in it:
			start  = int(it.split('-')[0].strip())
			end    = int(it.split('-')[1].strip())+1
			irange = map(str,range(start,end))
			chroms = chroms + list(irange)
		else:
			chroms = chroms + [it]
	return(chroms)	



# convert GENCODE to ENSEMBL format
def gencode2ensembl(gtf1,gtf2):
#gtf1: ensembl GTF
#gtf2: gencode GTF

	f  = open(gtf1)
	fo = open(gtf2,'w')

	for line in f:
		if line[0:3] == 'chr':
			t = line.strip().split('\t')
			# update
			cname = t[0].split('chr')[1]
			if cname == 'M':
				cname = 'MT'

			# replace
			t[0] =  cname
			out = '\t'.join(t)+'\n'
			fo.write(out)
		else:
			fo.write(line)

	f.close()
	fo.close()


# check format of GTF: ensembl or gencode
def gtf_format_check(gtf):
	f=open(gtf)
	f.seek(10000)
	f.readline()
	chrom=f.readline().split('\t')[0]
	if len(chrom) >= 4:
		ftype = 'GENCODE'
	else:
		ftype = 'ENSEMBL'
	return(ftype)

############################################
##############   END OF FUNCTIONS  #########
############################################



############################################
##############   START: SIMPLE MATH ########
############################################
def list_sum(x):
	# assume x is lists of numbers
	s = 0
	for i in x:
		s = s + i
	return(s)

def list_mean(x):
	m = list_sum(x)/float(len(x))
	return m

def list_median(x):
    x.sort()
    length = len(x)
    median=np.median(x)
    return median

############################################
##############   END: SIMPLE MATH  #########
############################################



############################################
#############  ENTRY  ######################
############################################


# argument parser
parser = argparse.ArgumentParser()  # create a parser
parser.add_argument('GTFfile',help="GTF file: only ENSEMBL or GENCODE GTF file accepted")
parser.add_argument('-c','--chroms',metavar='chromosomes',help="chromosome list to analyze. Chromosomes can be separated by comma(,) or dash(-). For example, '-c 1-5,X,Y' means chromosomes 1 to 5 plus X and Y. Default is: 1-22,X,Y")
parser.add_argument('-m','--merged_exon',metavar='merged_exon',help="output file name for outputing merged exons from all isoforms of a gene in bed format")
parser.add_argument('-e','--exon',metavar='exon',help="output file name for exon coordination of splice isoforms in bed format")
parser.add_argument('-i','--intron',metavar='intron',help="output file name for outputing intron coordination of splice isoforms in bed format")
parser.add_argument('-d','--independent_intron',metavar='independent_intron',help="output file name for outputing independent intron coordination of genes. Independent introns refer to those introns that do not overlap with any exon of isoforms. It is calcualted by merging all exons of a chromosome followed by substracting them from gene regions.")
parser.add_argument('-b','--intergenic_region',metavar='intergenic_region',help="output file name for coordinates of intergenic regions, which is calculated by subtracting gene regions from each chromosome.")
parser.add_argument('-l','--gene_length',metavar='gene_length',help="output file name for gene length. Four types of gene lengths are calculated. The first three are the mean, median, and max length of the isoforms of a gene. The fourth is the length of the non-overlapping exons of all isoforms.")
parser.add_argument('-r','--isoform_length',metavar='isoform_length',help="output file name for isoform length file. Isoform length is calculated as the summed length of its exons")
parser.add_argument('-k','--masked_intron',metavar='masked_intron',help="output file name for the intron that overlaps with exons of other isoforms/genes")
parser.add_argument('-u','--UTR',metavar='UTR',help="output file name for UTR regions")
parser.add_argument('-s','--isoform',metavar='isoform',help="output file name for isoform coordinates and names.")
parser.add_argument('-q','--splice_site',metavar='splice_site',help="output file name for 5' or 3' splice site in bed format. The region is based on MaxEntScan: the 5' donor site is 9 bases long with 3 bases in exon and 6 bases in intron, and the 3' acceptor site is 23 bases long with 20 bases in the intron and 3 bases in the exon.")
parser.add_argument('-g','--gene',metavar='gene',help="output file name for gene coordinates and names. If cis-range of the gene needs to be calculated, users can include the -f option to sepcify cis-range.")
parser.add_argument('-p','--snp',metavar='SNP',help="An input file containing a list of SNPs with at least three columns, with the first being chromosome and the second being coordinate and the third being SNP names such as rs ID number. With this option, GTFtools will search for and output cis-SNPs for each gene annotated in the provided GTF file.")
parser.add_argument('-f','--cis',metavar='cis',help="-f specifies the upstream and downstream distance used to calculate cis-range of a gene. -f is specified in the format of 'distup-distdown', where distup represent the upstream distance from TSS and distdown means the downstream distance from the end of the gene. Note that this parameter takes effect only when the '-g' option is used. For example, using 'python gtftools.py -g gene.bed -f 2000-1000  demo.gtf' means that 2000 bases upstream and 1000 bases downstream of the gene will be clculated as the cis-range and the cis-range will be output to the gene.bed file. By default, -f is set to 0-0, indicating that cis-range will not be calculated when using -g to calculate gene information.")
parser.add_argument('-t','--TSS',metavar='TSS',help="output file name for a region flanking transcription start site (TSS). It is calculated as (TSS-wup,TSS+wdown) where wup is a user-specified distance, say 1000bp, upstream of TSS, wdown is the distance downstream of TSS. wup and wdown is defined by the w parameter specified by '-w'.")
parser.add_argument('-w','--window',metavar='window_size',help="w specifies the upstream and downstream distance from TSS as described in '-t'. w is specified in the format of 'wup-wdown', where wup and wdown represent the upstream and downstream distance of TSS. Default w = 1000-300 (that is, 1000 bases upstream of TSS and 300 bases downstream of TSS). This range is based on promoter regions used in the dbSNP database based on ref: Genome-wide promoter extraction and analysis in human, mouse, and rat, Genome Biology, 2005.")
parser.add_argument('-v','--version',action='version',version="GTFtools version:0.9.0")

#parser.add_argument('-a','--file_a',help="bed file a.")
#parser.add_argument('-b','--file_b',help="bed file b.")
#parser.add_argument('-p','--merged_file',help="merge two bed files.")
#parser.add_argument('-q','--subtracted_file',help="subtract b from A, i.e. a-b.")


args = parser.parse_args()   # parse command-line arguments


######################################
#################  main   ############
######################################
GTFfile=args.GTFfile
ftype = gtf_format_check(GTFfile)
#print('GTF format is '+ftype)
if ftype == 'GENCODE':
	print('Converting GTF to ENSEMBL format')
	GTFfile_ensembl=GTFfile+'.ensembl'
	gencode2ensembl(GTFfile,GTFfile_ensembl)
	GTFfile = GTFfile_ensembl


#print '------> Analyzing '+GTFfile

if args.chroms:
    chroms = chroms_interpreter(args.chroms)
    print('Analyzing chromosomes:')
    print(chroms)
else:
	chroms = list(map(str,range(1,23)))+['X','Y'] 

# print merged exon in bed format
if args.merged_exon:
	merge_exon(GTFfile,merged_exon_file=args.merged_exon,chroms=chroms)

if args.intergenic_region:
    get_intergenic_region(GTFfile,intergenic_region_file=args.intergenic_region,chroms=chroms)

# print gene length of merged exons
if args.gene_length:
	get_gene_length(GTFfile,genelength_file=args.gene_length,chroms=chroms)

# print isoform length of summed exons
if args.isoform_length:
	get_isoform_length(GTFfile,isoformlength_file=args.isoform_length,chroms=chroms)


# print introns of splice isoforms in bed format
if args.exon:
	get_exon(GTFfile,exon_file=args.exon,chroms=chroms)


# print introns of splice isoforms in bed format
if args.intron:
	get_intron(GTFfile,intron_file=args.intron,chroms=chroms)


# print introns of splice isoforms in bed format
if args.independent_intron:
	get_independent_intron(GTFfile,independent_intron_file=args.independent_intron,chroms=chroms)


# print masked introns in bed format
if args.masked_intron:
	masked_intron(GTFfile,maskedfile=args.masked_intron)


# print UTR in bed format
if args.UTR:
	get_UTR(GTFfile,utr_file=args.UTR,chroms=chroms)





# cis-range of genes
if args.cis:
	cis = args.cis
else:
	cis = '0-0' 
# print gene bed
if args.gene:
	get_gene_bed(GTFfile,gene_bed_file=args.gene,chroms=chroms,cis=cis)


# print cis-SNPs of each gene
if args.snp:
	get_cissnp_bed(GTFfile,cissnp_file=args.snp,chroms=chroms,cis=cis)






# print isoform bed
if args.isoform:
	get_isoform_bed(GTFfile,isoform_bed_file=args.isoform,chroms=chroms)


# print splice site bed
if args.splice_site:
	get_splice_site(GTFfile,splice_site_file=args.splice_site,chroms=chroms)


# print subtracted bed
#if args.subtracted_file and args.file_a and args.file_b:
#	subtract(args.file_a,args.file_b)



# print TSS neighborhood bed
if args.window:
	w = args.window
else:
	w = '1000-300'  # This range is based on promoters used in dbSNP database based on ref: Genome-wide promoter extraction and analysis in human, mouse, and rat, Genome Biology, 2005.
if args.TSS:
	get_tss_region(GTFfile,w=w,tss_bed_file=args.TSS,chroms=chroms)






