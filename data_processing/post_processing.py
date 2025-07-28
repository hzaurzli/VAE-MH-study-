


data_file='../output/default/vae_gen.txt'
file_ID=open(data_file)

out_file='../output/default/len_4_det.txt'
out_ID=open(out_file,'a')

i=0
saved_lines=[]

### 38=+4; 36=+3; 34=+2
for line in file_ID.readlines():
	# line=line.strip()
	if len(line)==38 and line not in saved_lines:
		# print (line)
		# print (len(line))
		# sys.exit(0)
		out_ID.write(line[::-1])
		i+=1
	saved_lines.append(line)
	if i==100:
		break 
if i<100:
	print ('have ',i)
	print ('Results not enough')

file_ID.close()
out_ID.close()

