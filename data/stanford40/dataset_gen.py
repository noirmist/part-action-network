root_folder = '/media/hci-jw/Plextor1tb/workspace/data/'
bbox_folder = root_folder + "BBoxImage/"
jpeg_folder = root_folder + "JPEGImages/"
pann_folder = root_folder + "PartAnnotations2/"
part_folder = root_folder + "PartImages/"

#open bboxlist
bf = open(root_folder+"bboximg.list","r")
bboxfile = bf.read()
bboxlist = bboxfile.split("\n")
bboxlist.pop()
#print bboxlist

#open partimage list
pf = open(root_folder+"partimg.list",'r')
partfile = pf.read()
partlist = partfile.split("\n")
partlist.pop()
#print partlist

#open dataset list
df = open(root_folder+"dataset.list",'w')

img_label = 0
for counter, bboximg in enumerate(bboxlist):
    #bbox label increased index%100 
    img_label = int(counter/ 100)
    
    #add bboxlist and label
    line = bboximg + ' ' + str(img_label) + '\n'
    df.write(line)

#split / get file name
    temp = bboximg.split("/")
    file_name = temp[len(temp)-1]
    #print file_name

    #add jpegfolder+name and label
    line = "JPEGImages/" +file_name + ' ' + str(img_label) +'\n'
    df.write(line)

    #add part image and label
    #load correct partAnnodation2
    #partann path + file name[:-4].txt 
    partannfile = open(pann_folder+file_name[:-4]+".txt", "r")
    #readline
    ann = partannfile.readline()
    #split by space
    ann_list = ann.split(' ')

    #print partlist[counter*7:counter*7+7]
    #read 7times from head to rhand
    for partimg, partlabel in zip(partlist[counter*7:counter*7+7], ann_list):
        #print partimg, ", ",  partlabel
        #if the value is -1:
        if int(partlabel) == -1:
            #label == 74
            partlabel = "74"
        #else
            #previous label add 39
        else:
            partlabel = str( int(partlabel) + 39)
        #make line 
        line = partimg + ' ' + partlabel +'\n'

        # add file path  label
        df.write(line)

df.close()
pf.close()
bf.close()
