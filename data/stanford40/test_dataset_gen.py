root_folder = '/media/hci-jw/Plextor1tb/workspace/data/'
bbox_folder = root_folder + "test_BBoxImage/"
jpeg_folder = root_folder + "JPEGImages/"
part_folder = root_folder + "test_PartImages/"

#open bboxlist
bf = open(root_folder+"test_bboximg.list","r")
bboxfile = bf.read()
bboxlist = bboxfile.split("\n")
bboxlist.pop()
#print bboxlist

#open partimage list
pf = open(root_folder+"test_partimg.list",'r')
partfile = pf.read()
partlist = partfile.split("\n")
partlist.pop()
#print partlist

#open dataset list
df = open(root_folder+"test_dataset.list",'w')


img_label = 0
for counter, bboximg in enumerate(bboxlist):
    #add bboxlist
    line = bboximg + '\n'
    df.write(line)

#split / get file name
    temp = bboximg.split("/")
    file_name = temp[len(temp)-1]
    #add jpegfolder+name 
    line = "JPEGImages/" +file_name +'\n'
    df.write(line)

#add part image and label
#load correct partAnnodation2
    #read 7times from head to rhand
    for partimg in partlist[counter*7:counter*7+7]:
        line = partimg +'\n'
        # add file path  label
        df.write(line)

df.close()
pf.close()
bf.close()
