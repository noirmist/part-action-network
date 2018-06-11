import random

out = open('shuffled_dataset.list', 'w')

f = open("dataset.list", "r")
dataset = f.read()
data_list = dataset.split('\n')
data_list.pop()

#print data_list

list0 = []
list1 = []
list2 = []
list3 = []
list4 = []
list5 = []
list6 = []
list7 = []
list8 = []

for idx, item in enumerate(data_list):
    if idx%9 == 0:
        list0.append(item)
    elif idx%9 == 1:
        list1.append(item)
    elif idx%9 == 2:
        list2.append(item)
    elif idx%9 == 3:
        list3.append(item)
    elif idx%9 == 4:
        list4.append(item)
    elif idx%9 == 5:
        list5.append(item)
    elif idx%9 == 6:
        list6.append(item)
    elif idx%9 == 7:
        list7.append(item)
    elif idx%9 == 8:
        list8.append(item)

conbined_list = list(zip(list0, list1, list2, list3, list4, list5, list6, list7, list8))
random.shuffle(conbined_list)

list0, list1, list2, list3, list4, list5, list6, list7, list8 = zip(*conbined_list)
for i in range(len(list0)):
    out.write(list0[i]+'\n')
    out.write(list1[i]+'\n')
    out.write(list2[i]+'\n')
    out.write(list3[i]+'\n')
    out.write(list4[i]+'\n')
    out.write(list5[i]+'\n')
    out.write(list6[i]+'\n')
    out.write(list7[i]+'\n')
    out.write(list8[i]+'\n')

out.close()
f.close()



