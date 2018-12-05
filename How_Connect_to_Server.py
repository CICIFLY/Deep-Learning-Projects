# 1.open two terminals:
# one for access server, the other for where the file or folder is 
# 2.copy file : student@ is a must 
# how to connect : student@cs-server.nmhu.edu:~/Desktop/Xi_Zhou
# hscilab282-10@ThinkCentre-M90z:~/Desktop/Xi_Zhou/AI/11_6_2018_CNN/CNN_Letter$ rsync -av Xi_letter_gen.txt student@cs-server.nmhu.edu:~/Desktop/Xi_Zhou
# copy folder : 
# rsync -av All_Letters_Data/ student@cs-server.nmhu.edu:~/Desktop/Xi_Zhou/All_Letters_Data/

# 3. run python3 file there : MUST START WITH python3
# 4: install libraries :
# pip3 install -user + package name 
# 5: exit : ctr+d
# 6.remove all images : rm *.jpg
# 7.remove folder: rm -r folder_name  