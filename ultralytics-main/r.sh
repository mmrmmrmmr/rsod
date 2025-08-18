rsync -av --exclude-from /home/mamingrui/sod/ultralytics-main/exclude_file.txt \
mamingrui@10.68.155.95:/home/mamingrui/sod/ultralytics-main/ /home/mamingrui/sod/ultralytics-main
rsync -av --exclude-from exclude_file.txt \
mamingrui@10.168.40.142:/home/mamingrui/sod/ultralytics-main/ /home/mamingrui/sod/ultralytics-main
rsync -av --exclude-from exclude_file.txt \
/home/mamingrui/sod/ultralytics-main/ mamingrui@10.68.155.95:/home/mamingrui/sod/ultralytics-main
rsync -av --exclude-from exclude_file.txt \
/home/mamingrui/sod/ultralytics-main/ mamingrui@10.168.40.142:/home/mamingrui/sod/ultralytics-main




mmr7821976431