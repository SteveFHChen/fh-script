import os

wsPath="C:\\fh\\ws\\ws1"

exists=os.listdir(wsPath)

fo=open("./fhGitRepoConfig.txt")
lines=fo.read().splitlines()

for line in lines:
  if not line.startswith("#") and line!="" and line not in exists:
    print(f"=====Repository {line} not exists, will download now...");
    pipObject=os.popen(f"git clone https://github.com/SteveFHChen/{line}.git {wsPath}/{line}")
    print(pipObject.read().splitlines());
  else:
    print(f"=====Repository {line} exists.")