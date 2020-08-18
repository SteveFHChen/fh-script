import os
import sys

#Sample command:
#python C:/fh/ws/ws1/fh-script/pythonsc/webcat/covidsklearn.py "{'area': '美国', 'outputPath': 'C:/fh/testenv1/chart','diagramFileName':'plothello20200725-185829.png'}"

oriScript = sys.argv[0]
param=sys.argv[1]

#Test data
#oriScript="C:/fh/ws/ws1/fh-script/pythonsc/webcat/covidsklearn.py"
#param="{'area': '美国', 'outputPath': 'C:/fh/testenv1/chart','diagramFileName':'plothello20200725-185829.png'}"

oriPath = oriScript[:oriScript.rfind("/")+1]

#mainScript = f"{oriPath}covidsklearn-knn.py"
mainScript = f"{oriPath}covidsklearn-lr.py"

cmd = f"python {mainScript} \"{param}\""

'''
print("Run cmd:", cmd)

for s in sys.argv:
    print(s)
'''

code=os.system(cmd)
#print("=========")
#print(code)
