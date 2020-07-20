import sys
import json

#print("hello");
param1=sys.argv[1]

print("Check input parameters in python script:");
for i in range(1, len(sys.argv)):
	print("Param[",i,"]",sys.argv[i]);
	
print("param1: type= ", type(param1), ", value="+param1);

params = json.loads(param1.replace("'", "\""));
print("keys: ", params.keys());
print("outputPath: ", params['outputPath']);

params['diagramFileName']="abc";

print("params", params);

print("");



