
logLevel=4;
# 1-debug, 2-log, 3-warning, 4-error

def getLogLevel():
	if logLevel == 4:
		return "error level"
	elif logLevel == 3:
		return "warning level"
	elif logLevel == 2:
		return "log level"
	elif logLevel == 1:
		return "debug level"
	else:
		return "unknow level: ", logLevel
		
def myprint_error(*s):
	#if logLevel <= 4:
		print("Error - ", s)
		
def myprint_warn(*s):
	if logLevel <= 3:
		print("Warn - ", s)
		
def myprint_log(*s):
	if logLevel <= 2:
		print("Log - ", s)
		
def myprint_debug(*s):
	if logLevel <= 1:
		print("Debug - ", s)

#print("Hello world!");
#myprint_error(getLogLevel());
