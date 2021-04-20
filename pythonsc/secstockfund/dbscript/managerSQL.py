import sys
import json

"""
Command:
python C:/fh/ws/ws1/fh-script/pythonsc/secstockfund/dbscript/managerSQL.py stockAnalysis
python C:/fh/ws/ws1/fh-script/pythonsc/secstockfund/dbscript/managerSQL.py dataUpdate
"""

rootPath = 'C:/fh/ws/ws1/fh-script/pythonsc/secstockfund/dbscript'
fileName = 'stockAnalysis'#Debug
fileName = sys.argv[1]
sqlFile = f'{rootPath}/{fileName}.sql'
jsonFile = f'{rootPath}/generate/{fileName}.js'

fo = open(sqlFile, mode='r', encoding='utf-8')
lines = fo.read().splitlines()
headers = [{'rownum': line[0], 'headline': line[1]} for line in enumerate(lines) if '#@#' in line[1]]
fo.close()

for i in range(len(headers)):
    currHeader = headers[i]
    nextHeader = headers[i+1] if i<len(headers)-1 else {'rownum':len(lines)}
    content = lines[currHeader['rownum'] : nextHeader['rownum']]
    headers[i]['content'] = content

fo=open(jsonFile, mode='w', encoding='utf-8')
fo.write('var sqlList = ')
json.dump(headers, fo, ensure_ascii=False)
fo.flush()
fo.close()

