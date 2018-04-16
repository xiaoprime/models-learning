import re
import os
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

TESTCASE_FOLDER = config.get('PATH', 'TESTCASE_FOLDER')
DATA_FILE = config.get('PATH', 'DATA_FILE')
EXP_FILE = config.get('PATH', 'EXP_FILE')

pattern = config.get('MISC', 'pattern')

# clear
if os.path.exists(DATA_FILE):
	os.remove(DATA_FILE)
if os.path.exists(EXP_FILE):
	os.remove(EXP_FILE)

for file in os.listdir(TESTCASE_FOLDER):
	with open(TESTCASE_FOLDER+'/'+file,"r") as f:
		lines = f.readlines()

	with open(DATA_FILE,'a') as f:
		data = []
		for _line in lines:
			s = re.search(pattern, _line)
			if s:
				data.append(s.group(1))
		f.write(' '.join(data))
		f.write('\n')

	with open(EXP_FILE,'a') as f:
		f.write(file)
		f.write('\n')