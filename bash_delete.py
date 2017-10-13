import time
import subprocess
while True:
    bashCommand = "rm -r /tmp/tflearn_logs/"

    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    time.sleep(100)