import logging
import os
from datetime import datetime

# Create a log file name with the current date and time (e.g., 07_01_2025_23_45_10.log)
LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
## Create a path for the logs folder + the log file (e.g., /your/path/logs/06_29_2025_23_45_10.log)
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
# Create the "logs" folder and its parent directories if they don't exist
os.makedirs(logs_path,exist_ok=True)
## Final full path for the log file
LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

## Configure logging with file location, formatting, and level
logging.basicConfig(
    filename=LOG_FILE_PATH, # Set the output file path for logging
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s", # Format for each log entry
    level=logging.INFO, # Log level: INFO and above (INFO, WARNING, ERROR, CRITICAL)


)
if __name__=="__main__":
    logging.info("Logging has started")