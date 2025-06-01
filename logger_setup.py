import logging
import sys
from datetime import datetime
import os 
log_filename = "log_file.log"
if os.path.exists(log_filename):
    os.remove(log_filename)
# Create a custom logger
logger = logging.getLogger("pump_functinality_prediction_logger") # This "pump_functinality_prediction_logger" is just a name for the logger, you can change it to whatever you want
if logger.hasHandlers():
    logger.handlers.clear()  # Remove all old handlers
logger.setLevel(logging.DEBUG)  # Set logger (main gatekeeper) level to DEBUG (low level) so that we can sset the other levell for handlers (rooms door keeper))

# Formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s") # Format log messages (how the message looks like)

# We want the info to appear both in log file and in console, so we will use two handlers:
# 1. File handler to write logs to a file
# Optional: delete the old log file at the start of each run

file_handler = logging.FileHandler(log_filename, mode='w')  # log file creation,'w' mode to overwrite the file each time
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)  # Set INFO or DEBUG based on your preference
console_handler.setFormatter(formatter)

# Add handlers to logger (only once to avoid duplicates)
logger.addHandler(file_handler)
logger.addHandler(console_handler)