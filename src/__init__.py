import os
import logging

logging.getLogger().setLevel(logging.INFO) # Pass down the tree
h = logging.StreamHandler()
h.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
h.setLevel(level=logging.INFO)
# No default handler (some modules won't see logger otherwise)
logging.getLogger().addHandler(h)

REPOSITORY_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
logging.info("Autodiscovery repository dir: " + REPOSITORY_DIR)

if os.path.exists("/media/lupoglaz"):
	logging.info("Server detected")
	storage_dir = "/media/lupoglaz"
	DATA_DIR = os.path.join(storage_dir, "ProteinsDataset/QA")

	MODELS_DIR = os.path.join(storage_dir, "LocalQA", "Models")
	if not os.path.exists(MODELS_DIR):
		os.mkdir(MODELS_DIR)

	LOG_DIR = os.path.join(storage_dir, "LocalQA", "Experiments")
	if not os.path.exists(LOG_DIR):
		os.mkdir(LOG_DIR)
	
	RESULTS_DIR = os.path.join(REPOSITORY_DIR, "results")
	if not os.path.exists(RESULTS_DIR):
		os.mkdir(RESULTS_DIR)
else:
	raise Exception("Unknown system")