import os
from src.app import app, initialize_service
from src.service import FactCheckService
from cmn.config import Config

config_path = "configs/config.json"

if __name__ == '__main__':
    # Set the configuration before the app runs
    app.config['CONFIG_PATH'] = config_path  

    # Initialize the FactCheckService with the correct config path
    initialize_service()  

    # Run the app
    port = os.environ.get("PORT", 8976)
    app.run(host='0.0.0.0', port=int(port), debug=False)
