from src.app import App

config_path = "configs/config.json"

if __name__ == "__main__":
    app = App()
    app.run(config_path)