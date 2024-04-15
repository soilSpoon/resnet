from hydra_zen import store


auto_name_store = store(name=lambda cfg: cfg.name)
