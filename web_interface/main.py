if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method("spawn")

    from web_interface.aiohttp_local import run_aiohttp_server
    run_aiohttp_server()
