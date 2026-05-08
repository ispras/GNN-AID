if __name__ == '__main__':
    # from web_interface.main_flask import run_flask_server
    # run_flask_server()

    from web_interface.main_aiohttp import run_aiohttp_server
    run_aiohttp_server()
