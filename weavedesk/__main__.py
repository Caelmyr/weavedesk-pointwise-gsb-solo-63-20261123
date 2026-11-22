import os

from waitress import serve

from .web import create_app


def main():
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "19980"))
    threads = int(os.environ.get("WAITRESS_THREADS", "4"))
    app = create_app()
    print(f"WeaveDesk is listening on http://{host}:{port}", flush=True)
    serve(
        app,
        host=host,
        port=port,
        threads=threads,
        max_request_body_size=app.config["MAX_CONTENT_LENGTH"],
        channel_timeout=60,
        clear_untrusted_proxy_headers=True,
    )


if __name__ == "__main__":
    main()
