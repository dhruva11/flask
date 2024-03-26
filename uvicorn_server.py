from uvicorn import main

if __name__ == "__main__":
    main.run("app:covid_ct", host="0.0.0.0", port=8000)
