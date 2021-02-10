After setting up the virtual environment use [poetry](https://python-poetry.org/docs/) to install the project.



The `pyproject.toml` defines the project dependencies. Simply run poetry to install all required dependencies with:

```shell
poetry install
```

Afterwards you are ready to start the Flower server as well as the clients. You can simply start the server in a terminal as follows:

```shell
python3 server.py
```

Now you are ready to start the Flower clients which will participate in the learning. To do so simply open two more terminals and run the following command in each:

```shell
python3 client4.py --dataset dataset
```
modify the dataset path in the code
Alternatively you can run all of it in one shell as follows:

```shell
python3 server.py &
python3 client4.py --dataset dataset&
python3 client4.py --dataset dataset&
```


