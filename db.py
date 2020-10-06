import sqlite3


class LocalDB(object):
    """
    Minimalist SQLite wrapper to ensure the connection is closed upon program termination.
    """

    def __init__(self):
        self.conn = sqlite3.connect("alert.db")

    def __del__(self):
        self.conn.close()
