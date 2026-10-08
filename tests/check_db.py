import sqlite3

c = sqlite3.connect("data/bank.db")
print(c.execute("SELECT * FROM retention_requests").fetchall())

##print(c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())