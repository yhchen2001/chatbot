from pymongo import MongoClient
from dotenv import load_dotenv
import os

class PyMonkey():
    def __init__(self):
        self.files = MongoClient(os.getenv('MONGODB_URL')).tsidgrp2.files
        # self.files.insert_one({'filename': '1', 'content': '111'})

    def insert(self, files):
        self.files.insert_many(files)

    def contain(self, name):
        return self.files.find_one({'name': name}) != None
    
    def delete(self, names):
        for name in names:
            self.files.delete_one({'name': name})

if __name__ == '__main__':
    load_dotenv()
    pm = PyMonkey()
    pm.insert([{'name': 'a', 'content': ''}, {'name': 'b', 'content': ''}])
    x = pm.contain('sss')
    print(x)
    pm.delete(['sss', ''])