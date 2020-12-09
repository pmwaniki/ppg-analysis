

from pymongo import MongoClient
import pymongo
from bson.objectid import ObjectId




client=MongoClient('mongodb://%s:%s@127.0.0.1' % ("root", "example"))
triage=client["triage"]
ieee=client["ieee"]
#documents=db.original

def getTriageSegment(id):
    client = MongoClient('mongodb://%s:%s@127.0.0.1' % ("root", "example"),maxPoolSize=10000)
    triage = client["triage"]
    segments=triage.segments
    return get_by_id(id,segments)

def get_by_id(id,collection):
    return collection.find_one({"_id":ObjectId(id)})