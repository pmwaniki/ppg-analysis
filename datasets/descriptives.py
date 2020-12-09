from datasets.data import documents,data

for doc in documents.find():
    print(doc['admitted'])