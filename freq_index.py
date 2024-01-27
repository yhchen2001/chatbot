import os
from uuid import uuid4
import random
import time
from dotenv import load_dotenv
from pinecone import Pinecone, PodSpec

class FreqIndex():
	def __init__(self, pc, index_name):
		self.namespace = 'freq'

		if index_name not in pc.list_indexes().names():
			pc.create_index(
				name=index_name,
				dimension=1536,
				metric='cosine',
				spec=PodSpec(
					environment='gcp-starter',
				)
			)
		while not pc.describe_index(index_name).status['ready']:
			continue
		self.index = pc.Index(index_name)

	def upsert(self, embedding, answer):
		count = self.index.describe_index_stats()['total_vector_count']

		self.index.upsert(vectors=[{
			'id': str(uuid4()),
			'values': embedding,
			'metadata': {
				'answer': answer,
			},
		}], namespace=self.namespace)

		while count < self.index.describe_index_stats()['total_vector_count']:
			continue

	def query(self, embedding, top_k=1):
		result = self.index.query(vector=embedding, top_k=top_k, namespace=self.namespace, include_metadata=True)['matches']
		# print(result)
		metadata, score = [res['metadata'] for res in result], [res['score'] for res in result]
		return metadata, score
		# return {'answer': metadata[0]['answer'], 'score': score[0]}

if __name__ == '__main__':
	load_dotenv()
	# fi = FreqIndex(Pinecone(os.getenv('PINECONE_API_KEY')), 'tsidgrp2')

	# fi.upsert_solution([random.uniform(-1, 1) for _ in range(1536)], 'd')
	# fi.upsert_solution([random.uniform(-1, 1) for _ in range(1536)], 'e')
	# fi.upsert_solution([random.uniform(-1, 1) for _ in range(1536)], 'f')

	print(fi.query([random.uniform(-1, 1) for _ in range(1536)]))